"""
ICT Signal Engine - Fully Wired (V8)
=====================================
Replaces the hollow V7SignalGenerator with a pipeline that actually calls
every handler that was already written but never used.

Pipeline (in execution order):
  1. MarketStructureHandler  – real HTF & LTF bias, dealing range, BOS/MSS/CHoCH
  2. LiquidityHandler        – sweep detection, draw on liquidity, run type
  3. OrderBlockHandler       – nearest valid OB, breakers, propulsion blocks
  4. TradingModelsHandler    – Model 2022 stage scoring, Silver Bullet windows
  5. FVGHandler              – BISI/SIBI, high-probability FVGs  (unchanged)
  6. GapHandler              – overnight/weekend gaps             (unchanged)

Kill-zone fix:
  Original code used  `13.5 <= h < 16`  – since `h` is an integer this
  *never* fires for hour 13 (the PM session start), silently dropping every
  trade from 13:00–14:00 EST.  Fixed to `13 <= h < 16`.

Drop-in usage (in ict_v7_mt5_fixed.py):
  Replace:
      from <wherever> import V7SignalGenerator
  With:
      from signal_engine import ICTSignalEngine as V7SignalGenerator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Core handlers ─────────────────────────────────────────────────────────────
from fvg_handler import FVGHandler
from gap_handler import GapHandler
from market_structure_handler import (
    MarketStructureHandler,
    StructureBreakType,
    TrendState,
    PriceZone,
)
from liquidity_handler import (
    LiquidityHandler,
    LiquiditySide,
    LiquidityRunType,
    SweepType,
)
from order_block_handler import (
    OrderBlockHandler,
    OrderBlockType,
    OrderBlockStatus,
)
from trading_model_handler import (
    TradingModelsHandler,
    SilverBulletSession,
    SetupQuality,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    """14-period ATR at the last bar."""
    n = len(closes)
    start = max(0, n - period - 1)
    trs = []
    for i in range(start + 1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else float(highs[-1] - lows[-1])


def _in_kill_zone(hour: int) -> Tuple[bool, str]:
    """
    Returns (is_kill_zone, session_name).

    EST windows (MT5 server time may differ – adjust if needed):
      Asian       00:00 – 04:00   (servers usually UTC+2/3, adapt accordingly)
      London Open 02:00 – 05:00
      NY Open     07:00 – 10:00
      NY AM       09:30 – 12:00
      NY PM/LC    13:00 – 16:00   ← was broken: 13.5 <= h never fires for h=13
    """
    if 2 <= hour < 5:
        return True, "LONDON"
    if 7 <= hour < 10:
        return True, "NY_OPEN"
    if 10 <= hour < 12:
        return True, "NY_AM"
    if 13 <= hour < 16:           # ← FIX: was 13.5 <= hour < 16
        return True, "NY_PM"
    return False, "OFF_HOURS"


def _silver_bullet_session(hour: int, minute: int) -> Optional[SilverBulletSession]:
    """Map current EST hour:minute to the active Silver Bullet window, or None."""
    t = hour * 60 + minute
    if 3 * 60 <= t < 4 * 60:
        return SilverBulletSession.LONDON
    if 10 * 60 <= t < 11 * 60:
        return SilverBulletSession.NY_AM
    if 14 * 60 <= t < 15 * 60:
        return SilverBulletSession.NY_PM
    if 20 * 60 <= t < 21 * 60:
        return SilverBulletSession.ASIAN
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal quality weights  (total theoretical max ≈ 200, capped at 100)
# ─────────────────────────────────────────────────────────────────────────────
W = dict(
    htf_bull_bear     = 25,   # HTF BOS/MSS in trade direction
    htf_neutral       = 10,   # HTF ranging but LTF aligned
    ltf_bos           = 15,   # LTF BOS continuation
    ltf_mss           = 20,   # LTF MSS / SMS reversal
    discount_zone     = 10,   # Price in discount (longs) / premium (shorts)
    kill_zone         = 15,   # Inside a kill zone
    liquidity_swept   = 20,   # Opposing liquidity swept before entry
    stop_hunt         = 15,   # Stop-hunt reversal confirmed
    low_res_run       = 10,   # Low-resistance liquidity run
    ob_propulsion     = 25,   # Propulsion / extreme OB at entry
    ob_reclaimed      = 20,   # Reclaimed OB (tested & held)
    ob_standard       = 15,   # Untested/tested OB near price
    model_2022_a_plus = 30,   # Model 2022 A+ quality
    model_2022_a      = 22,   # Model 2022 A quality
    model_2022_b      = 12,   # Model 2022 B quality
    silver_bullet_a   = 20,   # Silver Bullet A/A+ in window
    silver_bullet_b   = 10,   # Silver Bullet B
    fvg_high_prob     = 20,   # High-probability FVG
    fvg_standard      = 12,   # Standard FVG near price
    gap_ce            = 10,   # At gap CE level
    ote_zone          = 15,   # Inside OTE 62–79% retracement
)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ICTSignalEngine:
    """
    Fully-wired ICT signal generator.

    All analytical layers are instantiated once and reused.  Each call to
    ``analyze_symbol`` runs the full pipeline on the latest data slice.
    """

    def __init__(self, rr_ratio: float = 3.0):
        self.rr_ratio = rr_ratio

        # ── Analytical layers ────────────────────────────────────────────────
        self.ms_handler = MarketStructureHandler(
            swing_lookback=5,
            min_displacement_pct=0.1,
        )
        self.liq_handler = LiquidityHandler(
            equal_threshold_pips=5.0,
            min_touches=2,
            lookback_bars=50,
            displacement_threshold_pct=0.3,
            stop_hunt_reversal_pct=0.5,
        )
        self.ob_handler = OrderBlockHandler(
            displacement_threshold=0.0005,
            min_body_size=0.0001,
            max_consecutive_candles=5,
            track_body_respect=True,
        )
        self.model_handler = TradingModelsHandler()
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=True,
            detect_volume_imbalances=True,
            detect_suspension_blocks=True,
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3,
        )

        # ── Per-symbol state ─────────────────────────────────────────────────
        self.last_analysis: Dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_symbol(self, symbol: str, data: Dict, current_price: float) -> Dict:
        """
        Run the full ICT analysis pipeline for one symbol.

        Parameters
        ----------
        symbol       : instrument name (e.g. "EURUSD")
        data         : dict returned by ``prepare_data_mt5()``
        current_price: live bid price from MT5

        Returns
        -------
        signal dict compatible with V7MT5LiveTrader._enter_trade()
        """
        df = pd.DataFrame({
            "open":  data["opens"],
            "high":  data["highs"],
            "low":   data["lows"],
            "close": data["closes"],
        })

        now        = datetime.now()
        kz, kz_name = _in_kill_zone(now.hour)
        sb_session = _silver_bullet_session(now.hour, now.minute)

        # ── Run all layers ────────────────────────────────────────────────────
        ms   = self._run_market_structure(df)
        liq  = self._run_liquidity(df)
        ob   = self._run_order_blocks(df)
        fvg  = self._run_fvg(df)
        gap  = self._run_gap(df, current_price)
        m22  = self._run_model_2022(df, ms)
        sb   = self._run_silver_bullet(df, now, sb_session)
        ote  = self._run_ote(df, ms, current_price)

        # ── Build signal from layers ──────────────────────────────────────────
        signal = self._build_signal(
            symbol, current_price, data, now,
            ms, liq, ob, fvg, gap, m22, sb, ote,
            kz, kz_name, sb_session,
        )

        # ── Cache last analysis for status/reporting ─────────────────────────
        self.last_analysis[symbol] = {
            "timestamp":   now.isoformat(),
            "direction":   signal["direction"],
            "confluence":  signal["confluence"],
            "confidence":  signal["confidence"],
            "kill_zone":   kz_name,
            "htf_trend":   ms["htf_trend"].value if ms["htf_trend"] else "N/A",
            "ltf_trend":   ms["ltf_trend"].value if ms["ltf_trend"] else "N/A",
            "ob_type":     ob["type"] if ob else "none",
            "liq_swept":   liq["swept_side"],
            "model_2022":  m22["quality"] if m22 else "none",
            "silver_bullet": bool(sb),
        }
        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical layers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Run MarketStructureHandler and extract actionable fields.

        Returns a summary dict so the rest of the pipeline never has to
        inspect the raw MarketStructureAnalysis dataclass directly.
        """
        result = dict(
            htf_trend=None,
            ltf_trend=None,
            current_zone=None,
            dealing_range=None,
            last_break=None,
            recent_mss=None,
            recent_bos=None,
            retest_held=False,
        )
        try:
            analysis = self.ms_handler.analyze(df)
            result["htf_trend"]    = analysis.state.trend
            result["ltf_trend"]    = analysis.state.trend          # same TF here; swap for MTF
            result["current_zone"] = analysis.state.current_zone
            result["dealing_range"]= analysis.dealing_range
            result["last_break"]   = analysis.state.last_break

            # Most recent MSS / SMS (confirmed reversal)
            mss_list = [b for b in analysis.structure_breaks
                        if b.break_type in (StructureBreakType.MSS, StructureBreakType.SMS)]
            result["recent_mss"] = mss_list[-1] if mss_list else None

            # Most recent confirmed BOS (continuation)
            bos_list = [b for b in analysis.bos_breaks if b.is_confirmed]
            result["recent_bos"] = bos_list[-1] if bos_list else None

            # Did the last break get retested and held?
            if result["last_break"]:
                result["retest_held"] = result["last_break"].retest_held
        except Exception as e:
            print(f"[MS] Error: {e}")
        return result

    def _run_liquidity(self, df: pd.DataFrame) -> Dict:
        """Run LiquidityHandler and return a flat summary dict."""
        result = dict(
            swept_side=None,          # "buy_side" / "sell_side" / None
            recent_stop_hunt=False,
            draw=None,                # LiquidityPool or None
            run_type=LiquidityRunType.NEUTRAL,
            nearest_buy_liq=None,     # nearest intact buy-side pool
            nearest_sell_liq=None,    # nearest intact sell-side pool
        )
        try:
            analysis = self.liq_handler.analyze(df)

            # Most recent sweep
            if analysis.sweep_events:
                last_sweep = analysis.sweep_events[-1]
                result["swept_side"] = last_sweep.pool.side.value

                # Stop hunt = sweep + reversal within a few bars
                result["recent_stop_hunt"] = (
                    last_sweep.sweep_type == SweepType.STOP_HUNT
                )

            result["draw"] = analysis.current_draw

            if analysis.run_analysis:
                result["run_type"] = analysis.run_analysis.run_type

            # Nearest intact liquidity pools on each side
            current_price = df["close"].iloc[-1]
            buy_pools = [p for p in analysis.buy_side_pools if not p.is_swept]
            sell_pools = [p for p in analysis.sell_side_pools if not p.is_swept]
            if buy_pools:
                result["nearest_buy_liq"] = min(buy_pools,
                                                key=lambda p: abs(p.price - current_price))
            if sell_pools:
                result["nearest_sell_liq"] = min(sell_pools,
                                                 key=lambda p: abs(p.price - current_price))
        except Exception as e:
            print(f"[LIQ] Error: {e}")
        return result

    def _run_order_blocks(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect all OBs and return a summary of the best one near current price.
        Returns None if no tradeable OB is found.
        """
        current_price = df["close"].iloc[-1]
        try:
            obs = self.ob_handler.detect_order_blocks(df)
            if not obs:
                return None

            # Filter to active (not failed/invalidated) OBs within 2× ATR
            atr_val = _atr(df["high"].values, df["low"].values, df["close"].values)
            window  = atr_val * 2

            bull_obs = [o for o in obs
                        if o.block_type == OrderBlockType.BULLISH
                        and o.status not in (OrderBlockStatus.FAILED,
                                             OrderBlockStatus.INVALIDATED)
                        and o.body_low <= current_price <= o.body_high + window]

            bear_obs = [o for o in obs
                        if o.block_type == OrderBlockType.BEARISH
                        and o.status not in (OrderBlockStatus.FAILED,
                                             OrderBlockStatus.INVALIDATED)
                        and o.body_low - window <= current_price <= o.body_high]

            # Score and pick best per side
            def _score(ob):
                s = 0
                if ob.is_propulsion:       s += 30
                if ob.is_reclaimed:        s += 25
                if ob.is_extreme_ob:       s += 20
                s += ob.strength.value * 5
                total = ob.body_respected + ob.body_violated
                if total:
                    s += int(ob.body_respected / total * 15)
                s += min(ob.wick_only_tests * 5, 15)
                return s

            best_bull = max(bull_obs, key=_score) if bull_obs else None
            best_bear = max(bear_obs, key=_score) if bear_obs else None

            if not best_bull and not best_bear:
                return None

            # Return whichever is closer
            def _dist(ob):
                return min(abs(current_price - ob.body_high),
                           abs(current_price - ob.body_low))

            candidates = [o for o in [best_bull, best_bear] if o is not None]
            best = min(candidates, key=_dist)

            ob_type_label = "bullish" if best.block_type == OrderBlockType.BULLISH else "bearish"
            return dict(
                ob=best,
                type=ob_type_label,
                entry=best.open,           # ICT entry = opening price of OB candle
                mean_threshold=best.mean_threshold,
                body_high=best.body_high,
                body_low=best.body_low,
                is_propulsion=best.is_propulsion,
                is_reclaimed=best.is_reclaimed,
                is_extreme=best.is_extreme_ob,
                score=_score(best),
            )
        except Exception as e:
            print(f"[OB] Error: {e}")
            return None

    def _run_fvg(self, df: pd.DataFrame) -> Dict:
        """Run FVGHandler and return a flat summary."""
        result = dict(
            best_bisi=None,
            best_sibi=None,
            high_prob_count=0,
        )
        try:
            analysis = self.fvg_handler.analyze_fvgs(df)
            result["best_bisi"]      = analysis.best_bisi_fvg
            result["best_sibi"]      = analysis.best_sibi_fvg
            result["high_prob_count"]= len(analysis.high_prob_fvgs)
        except Exception as e:
            print(f"[FVG] Error: {e}")
        return result

    def _run_gap(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Run GapHandler and return a flat summary."""
        result = dict(in_gap=False, at_ce=False, gap_type=None)
        try:
            analysis = self.gap_handler.analyze(df, current_price)
            result["in_gap"] = analysis.in_gap_zone
            if analysis.current_gap and analysis.current_gap.quadrants:
                ce = analysis.current_gap.quadrants.ce
                result["at_ce"] = abs(current_price - ce) < (
                    analysis.current_gap.quadrants.range_size * 0.1)
            if analysis.current_gap:
                result["gap_type"] = analysis.current_gap.gap_type.value
        except Exception as e:
            print(f"[GAP] Error: {e}")
        return result

    def _run_model_2022(self, df: pd.DataFrame, ms: Dict) -> Optional[Dict]:
        """Run ICT 2022 Model and return quality summary."""
        try:
            bias = "bullish" if ms["htf_trend"] == TrendState.BULLISH else "bearish"
            setup = self.model_handler.analyze_model_2022(df, bias, datetime.now())
            if setup is None:
                return None
            return dict(
                quality=setup.quality.value,
                favorability=setup.favorability,
                direction="long" if setup.direction == "long" else "short",
                in_ote=setup.in_ote,
                stage_4_entry=setup.stage_4_entry,
                target_1=setup.target_1,
                target_2=setup.target_2,
                stop_loss=setup.stop_loss,
                notes=setup.notes,
            )
        except Exception as e:
            print(f"[M22] Error: {e}")
            return None

    def _run_silver_bullet(self, df: pd.DataFrame, now: datetime,
                            session: Optional[SilverBulletSession]) -> Optional[Dict]:
        """Check for Silver Bullet setup in the active session window."""
        if session is None:
            return None
        try:
            setup = self.model_handler.analyze_silver_bullet(df, now, session)
            if setup is None:
                return None
            return dict(
                session=setup.session.value,
                direction=setup.direction,
                quality=setup.quality.value,
                favorability=setup.favorability,
                entry=setup.entry_price,
                stop_loss=setup.stop_loss,
                target=setup.target,
            )
        except Exception as e:
            print(f"[SB] Error: {e}")
            return None

    def _run_ote(self, df: pd.DataFrame, ms: Dict,
                 current_price: float) -> Optional[Dict]:
        """Check if current price is inside the OTE 62–79% retracement zone."""
        try:
            setup = self.model_handler.analyze_ote(df, "bullish"
                    if ms["htf_trend"] == TrendState.BULLISH else "bearish")
            if setup and setup.in_ote_zone:
                return dict(
                    in_ote=True,
                    ote_low=setup.ote_low,
                    ote_high=setup.ote_high,
                    ote_mid=setup.ote_mid,
                    retracement_pct=setup.retracement_percent,
                )
        except Exception as e:
            print(f"[OTE] Error: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Signal assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        symbol: str,
        current_price: float,
        data: Dict,
        now: datetime,
        ms: Dict,
        liq: Dict,
        ob: Optional[Dict],
        fvg: Dict,
        gap: Dict,
        m22: Optional[Dict],
        sb: Optional[Dict],
        ote: Optional[Dict],
        kz: bool,
        kz_name: str,
        sb_session: Optional[SilverBulletSession],
    ) -> Dict:

        signal: Dict = dict(
            symbol=symbol,
            direction=0,
            confluence=0,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            confidence="LOW",
            reasoning=[],
            fvg_data=None,
            gap_data=None,
        )

        score = 0
        r = signal["reasoning"]   # shorthand

        # ── Layer 1: Market Structure – determine directional bias ─────────────
        htf = ms["htf_trend"]
        zone = ms["current_zone"]
        recent_mss = ms["recent_mss"]
        recent_bos = ms["recent_bos"]

        direction = 0   # +1 = long, -1 = short

        if htf == TrendState.BULLISH:
            direction = 1
            score += W["htf_bull_bear"]
            r.append(f"HTF Bullish: +{W['htf_bull_bear']}")
        elif htf == TrendState.BEARISH:
            direction = -1
            score += W["htf_bull_bear"]
            r.append(f"HTF Bearish: +{W['htf_bull_bear']}")
        elif htf == TrendState.RANGING:
            # Still tradeable if a strong MSS just fired
            if recent_mss:
                direction = 1 if recent_mss.direction == "bullish" else -1
                score += W["htf_neutral"]
                r.append(f"HTF Ranging + MSS {recent_mss.direction}: +{W['htf_neutral']}")

        if direction == 0:
            # No bias – no trade
            return signal

        signal["direction"] = direction

        # ── Layer 2: LTF structure confirmation ───────────────────────────────
        if recent_mss:
            if (direction == 1 and recent_mss.direction == "bullish") or \
               (direction == -1 and recent_mss.direction == "bearish"):
                score += W["ltf_mss"]
                r.append(f"MSS/SMS confirmed: +{W['ltf_mss']}")
        elif recent_bos:
            if (direction == 1 and recent_bos.direction == "bullish") or \
               (direction == -1 and recent_bos.direction == "bearish"):
                score += W["ltf_bos"]
                r.append(f"BOS confirmed: +{W['ltf_bos']}")

        # ── Layer 3: Premium / Discount ───────────────────────────────────────
        if zone == PriceZone.DISCOUNT and direction == 1:
            score += W["discount_zone"]
            r.append(f"Discount zone (longs): +{W['discount_zone']}")
        elif zone == PriceZone.PREMIUM and direction == -1:
            score += W["discount_zone"]
            r.append(f"Premium zone (shorts): +{W['discount_zone']}")

        # ── Layer 4: Kill zone ────────────────────────────────────────────────
        if kz:
            score += W["kill_zone"]
            r.append(f"Kill zone ({kz_name}): +{W['kill_zone']}")

        # ── Layer 5: Liquidity ────────────────────────────────────────────────
        swept = liq["swept_side"]

        # A buy-side sweep before a short, or sell-side sweep before a long,
        # confirms that inducement is complete – the textbook ICT entry condition.
        if swept == "buy_side"  and direction == -1:
            score += W["liquidity_swept"]
            r.append(f"Buy-side liquidity swept (SHORT setup): +{W['liquidity_swept']}")
        elif swept == "sell_side" and direction == 1:
            score += W["liquidity_swept"]
            r.append(f"Sell-side liquidity swept (LONG setup): +{W['liquidity_swept']}")

        if liq["recent_stop_hunt"]:
            score += W["stop_hunt"]
            r.append(f"Stop hunt / Turtle Soup confirmed: +{W['stop_hunt']}")

        if liq["run_type"] == LiquidityRunType.LOW_RESISTANCE:
            score += W["low_res_run"]
            r.append(f"Low-resistance liquidity run: +{W['low_res_run']}")

        # ── Layer 6: Order Block ──────────────────────────────────────────────
        ob_entry   = None
        ob_stop    = None

        if ob:
            ob_dir = 1 if ob["type"] == "bullish" else -1
            if ob_dir == direction:
                if ob["is_propulsion"] or ob["is_extreme"]:
                    score += W["ob_propulsion"]
                    r.append(f"Propulsion/Extreme OB: +{W['ob_propulsion']}")
                elif ob["is_reclaimed"]:
                    score += W["ob_reclaimed"]
                    r.append(f"Reclaimed OB: +{W['ob_reclaimed']}")
                else:
                    score += W["ob_standard"]
                    r.append(f"Standard OB: +{W['ob_standard']}")

                # Use OB opening price as entry (ICT: change in state of delivery)
                ob_entry = ob["entry"]

                # Stop: beyond OB body (with 10 % buffer)
                body_range = ob["body_high"] - ob["body_low"]
                if direction == 1:
                    ob_stop = ob["body_low"] - body_range * 0.1
                else:
                    ob_stop = ob["body_high"] + body_range * 0.1

        # ── Layer 7: ICT 2022 Model ───────────────────────────────────────────
        m22_entry  = None
        m22_stop   = None
        m22_target = None

        if m22 and ((m22["direction"] == "long") == (direction == 1)):
            q = m22["quality"]
            if q == "A+":
                score += W["model_2022_a_plus"]
                r.append(f"Model 2022 A+: +{W['model_2022_a_plus']}")
            elif q == "A":
                score += W["model_2022_a"]
                r.append(f"Model 2022 A: +{W['model_2022_a']}")
            elif q == "B":
                score += W["model_2022_b"]
                r.append(f"Model 2022 B: +{W['model_2022_b']}")

            m22_stop   = m22["stop_loss"]
            m22_target = m22["target_1"]

        # ── Layer 8: Silver Bullet ────────────────────────────────────────────
        sb_entry  = None
        sb_stop   = None
        sb_target = None

        if sb and ((sb["direction"] == "long") == (direction == 1)):
            q = sb["quality"]
            if q in ("A+", "A"):
                score += W["silver_bullet_a"]
                r.append(f"Silver Bullet {sb['session']} ({q}): +{W['silver_bullet_a']}")
            else:
                score += W["silver_bullet_b"]
                r.append(f"Silver Bullet {sb['session']} ({q}): +{W['silver_bullet_b']}")

            sb_entry  = sb["entry"]
            sb_stop   = sb["stop_loss"]
            sb_target = sb["target"]

        # ── Layer 9: FVG ─────────────────────────────────────────────────────
        fvg_entry = None

        if fvg["best_bisi"] and direction == 1:
            f = fvg["best_bisi"]
            dist = abs(current_price - f.consequent_encroachment)
            if dist < f.size * 3:
                pts = W["fvg_high_prob"] if f.is_high_probability else W["fvg_standard"]
                score += pts
                r.append(f"BISI FVG{'(HP)' if f.is_high_probability else ''}: +{pts}")
                fvg_entry = f.consequent_encroachment
                signal["fvg_data"] = {"type": "BISI", "ce": f.consequent_encroachment}

        elif fvg["best_sibi"] and direction == -1:
            f = fvg["best_sibi"]
            dist = abs(current_price - f.consequent_encroachment)
            if dist < f.size * 3:
                pts = W["fvg_high_prob"] if f.is_high_probability else W["fvg_standard"]
                score += pts
                r.append(f"SIBI FVG{'(HP)' if f.is_high_probability else ''}: +{pts}")
                fvg_entry = f.consequent_encroachment
                signal["fvg_data"] = {"type": "SIBI", "ce": f.consequent_encroachment}

        # ── Layer 10: Gap ─────────────────────────────────────────────────────
        if gap["in_gap"]:
            r.append(f"In {gap['gap_type']} gap zone")
            if gap["at_ce"]:
                score += W["gap_ce"]
                r.append(f"At gap CE: +{W['gap_ce']}")
                signal["gap_data"] = {"type": gap["gap_type"]}

        # ── Layer 11: OTE zone ────────────────────────────────────────────────
        if ote and ote["in_ote"]:
            score += W["ote_zone"]
            r.append(f"OTE zone ({ote['retracement_pct']:.0f}%): +{W['ote_zone']}")

        # ── Cap and classify ──────────────────────────────────────────────────
        score = min(score, 100)
        signal["confluence"] = score

        if score >= 80:
            signal["confidence"] = "HIGH"
        elif score >= 65:
            signal["confidence"] = "MEDIUM"
        elif score >= 50:
            signal["confidence"] = "LOW"
        else:
            signal["direction"] = 0
            return signal

        # ── Entry price priority: OB > Silver Bullet > FVG > current ─────────
        # ICT principle: enter at the PD array, not at market
        entry = ob_entry or sb_entry or fvg_entry or current_price
        signal["entry_price"] = entry

        # ── Stop loss priority: OB > Model 2022 > Silver Bullet > ATR ────────
        highs  = data["highs"]
        lows   = data["lows"]
        closes = data["closes"]
        atr_val = _atr(highs, lows, closes)

        stop = ob_stop or m22_stop or sb_stop

        if stop is None:
            # ATR fallback: 2× ATR from entry
            stop_dist = atr_val * 2.0
            stop = (entry - stop_dist) if direction == 1 else (entry + stop_dist)

        stop_distance = abs(entry - stop)

        # Sanity: stop must be on the correct side and non-zero
        if stop_distance < 1e-8:
            signal["direction"] = 0
            return signal
        if direction == 1 and stop >= entry:
            stop = entry - atr_val
            stop_distance = atr_val
        if direction == -1 and stop <= entry:
            stop = entry + atr_val
            stop_distance = atr_val

        signal["stop_loss"] = stop

        # ── Take profit: Model 2022 target > Silver Bullet > R:R multiple ─────
        target = m22_target or sb_target
        if target is None or abs(target - entry) < stop_distance:
            # Ensure at minimum the configured R:R
            target = (entry + stop_distance * self.rr_ratio) if direction == 1 \
                     else (entry - stop_distance * self.rr_ratio)

        # Verify TP is beyond configured R:R; extend if model target is farther
        min_tp = (entry + stop_distance * self.rr_ratio) if direction == 1 \
                 else (entry - stop_distance * self.rr_ratio)

        if direction == 1:
            signal["take_profit"] = max(target, min_tp)
        else:
            signal["take_profit"] = min(target, min_tp)

        # Log effective R:R for transparency
        reward = abs(signal["take_profit"] - entry)
        eff_rr = reward / stop_distance if stop_distance > 0 else 0
        r.append(f"R:R 1:{eff_rr:.2f} (configured 1:{self.rr_ratio})")

        return signal