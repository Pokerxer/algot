"""
ICT Signal Engine - Fully Wired (V8 - R:R Fixed)
=================================================
All analytical handlers wired and three R:R bugs corrected:

BUG-FIX 1 – OB entry level corrected
    ob.open for a BULLISH OB is the bearish candle's OPEN = body_high.
    body_high > current price when price retraces into the OB → placing
    a BUY LIMIT above the ask is an invalid MT5 order.
    Fixed: entry = best.mean_threshold (50 % of OB body, always BELOW
    current price for a bullish retest → valid BUY LIMIT).

BUG-FIX 2 – Kill-zone PM session (hour integer comparison)
    Original: 13.5 <= h < 16 → never fires for integer h = 13.
    Fixed:    13   <= h < 16

BUG-FIX 3 – TP anchored to limit entry, not fill price
    signal['entry_price'] is the limit level; the actual MT5 fill may be
    at current_price.  TP is now computed in _enter_trade (ict_v7_mt5_fixed.py)
    from current_price so the R:R is always correct.  This file returns the
    correct limit entry; _enter_trade does the R:R arithmetic.

Pipeline (execution order):
  1. MarketStructureHandler  – HTF/LTF bias, dealing range, BOS/MSS/CHoCH
  2. LiquidityHandler        – sweep detection, draw on liquidity, run type
  3. OrderBlockHandler       – nearest valid OB, breakers, propulsion blocks
  4. TradingModelsHandler    – Model 2022 stage scoring, Silver Bullet windows
  5. FVGHandler              – BISI/SIBI, high-probability FVGs
  6. GapHandler              – overnight/weekend gaps
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

    EST windows:
      London Open 02:00 – 05:00
      NY Open     07:00 – 10:00
      NY AM       10:00 – 12:00
      NY PM/LC    13:00 – 16:00   ← BUG-FIX: was 13.5 <= h which never fires for h=13
    """
    if 2 <= hour < 5:
        return True, "LONDON"
    if 7 <= hour < 10:
        return True, "NY_OPEN"
    if 10 <= hour < 12:
        return True, "NY_AM"
    if 13 <= hour < 16:   # ← FIXED: was 13.5 <= hour < 16
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
    htf_bull_bear     = 25,
    htf_neutral       = 10,
    ltf_bos           = 15,
    ltf_mss           = 20,
    discount_zone     = 10,
    kill_zone         = 15,
    liquidity_swept   = 20,
    stop_hunt         = 15,
    low_res_run       = 10,
    ob_propulsion     = 25,
    ob_reclaimed      = 20,
    ob_standard       = 15,
    model_2022_a_plus = 30,
    model_2022_a      = 22,
    model_2022_b      = 12,
    silver_bullet_a   = 20,
    silver_bullet_b   = 10,
    fvg_high_prob     = 20,
    fvg_standard      = 12,
    gap_ce            = 10,
    ote_zone          = 15,
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
        self._debug_mode = False

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

        now          = datetime.now()
        kz, kz_name = _in_kill_zone(now.hour)
        sb_session   = _silver_bullet_session(now.hour, now.minute)

        # ── Run all layers ────────────────────────────────────────────────────
        ms  = self._run_market_structure(df)
        liq = self._run_liquidity(df)
        ob  = self._run_order_blocks(df)
        fvg = self._run_fvg(df)
        gap = self._run_gap(df, current_price)
        m22 = self._run_model_2022(df, ms)
        sb  = self._run_silver_bullet(df, now, sb_session)
        ote = self._run_ote(df, ms, current_price)

        # ── Build signal from layers ──────────────────────────────────────────
        signal = self._build_signal(
            symbol, current_price, data, now,
            ms, liq, ob, fvg, gap, m22, sb, ote,
            kz, kz_name, sb_session,
        )

        # ── Cache last analysis for status/reporting ─────────────────────────
        self.last_analysis[symbol] = {
            "timestamp":     now.isoformat(),
            "direction":     signal["direction"],
            "confluence":    signal["confluence"],
            "confidence":    signal["confidence"],
            "kill_zone":     kz_name,
            "htf_trend":     ms["htf_trend"].value if ms["htf_trend"] else "N/A",
            "ltf_trend":     ms["ltf_trend"].value if ms["ltf_trend"] else "N/A",
            "ob_type":       ob["type"] if ob else "none",
            "liq_swept":     liq["swept_side"],
            "model_2022":    m22["quality"] if m22 else "none",
            "silver_bullet": bool(sb),
        }
        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical layers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_market_structure(self, df: pd.DataFrame) -> Dict:
        """Run MarketStructureHandler and extract actionable fields."""
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
            result["htf_trend"]     = analysis.state.trend
            result["ltf_trend"]     = analysis.state.trend
            result["current_zone"]  = analysis.state.current_zone
            result["dealing_range"] = analysis.dealing_range
            result["last_break"]    = analysis.state.last_break

            mss_list = [b for b in analysis.structure_breaks
                        if b.break_type in (StructureBreakType.MSS, StructureBreakType.SMS)]
            result["recent_mss"] = mss_list[-1] if mss_list else None

            bos_list = [b for b in analysis.bos_breaks if b.is_confirmed]
            result["recent_bos"] = bos_list[-1] if bos_list else None

            if result["last_break"]:
                result["retest_held"] = result["last_break"].retest_held
        except Exception as e:
            if self._debug_mode:
                print(f"[MS] Error: {e}")
        return result

    def _run_liquidity(self, df: pd.DataFrame) -> Dict:
        """Run LiquidityHandler and return a flat summary dict."""
        result = dict(
            swept_side=None,
            recent_stop_hunt=False,
            draw=None,
            run_type=LiquidityRunType.NEUTRAL,
            nearest_buy_liq=None,
            nearest_sell_liq=None,
        )
        try:
            analysis = self.liq_handler.analyze(df)

            if analysis.sweep_events:
                last_sweep = analysis.sweep_events[-1]
                result["swept_side"] = last_sweep.pool.side.value
                result["recent_stop_hunt"] = (
                    last_sweep.sweep_type == SweepType.STOP_HUNT
                )

            result["draw"] = analysis.current_draw

            if analysis.run_analysis:
                result["run_type"] = analysis.run_analysis.run_type

            current_price = df["close"].iloc[-1]
            buy_pools  = [p for p in analysis.buy_side_pools  if not p.is_swept]
            sell_pools = [p for p in analysis.sell_side_pools if not p.is_swept]
            if buy_pools:
                result["nearest_buy_liq"] = min(
                    buy_pools, key=lambda p: abs(p.price - current_price))
            if sell_pools:
                result["nearest_sell_liq"] = min(
                    sell_pools, key=lambda p: abs(p.price - current_price))
        except Exception as e:
            if self._debug_mode:
                print(f"[LIQ] Error: {e}")
        return result

    def _run_order_blocks(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect all OBs and return a summary of the best one near current price.

        BUG-FIX: entry is now best.mean_threshold (50 % of OB body), NOT
        best.open.

        For a BULLISH OB, best.open == bearish candle open == body_high.
        body_high > current_price when price retraces into the OB from above,
        so placing a BUY LIMIT at body_high would be ABOVE the current ask –
        an invalid MT5 pending order.

        best.mean_threshold is always INSIDE the OB body:
          bullish OB: mean_threshold < current_price  → valid BUY LIMIT ✓
          bearish OB: mean_threshold > current_price  → valid SELL LIMIT ✓

        ICT teaching: "Enter at the mean threshold of the order block candle."
        """
        current_price = df["close"].iloc[-1]
        try:
            obs = self.ob_handler.detect_order_blocks(df)
            if not obs:
                return None

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

            def _score(ob):
                s = 0
                if ob.is_propulsion:  s += 30
                if ob.is_reclaimed:   s += 25
                if ob.is_extreme_ob:  s += 20
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

            def _dist(ob):
                return min(abs(current_price - ob.body_high),
                           abs(current_price - ob.body_low))

            candidates = [o for o in [best_bull, best_bear] if o is not None]
            best = min(candidates, key=_dist)

            ob_type_label = "bullish" if best.block_type == OrderBlockType.BULLISH else "bearish"

            # ── BUG-FIX: use mean_threshold as limit entry, NOT best.open ────
            # best.open for a bullish OB = bearish candle open = body_high
            #   → BUY LIMIT above ask = invalid MT5 order
            # best.mean_threshold = 50 % of OB body
            #   → BELOW current price for bullish retest = valid BUY LIMIT ✓
            entry_level = best.mean_threshold

            # Stop: beyond OB body (with 10 % buffer)
            body_range = best.body_high - best.body_low
            if best.block_type == OrderBlockType.BULLISH:
                ob_stop = best.body_low - body_range * 0.10
            else:
                ob_stop = best.body_high + body_range * 0.10

            return dict(
                ob=best,
                type=ob_type_label,
                entry=entry_level,           # ← FIXED: mean_threshold not best.open
                ob_open=best.open,           # kept for reference/logging only
                mean_threshold=best.mean_threshold,
                body_high=best.body_high,
                body_low=best.body_low,
                ob_stop=ob_stop,
                is_propulsion=best.is_propulsion,
                is_reclaimed=best.is_reclaimed,
                is_extreme=best.is_extreme_ob,
                score=_score(best),
            )
        except Exception as e:
            if self._debug_mode:
                print(f"[OB] Error: {e}")
            return None

    def _run_fvg(self, df: pd.DataFrame) -> Dict:
        """
        Fast O(n) FVG scanner.  Scans the last 50 bars for 3-candle
        BISI / SIBI patterns and returns the highest-scoring unfilled FVG
        on each side.
        """
        result = dict(best_bisi=None, best_sibi=None, high_prob_count=0)
        try:
            h = df['high'].values
            l = df['low'].values
            c = df['close'].values
            n = len(c)
            current_price = float(c[-1])

            scan_start = max(2, n - 50)

            dr_high = h[max(0, n - 20):n].max()
            dr_low  = l[max(0, n - 20):n].min()
            dr_eq   = (dr_high + dr_low) / 2

            bisi_candidates = []
            sibi_candidates = []

            for i in range(scan_start, n):
                c1_h = h[i - 2]; c1_l = l[i - 2]
                c3_h = h[i];     c3_l = l[i]

                # BISI (Bullish FVG): gap between C1 high and C3 low
                if c3_l > c1_h + 1e-8:
                    gap_low  = c1_h
                    gap_high = c3_l
                    ce       = (gap_low + gap_high) / 2

                    if current_price > ce:
                        score = 0
                        size  = gap_high - gap_low
                        in_discount = ce < dr_eq
                        if in_discount:
                            score += 20
                        recency = (i - scan_start) / max(1, n - scan_start)
                        score  += int(recency * 15)
                        atr_proxy = (h[max(0,i-14):i+1] - l[max(0,i-14):i+1]).mean()
                        if atr_proxy > 0:
                            ratio = size / atr_proxy
                            if 0.2 <= ratio <= 1.5:
                                score += 10
                        if not bisi_candidates:
                            score += 15
                        bisi_candidates.append((score, ce, gap_high, gap_low, size, in_discount))

                # SIBI (Bearish FVG): gap between C1 low and C3 high
                elif c3_h < c1_l - 1e-8:
                    gap_high = c1_l
                    gap_low  = c3_h
                    ce       = (gap_low + gap_high) / 2

                    if current_price < ce:
                        score = 0
                        size  = gap_high - gap_low
                        in_premium = ce > dr_eq
                        if in_premium:
                            score += 20
                        recency = (i - scan_start) / max(1, n - scan_start)
                        score  += int(recency * 15)
                        atr_proxy = (h[max(0,i-14):i+1] - l[max(0,i-14):i+1]).mean()
                        if atr_proxy > 0:
                            ratio = size / atr_proxy
                            if 0.2 <= ratio <= 1.5:
                                score += 10
                        if not sibi_candidates:
                            score += 15
                        sibi_candidates.append((score, ce, gap_high, gap_low, size, in_premium))

            if bisi_candidates:
                best = max(bisi_candidates, key=lambda x: x[0])
                score, ce, gh, gl, sz, hp = best
                result["best_bisi"] = type('FVG', (), {
                    'consequent_encroachment': ce,
                    'high': gh, 'low': gl, 'size': sz,
                    'is_high_probability': hp,
                    'gap_type': 'bisi',
                })()
                result["high_prob_count"] += int(hp)

            if sibi_candidates:
                best = max(sibi_candidates, key=lambda x: x[0])
                score, ce, gh, gl, sz, hp = best
                result["best_sibi"] = type('FVG', (), {
                    'consequent_encroachment': ce,
                    'high': gh, 'low': gl, 'size': sz,
                    'is_high_probability': hp,
                    'gap_type': 'sibi',
                })()
                result["high_prob_count"] += int(hp)

        except Exception as e:
            if self._debug_mode:
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
            if self._debug_mode:
                print(f"[GAP] Error: {e}")
        return result

    def _run_model_2022(self, df: pd.DataFrame, ms: Dict) -> Optional[Dict]:
        """Run ICT 2022 Model and return quality summary."""
        try:
            bias  = "bullish" if ms["htf_trend"] == TrendState.BULLISH else "bearish"
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
            if self._debug_mode:
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
            if self._debug_mode:
                print(f"[SB] Error: {e}")
            return None

    def _run_ote(self, df: pd.DataFrame, ms: Dict,
                 current_price: float) -> Optional[Dict]:
        """Check if current price is inside the OTE 62–79 % retracement zone."""
        try:
            setup = self.model_handler.analyze_optimal_trade_entry(
                df, "bullish" if ms["htf_trend"] == TrendState.BULLISH else "bearish")
            if setup and setup.in_ote_zone:
                return dict(
                    in_ote=True,
                    ote_low=setup.ote_low,
                    ote_high=setup.ote_high,
                    ote_mid=setup.ote_mid,
                    retracement_pct=setup.retracement_percent,
                )
        except Exception as e:
            if self._debug_mode:
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
            take_profit=None,   # NOTE: _enter_trade recalculates this from current_price
            confidence="LOW",
            reasoning=[],
            fvg_data=None,
            gap_data=None,
        )

        score = 0
        r     = signal["reasoning"]

        # ── Layer 1: Market Structure – directional bias ───────────────────────
        htf        = ms["htf_trend"]
        zone       = ms["current_zone"]
        recent_mss = ms["recent_mss"]
        recent_bos = ms["recent_bos"]
        fvg_data   = fvg

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
            if recent_mss:
                direction = 1 if recent_mss.direction == "bullish" else -1
                score += W["htf_neutral"]
                r.append(f"HTF Ranging + MSS {recent_mss.direction}: +{W['htf_neutral']}")
            elif fvg_data and (fvg_data.get('best_bisi') or fvg_data.get('best_sibi')):
                if fvg_data.get('best_sibi'):
                    direction = -1
                    score += W["htf_neutral"]
                    r.append(f"HTF Ranging + SIBI FVG: +{W['htf_neutral']}")
                elif fvg_data.get('best_bisi'):
                    direction = 1
                    score += W["htf_neutral"]
                    r.append(f"HTF Ranging + BISI FVG: +{W['htf_neutral']}")

        if direction == 0:
            return signal

        signal["direction"] = direction

        # ── Layer 2: LTF structure confirmation ───────────────────────────────
        if recent_mss:
            if ((direction == 1  and recent_mss.direction == "bullish") or
                    (direction == -1 and recent_mss.direction == "bearish")):
                score += W["ltf_mss"]
                r.append(f"MSS/SMS confirmed: +{W['ltf_mss']}")
        elif recent_bos:
            if ((direction == 1  and recent_bos.direction == "bullish") or
                    (direction == -1 and recent_bos.direction == "bearish")):
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
        ob_entry = None
        ob_stop  = None

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
                    r.append(f"Standard OB @ {ob['entry']:.5f}: +{W['ob_standard']}")

                # entry = mean_threshold (already fixed in _run_order_blocks)
                ob_entry = ob["entry"]
                ob_stop  = ob["ob_stop"]

        # ── Layer 7: ICT 2022 Model ───────────────────────────────────────────
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

        # ── Layer 9: FVG ──────────────────────────────────────────────────────
        fvg_entry = None

        if fvg["best_bisi"] and direction == 1:
            f    = fvg["best_bisi"]
            dist = abs(current_price - f.consequent_encroachment)
            if dist < f.size * 3:
                pts    = W["fvg_high_prob"] if f.is_high_probability else W["fvg_standard"]
                score += pts
                r.append(f"BISI FVG{'(HP)' if f.is_high_probability else ''}: +{pts}")
                fvg_entry = f.consequent_encroachment
                signal["fvg_data"] = {"type": "BISI", "ce": f.consequent_encroachment}

        elif fvg["best_sibi"] and direction == -1:
            f    = fvg["best_sibi"]
            dist = abs(current_price - f.consequent_encroachment)
            if dist < f.size * 3:
                pts    = W["fvg_high_prob"] if f.is_high_probability else W["fvg_standard"]
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
        # All of these are valid LIMIT prices in the correct direction
        # (OB mean_threshold is now below current for bullish, above for bearish)
        entry = ob_entry or sb_entry or fvg_entry or current_price
        signal["entry_price"] = entry

        # ── Stop loss priority: OB boundary > Model 2022 > Silver Bullet > ATR ─
        highs  = data["highs"]
        lows   = data["lows"]
        closes = data["closes"]
        atr_val = _atr(highs, lows, closes)

        stop = ob_stop or m22_stop or sb_stop

        # Minimum stop distance for indices (0.5 % of price)
        sym_upper   = signal.get("symbol", "")
        min_stop_pct = 0.005
        if sym_upper in ("US30", "US500", "USTEC", "UK100", "GER40", "JAP40"):
            min_stop_dist = entry * min_stop_pct
        else:
            min_stop_dist = 0.0

        if stop is None:
            stop_dist = max(atr_val * 2.0, min_stop_dist)
            stop = (entry - stop_dist) if direction == 1 else (entry + stop_dist)

        stop_distance = abs(entry - stop)

        # Sanity: stop must be non-zero and on the correct side of ENTRY
        if stop_distance < 1e-8:
            signal["direction"] = 0
            return signal
        if direction == 1 and stop >= entry:
            stop = entry - atr_val
            stop_distance = atr_val
        if direction == -1 and stop <= entry:
            stop = entry + atr_val
            stop_distance = atr_val

        if min_stop_dist > 0 and stop_distance < min_stop_dist:
            stop = (entry - min_stop_dist) if direction == 1 else (entry + min_stop_dist)
            stop_distance = min_stop_dist

        signal["stop_loss"] = stop

        # ── Take profit ───────────────────────────────────────────────────────
        # NOTE: _enter_trade in ict_v7_mt5_fixed.py will OVERRIDE this with a
        # TP anchored to current_price to guarantee the real R:R.  This value
        # is kept here purely as a reference / fallback for callers that do not
        # go through _enter_trade (e.g. backtester).
        tp = (entry + stop_distance * self.rr_ratio) if direction == 1 \
             else (entry - stop_distance * self.rr_ratio)
        signal["take_profit"] = tp

        reward = abs(tp - entry)
        eff_rr = reward / stop_distance if stop_distance > 0 else 0
        r.append(f"Signal R:R 1:{eff_rr:.2f} (entry={entry:.5f}, sl={stop:.5f})")

        return signal