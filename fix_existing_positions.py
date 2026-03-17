"""
fix_existing_positions.py
=========================
Recalculates and modifies the Take-Profit for any open position whose TP
gives a worse-than-configured R:R.

Run standalone:
    python3 fix_existing_positions.py --login 12345 --password pass --rr 3.0

Or import and call from within a running bot:
    from fix_existing_positions import fix_position_tps
    fix_position_tps(rr_ratio=3.0)

How it works:
  For each open position:
    stop_dist  = abs(price_open - sl)
    correct_tp = price_open + stop_dist * rr_ratio   (buy)
               = price_open - stop_dist * rr_ratio   (sell)

  If the existing TP gives a lower R:R than the configured ratio, the
  position is modified with the correct TP.
"""

import sys
import os

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("ERROR: pip install MetaTrader5")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

def _calc_correct_tp(price_open: float, sl: float, rr_ratio: float,
                     pos_type: int) -> float:
    """
    Compute correct TP so that reward/risk == rr_ratio.

    pos_type: 0 = BUY, 1 = SELL
    """
    stop_dist  = abs(price_open - sl)
    if pos_type == 0:   # BUY
        return price_open + stop_dist * rr_ratio
    else:               # SELL
        return price_open - stop_dist * rr_ratio


def _current_rr(price_open: float, sl: float, tp: float, pos_type: int) -> float:
    stop_dist   = abs(price_open - sl)
    reward_dist = abs(tp - price_open)
    if stop_dist <= 0:
        return 0.0
    return reward_dist / stop_dist


def modify_position_tp(ticket: int, new_sl: float, new_tp: float,
                       symbol: str) -> bool:
    """Send SLTP-modify request for an open position."""
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   symbol,
        "sl":       new_sl,
        "tp":       new_tp,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"  [ticket {ticket}] order_send returned None: {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  [ticket {ticket}] modify failed: {result.comment} (code {result.retcode})")
        return False
    return True


def fix_position_tps(
    rr_ratio: float = 3.0,
    dry_run: bool = False,
    min_improvement: float = 0.10,   # only fix if current R:R is this much below target
) -> None:
    """
    Scan all open positions and correct any TP that gives a sub-standard R:R.

    Parameters
    ----------
    rr_ratio        : the configured R:R target (e.g. 3.0 → 1:3)
    dry_run         : if True, print what WOULD change but don't send orders
    min_improvement : only modify if existing R:R < rr_ratio - min_improvement
                      (avoids unnecessary edits on positions already close to target)
    """
    if not MT5_AVAILABLE:
        print("MetaTrader5 not available")
        return

    positions = mt5.positions_get()
    if not positions:
        print("No open positions found")
        return

    print(f"\n{'='*62}")
    print(f"Position TP Fix  |  Target R:R 1:{rr_ratio}  |  "
          f"{'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*62}")

    fixed = 0
    skipped = 0

    for pos in positions:
        ticket     = pos.ticket
        symbol     = pos.symbol
        pos_type   = pos.type       # 0=BUY, 1=SELL
        price_open = pos.price_open
        sl         = pos.sl
        tp         = pos.tp
        profit     = pos.profit
        comment    = pos.comment

        type_str = "BUY " if pos_type == 0 else "SELL"

        # Cannot fix if no stop loss set
        if sl == 0.0:
            print(f"  {symbol:<12} {type_str} ticket={ticket}  "
                  f"SKIP – no stop loss set")
            skipped += 1
            continue

        stop_dist = abs(price_open - sl)
        if stop_dist <= 1e-8:
            print(f"  {symbol:<12} {type_str} ticket={ticket}  "
                  f"SKIP – stop distance ~0")
            skipped += 1
            continue

        current_rr = _current_rr(price_open, sl, tp, pos_type)
        correct_tp = _calc_correct_tp(price_open, sl, rr_ratio, pos_type)

        # Round to symbol's decimal places
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            point  = sym_info.point
            digits = sym_info.digits
            correct_tp = round(correct_tp, digits)
            sl_rounded = round(sl, digits)
        else:
            sl_rounded = sl

        needs_fix = current_rr < (rr_ratio - min_improvement)

        status = "FIX " if needs_fix else "OK  "
        print(f"  {symbol:<12} {type_str} @ {price_open:.5f}  "
              f"SL={sl:.5f}  TP={tp:.5f}  "
              f"R:R=1:{current_rr:.2f}  {status}", end="")

        if needs_fix:
            print(f"→ new TP={correct_tp:.5f}  (1:{rr_ratio})", end="")
            if not dry_run:
                ok = modify_position_tp(ticket, sl_rounded, correct_tp, symbol)
                print(f"  {'✓ DONE' if ok else '✗ FAILED'}")
                if ok:
                    fixed += 1
                else:
                    skipped += 1
            else:
                print("  [dry run – not sent]")
                fixed += 1
        else:
            print()
            skipped += 1

    print(f"\nSummary: {fixed} fixed, {skipped} skipped")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def init_mt5(login=None, password=None, server=None) -> bool:
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if login and password:
        kw = dict(login=login, password=password)
        if server:
            kw['server'] = server
        if not mt5.login(**kw):
            print(f"MT5 login failed: {mt5.last_error()}")
            return False
    info = mt5.account_info()
    if info is None:
        print("Could not get account info")
        return False
    print(f"Connected – account {info.login}  balance ${info.balance:,.2f}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix TP on open positions to match configured R:R")
    parser.add_argument("--login",    type=int,   default=None)
    parser.add_argument("--password", type=str,   default=None)
    parser.add_argument("--server",   type=str,   default=None)
    parser.add_argument("--rr",       type=float, default=3.0,
                        help="Target R:R ratio (default 3.0)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview changes without sending orders")
    args = parser.parse_args()

    if not init_mt5(args.login, args.password, args.server):
        sys.exit(1)

    try:
        fix_position_tps(rr_ratio=args.rr, dry_run=args.dry_run)
    finally:
        mt5.shutdown()