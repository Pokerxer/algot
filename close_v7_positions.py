"""
close_v7_positions.py
=====================
Closes all open positions tagged "ICT V7" and leaves every "ICT V8"
pending order exactly as-is.

Run:
    python3 close_v7_positions.py --login 12345 --password yourpass
    python3 close_v7_positions.py --login 12345 --password yourpass --dry-run
"""

import sys
import argparse

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: pip install MetaTrader5")
    sys.exit(1)


V7_TAG = "ICT V7"
V8_TAG = "ICT V8"


def init_mt5(login=None, password=None, server=None) -> bool:
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if login and password:
        kw = dict(login=login, password=password)
        if server:
            kw["server"] = server
        if not mt5.login(**kw):
            print(f"MT5 login failed: {mt5.last_error()}")
            return False
    info = mt5.account_info()
    if info is None:
        print("Could not get account info")
        return False
    print(f"Connected – account {info.login}  balance ${info.balance:,.2f}\n")
    return True


def close_position(pos, dry_run: bool) -> bool:
    """Market-close a single open position."""
    symbol   = pos.symbol
    ticket   = pos.ticket
    volume   = pos.volume
    pos_type = pos.type   # 0=BUY, 1=SELL

    # Closing a BUY = SELL at bid; closing a SELL = BUY at ask
    close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
    sym_info   = mt5.symbol_info(symbol)
    if sym_info is None:
        print(f"  [ticket {ticket}] Could not get symbol info for {symbol}")
        return False

    close_price = sym_info.bid if pos_type == 0 else sym_info.ask
    type_str    = "BUY" if pos_type == 0 else "SELL"

    print(f"  Closing {symbol} {type_str} x{volume} @ {pos.price_open:.5f}  "
          f"comment='{pos.comment}'  profit=${pos.profit:.2f}", end="")

    if dry_run:
        print("  [DRY RUN – not sent]")
        return True

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       volume,
        "type":         close_type,
        "position":     ticket,
        "price":        close_price,
        "deviation":    30,
        "magic":        123456,
        "comment":      "close V7",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  ✓  closed @ {result.price:.5f}")
        return True
    else:
        err = result.comment if result else mt5.last_error()
        print(f"  ✗  FAILED: {err}")
        return False


def run(dry_run: bool = False):
    positions = mt5.positions_get()
    if not positions:
        print("No open positions found.")
        return

    v7_positions = [p for p in positions if V7_TAG in (p.comment or "")]
    v8_positions = [p for p in positions if V8_TAG in (p.comment or "")]
    other        = [p for p in positions
                    if V7_TAG not in (p.comment or "")
                    and V8_TAG not in (p.comment or "")]

    print(f"Open positions:  {len(positions)} total")
    print(f"  ICT V7 (will close): {len(v7_positions)}")
    print(f"  ICT V8 (untouched):  {len(v8_positions)}")
    print(f"  Other  (untouched):  {len(other)}")

    # Preview V8 / other positions (informational only)
    if v8_positions or other:
        print("\nLeaving these positions open:")
        for p in v8_positions + other:
            t = "BUY" if p.type == 0 else "SELL"
            print(f"  {p.symbol:<12} {t}  x{p.volume}  @ {p.price_open:.5f}  "
                  f"comment='{p.comment}'  profit=${p.profit:.2f}")

    # Close V7 positions
    if not v7_positions:
        print("\nNo ICT V7 positions to close.")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Closing {len(v7_positions)} ICT V7 position(s):")
    closed = 0
    failed = 0
    for pos in v7_positions:
        ok = close_position(pos, dry_run)
        if ok:
            closed += 1
        else:
            failed += 1

    print(f"\nResult: {closed} closed, {failed} failed")

    # Pending orders summary
    orders = mt5.orders_get()
    if orders:
        v8_orders = [o for o in orders if V8_TAG in (o.comment or "")]
        print(f"\nPending orders unchanged: {len(v8_orders)} ICT V8 orders still placed")
        for o in v8_orders:
            t = {
                mt5.ORDER_TYPE_BUY_LIMIT:  "buy limit",
                mt5.ORDER_TYPE_SELL_LIMIT: "sell limit",
                mt5.ORDER_TYPE_BUY_STOP:   "buy stop",
                mt5.ORDER_TYPE_SELL_STOP:  "sell stop",
            }.get(o.type, str(o.type))
            print(f"  {o.symbol:<12} {t:<11} x{o.volume_initial}  "
                  f"@ {o.price_open:.5f}  "
                  f"SL={o.sl:.5f}  TP={o.tp:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Close ICT V7 positions, keep V8 orders")
    parser.add_argument("--login",    type=int, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--server",   type=str, default=None)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview which positions would be closed without sending orders")
    args = parser.parse_args()

    if not init_mt5(args.login, args.password, args.server):
        sys.exit(1)

    try:
        run(dry_run=args.dry_run)
    finally:
        mt5.shutdown()
