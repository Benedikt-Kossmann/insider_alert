"""CLI entry point for the Insider-Activity Detection Bot."""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


def cmd_scan(args, config) -> None:
    """Run one-off analysis for configured (or specified) tickers."""
    from insider_alert.scheduler.jobs import run_analysis_for_ticker, _fetch_macro_features

    macro_features = _fetch_macro_features(config)
    tickers = [args.ticker] if args.ticker else config.tickers
    logger.info("Scanning %d ticker(s): %s", len(tickers), tickers)
    for ticker in tickers:
        run_analysis_for_ticker(ticker, config, macro_features)


def cmd_schedule(args, config) -> None:
    """Start the scheduler."""
    from insider_alert.scheduler.jobs import start_scheduler
    from insider_alert.alert_engine.telegram_alert import send_welcome_message

    send_welcome_message(config)
    logger.info("Starting scheduler...")
    start_scheduler(config, blocking=True)


def cmd_backtest(args, config) -> None:
    """Run walk-forward backtest on OHLCV-based signals."""
    from insider_alert.backtest.engine import run_backtest
    from insider_alert.backtest.metrics import generate_report

    tickers = [args.ticker] if args.ticker else config.tickers
    logger.info("Backtesting %d ticker(s) over %s ...", len(tickers), args.period)

    results = run_backtest(tickers, period=args.period)
    if not results:
        logger.warning("No backtest results produced.")
        return

    report = generate_report(results, threshold=args.threshold)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Insider-Activity Detection Bot",
    )
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="Run one-off analysis for all configured tickers")
    scan_parser.add_argument("--ticker", type=str, default=None, help="Analyze a specific ticker")

    run_parser = subparsers.add_parser("run", help="Alias for scan")
    run_parser.add_argument("--ticker", type=str, default=None, help="Analyze a specific ticker")

    subparsers.add_parser("schedule", help="Start the scheduler (EOD + intraday)")

    bt_parser = subparsers.add_parser("backtest", help="Walk-forward backtest of OHLCV signals")
    bt_parser.add_argument("--ticker", type=str, default=None, help="Backtest a specific ticker")
    bt_parser.add_argument("--period", type=str, default="1y", help="yfinance period (default: 1y)")
    bt_parser.add_argument("--threshold", type=float, default=50.0, help="Signal score threshold (default: 50)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from insider_alert.config import load_config
    from insider_alert.persistence.storage import init_db

    config = load_config()
    init_db()

    if args.command in ("scan", "run"):
        cmd_scan(args, config)
    elif args.command == "schedule":
        cmd_schedule(args, config)
    elif args.command == "backtest":
        cmd_backtest(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
