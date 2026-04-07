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
    from insider_alert.scheduler.jobs import run_analysis_for_ticker

    tickers = [args.ticker] if args.ticker else config.tickers
    logger.info("Scanning %d ticker(s): %s", len(tickers), tickers)
    for ticker in tickers:
        run_analysis_for_ticker(ticker, config)


def cmd_schedule(args, config) -> None:
    """Start the scheduler."""
    from insider_alert.scheduler.jobs import start_scheduler
    from insider_alert.alert_engine.telegram_alert import send_welcome_message

    send_welcome_message(config)
    logger.info("Starting scheduler...")
    start_scheduler(config, blocking=True)


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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from insider_alert.config import load_config
    config = load_config()

    if args.command in ("scan", "run"):
        cmd_scan(args, config)
    elif args.command == "schedule":
        cmd_schedule(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
