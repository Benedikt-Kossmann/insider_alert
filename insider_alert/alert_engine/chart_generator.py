"""Chart generation for Telegram alerts."""
import glob
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Chart-Output-Verzeichnis
_CHART_DIR = os.path.join(tempfile.gettempdir(), "insider_alert_charts")


def _ensure_chart_dir() -> None:
    os.makedirs(_CHART_DIR, exist_ok=True)


def generate_ticker_chart(
    ohlcv: pd.DataFrame,
    ticker: str,
    score: float,
    support_levels: list[float] | None = None,
    resistance_levels: list[float] | None = None,
    signal_flags: list[str] | None = None,
    days: int = 30,
    style: str = "charles",
) -> Optional[str]:
    """Generiere Candlestick-Chart für Ticker-Alert.

    Parameters
    ----------
    ohlcv : DataFrame mit Open, High, Low, Close, Volume (DatetimeIndex)
    ticker : Ticker-Symbol
    score : Composite Score (0-100)
    support_levels : S/R Support Levels
    resistance_levels : S/R Resistance Levels
    signal_flags : Liste der wichtigsten Signal-Flags
    days : Anzahl Tage im Chart
    style : mplfinance Style-Name

    Returns
    -------
    str — Pfad zur PNG-Datei, oder None bei Fehler
    """
    try:
        import mplfinance as mpf
    except ImportError:
        logger.warning("mplfinance not installed, skipping chart generation")
        return None

    _ensure_chart_dir()

    # Spalten normalisieren
    df = ohlcv.copy()
    df.columns = [c.capitalize() for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        logger.warning("OHLCV missing required columns for %s: %s", ticker, df.columns.tolist())
        return None

    # Letzte N Tage
    data = df.tail(days)
    if len(data) < 5:
        return None

    # Sicherstellen dass Index ein DatetimeIndex ist
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Moving Averages als Overlays
    add_plots = []
    ema10 = data["Close"].ewm(span=10).mean()
    add_plots.append(mpf.make_addplot(ema10, color="blue", width=1.0))
    if len(data) >= 20:
        ema50 = data["Close"].ewm(span=50).mean()
        add_plots.append(mpf.make_addplot(ema50, color="orange", width=1.0))

    # Score-Farbe
    if score >= 75:
        score_label = "HIGH"
    elif score >= 60:
        score_label = "MED"
    else:
        score_label = "LOW"

    title = f"{ticker}  |  Score: {score:.0f}/100 [{score_label}]"

    # Chart kwargs
    kwargs: dict = {
        "type": "candle",
        "style": style,
        "title": title,
        "volume": True,
        "addplot": add_plots,
        "figsize": (12, 8),
        "tight_layout": True,
    }

    # S/R Levels als horizontale Linien
    hlines_values: list[float] = []
    hlines_colors: list[str] = []
    if support_levels:
        for lvl in support_levels[:3]:
            hlines_values.append(lvl)
            hlines_colors.append("green")
    if resistance_levels:
        for lvl in resistance_levels[:3]:
            hlines_values.append(lvl)
            hlines_colors.append("red")

    if hlines_values:
        kwargs["hlines"] = dict(
            hlines=hlines_values,
            colors=hlines_colors,
            linestyle="--",
            linewidths=0.8,
        )

    filepath = os.path.join(_CHART_DIR, f"{ticker}_{datetime.now():%Y%m%d_%H%M}.png")
    kwargs["savefig"] = filepath

    try:
        mpf.plot(data, **kwargs)
        logger.info("Chart generated: %s", filepath)
        return filepath
    except Exception as exc:
        logger.warning("Chart generation failed for %s: %s", ticker, exc)
        return None


def generate_macro_dashboard(
    market_ctx: dict,
    days: int = 90,
) -> Optional[str]:
    """Generiere 4-Panel Makro-Dashboard für den Weekly Report.

    Panels: VIX, Yield Curve (10Y-3M), Dollar (UUP)

    Parameters
    ----------
    market_ctx : Market-Kontext-Dict (aus build_market_context)
    days : Lookback-Periode in Tagen

    Returns
    -------
    str — Pfad zur PNG-Datei, oder None bei Fehler
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed, skipping macro dashboard")
        return None

    _ensure_chart_dir()

    panels: dict[str, dict] = {}

    try:
        import yfinance as yf

        # VIX
        vix = yf.download("^VIX", period=f"{days}d", progress=False, auto_adjust=True)
        if not vix.empty:
            close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
            panels["VIX (Fear Index)"] = {
                "data": close.squeeze(),
                "color": "red",
                "zones": [(0, 20, "green"), (20, 30, "yellow"), (30, 80, "red")],
            }

        # 10Y-3M Yield Spread
        tnx = yf.download("^TNX", period=f"{days}d", progress=False, auto_adjust=True)
        irx = yf.download("^IRX", period=f"{days}d", progress=False, auto_adjust=True)
        if not tnx.empty and not irx.empty:
            tnx_close = tnx["Close"].squeeze() if "Close" in tnx.columns else tnx.iloc[:, 0].squeeze()
            irx_close = irx["Close"].squeeze() if "Close" in irx.columns else irx.iloc[:, 0].squeeze()
            common = tnx_close.index.intersection(irx_close.index)
            if len(common) > 5:
                spread = tnx_close.loc[common] - irx_close.loc[common]
                panels["Yield Curve (10Y-3M)"] = {
                    "data": spread,
                    "color": "blue",
                    "zones": [(-5, 0, "red"), (0, 5, "green")],
                }

        # Dollar Proxy (UUP ETF)
        dxy = yf.download("UUP", period=f"{days}d", progress=False, auto_adjust=True)
        if not dxy.empty:
            close = dxy["Close"].squeeze() if "Close" in dxy.columns else dxy.iloc[:, 0].squeeze()
            panels["Dollar (UUP)"] = {"data": close, "color": "darkgreen"}

    except Exception as exc:
        logger.warning("Macro dashboard data fetch failed: %s", exc)

    if not panels:
        logger.warning("No macro data available for dashboard")
        return None

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, panels.items()):
        data_series = info["data"]
        ax.plot(data_series.index, data_series.values, color=info.get("color", "black"), linewidth=1.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Regime-Zonen (optional)
        if "zones" in info:
            y_min = float(data_series.min())
            y_max = float(data_series.max())
            for low, high, color in info["zones"]:
                # Clip zones to data range to avoid huge blank space
                z_low = max(low, y_min - 1)
                z_high = min(high, y_max + 1)
                if z_low < z_high:
                    ax.axhspan(z_low, z_high, alpha=0.1, color=color)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    filepath = os.path.join(_CHART_DIR, f"macro_dashboard_{datetime.now():%Y%m%d}.png")
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.info("Macro dashboard generated: %s", filepath)
    return filepath


def cleanup_old_charts(max_age_days: int = 7) -> int:
    """Lösche Charts älter als max_age_days Tage.

    Returns
    -------
    int — Anzahl gelöschter Dateien
    """
    cutoff = datetime.now().timestamp() - max_age_days * 86400
    deleted = 0
    for path in glob.glob(os.path.join(_CHART_DIR, "*.png")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                deleted += 1
        except OSError as exc:
            logger.debug("Could not remove chart %s: %s", path, exc)
    if deleted:
        logger.info("Cleaned up %d old charts", deleted)
    return deleted
