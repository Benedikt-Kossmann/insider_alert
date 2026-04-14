"""Black-Scholes Greeks computation using scipy."""
import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm as _norm
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _norm = None
    _SCIPY_AVAILABLE = False
    logger.debug("scipy not available — Greeks computation disabled (install scipy to enable)")


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float           # pass-through für Kontext
    contract_type: str   # "call" | "put"


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d1."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    contract_type: Literal["call", "put"] = "call",
) -> Greeks:
    """Compute Black-Scholes Greeks for a single option contract.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years (e.g. 30/365 for 30 days).
    r : float
        Risk-free rate as a decimal (e.g. 0.05 for 5%).
    sigma : float
        Implied volatility as a decimal (e.g. 0.30 for 30%).
    contract_type : {"call", "put"}
        Option contract type.

    Returns
    -------
    Greeks
        Dataclass with delta, gamma, vega, theta, iv, contract_type.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, iv=sigma, contract_type=contract_type)

    if not _SCIPY_AVAILABLE:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, iv=sigma, contract_type=contract_type)

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    # Gamma & Vega are identical for calls and puts
    gamma = _norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * _norm.pdf(d1) * sqrt_T / 100  # per 1% IV change

    if contract_type == "call":
        delta = _norm.cdf(d1)
        theta = (
            -S * _norm.pdf(d1) * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * _norm.cdf(d2)
        ) / 365
    else:
        delta = _norm.cdf(d1) - 1
        theta = (
            -S * _norm.pdf(d1) * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * _norm.cdf(-d2)
        ) / 365

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta=round(theta, 4),
        iv=sigma,
        contract_type=contract_type,
    )


def compute_chain_greeks(
    options_df,
    spot: float,
    risk_free_rate: float = 0.05,
) -> list[dict]:
    """Compute Greeks for a complete options chain.

    Parameters
    ----------
    options_df : pd.DataFrame
        Options chain with columns: strike, impliedvolatility (or impliedVolatility),
        contracttype (or contractType), expiration, volume, openinterest (or openInterest).
    spot : float
        Current underlying spot price.
    risk_free_rate : float
        Risk-free interest rate as a decimal.

    Returns
    -------
    list[dict]
        List of dicts with Greeks + volume + strike per contract.
    """
    from datetime import datetime

    results = []
    today = datetime.now().date()

    for _, row in options_df.iterrows():
        try:
            strike = float(row.get("strike", 0))
            iv = float(row.get("impliedvolatility", row.get("impliedVolatility", 0)))
            volume = float(row.get("volume", 0))
            oi = float(row.get("openinterest", row.get("openInterest", 0)))
            ct = str(row.get("contracttype", row.get("contractType", "call"))).lower()

            exp = row.get("expiration", row.get("exp_date"))
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d").date()
            elif hasattr(exp, "date"):
                exp = exp.date()

            T = max((exp - today).days, 1) / 365.0

            greeks = compute_greeks(spot, strike, T, risk_free_rate, iv, ct)
            results.append({
                "strike": strike,
                "expiration": exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp),
                "contract_type": ct,
                "volume": volume,
                "open_interest": oi,
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "vega": greeks.vega,
                "theta": greeks.theta,
                "iv": iv,
                "T": T,
            })
        except Exception as exc:
            logger.debug("Greeks calc failed for strike=%.2f: %s", row.get("strike", 0), exc)

    return results
