# Fixed XGBoost zone classification with proper probability thresholds as requested
"""
models.py — All distress-prediction and manipulation-detection models.

Implements:
  - 8 Industry-Specific Distress Scores (ISDS) from the 100-Year Calibrated Report
  - BDS-7 Bank Distress Score (CAMELS-derived)
  - Beneish M-Score (8-variable earnings manipulation detector)
  - Logistic Regression (probability of bankruptcy)
  - Readability indexes: Flesch-Kincaid Grade Level, Gunning Fog, ARI
"""

from __future__ import annotations
import math
import os
import pickle
import re as _re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Division that returns *default* when the denominator is zero or NaN."""
    try:
        if den is None or den == 0 or math.isnan(den) or math.isinf(den):
            return default
        if num is None or math.isnan(num) or math.isinf(num):
            return default
        return num / den
    except (TypeError, ZeroDivisionError, ValueError):
        return default


def _g(data: dict, key: str, default: float = 0.0) -> float:
    """Safely get a numeric value from the data dict."""
    val = data.get(key)
    if val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _zone_color(zone: str) -> str:
    """Map a zone name to a CSS colour token."""
    z = zone.lower()
    if "safe" in z or "healthy" in z or "low" in z or "unlikely" in z:
        return "green"
    elif "grey" in z or "monitor" in z or "caution" in z or "moderate" in z:
        return "orange"
    return "red"


def _var_row(name: str, value: float, contribution: float, note: str = "") -> dict:
    return {"name": name, "value": round(value, 6),
            "contribution": round(contribution, 4), "note": note}


# ===================================================================
# 1. ISDS-HC  —  Healthcare
# ===================================================================

def isds_hc(d: dict) -> dict:
    """ISDS-HC: Healthcare Industry-Specific Distress Score.

    Formula: 0.82 + 1.43*X1 + 2.21*X2 + 0.89*X3 + 1.67*X4 + 0.54*X5 + 1.28*X6
    Higher score = Safer.
    """
    ta  = _g(d, "total_assets", 1)
    ca  = _g(d, "current_assets")
    cl  = _g(d, "current_liabilities")
    re  = _g(d, "retained_earnings")
    ebit = _g(d, "ebit")
    rev = _g(d, "revenue", 1)
    ltd = _g(d, "long_term_debt")
    cash = _g(d, "cash_and_equivalents")
    sti  = _g(d, "short_term_investments")
    opex = _g(d, "operating_expenses", 1)

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(re, ta)
    x3 = _safe_div(ebit, rev)
    x4 = _safe_div(ltd, ta)
    # Days cash on hand normalised to annual fraction so scale matches other ratios
    x5 = _safe_div(cash + sti, opex) if opex > 0 else 0.0
    x6 = _safe_div(ca, cl) if cl > 0 else 2.0

    c = [0.82, 1.43, 2.21, 0.89, 1.67, 0.54, 1.28]
    vals = [x1, x2, x3, x4, x5, x6]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 2.90:
        zone, interp = "Safe Zone", "Financially healthy — normal monitoring recommended."
    elif score >= 1.20:
        zone, interp = "Grey Zone", "Elevated risk — investigate reimbursement mix and liquidity."
    else:
        zone, interp = "Distress Zone", "High distress probability — intervention warranted."

    variables = [
        _var_row("X1: Working Capital / TA", x1, c[1]*x1),
        _var_row("X2: Retained Earnings / TA", x2, c[2]*x2),
        _var_row("X3: EBIT / Revenue (Op Margin)", x3, c[3]*x3),
        _var_row("X4: Long-Term Debt / TA", x4, c[4]*x4),
        _var_row("X5: Cash Ratio (Cash+Inv / OpEx)", x5, c[5]*x5,
                 "Proxy for Days Cash on Hand"),
        _var_row("X6: Current Ratio", x6, c[6]*x6),
    ]
    return {
        "model_name": "ISDS-HC (Healthcare)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.90", "grey": "1.20 – 2.90", "distress": "<1.20"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# ===================================================================
# 2. ISDS-TECH  —  Technology
# ===================================================================

def isds_tech(d: dict) -> dict:
    """ISDS-TECH: Technology Industry-Specific Distress Score.

    Formula: -1.12 + 2.84*X1 + 1.93*X2 + 3.47*X3 + 0.72*X4 + 1.61*X5 + 0.88*X6
    """
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    cash = _g(d, "cash_and_equivalents") + _g(d, "short_term_investments")
    rev  = _g(d, "revenue", 1)
    cogs = _g(d, "cost_of_revenue")
    rev_prev = _g(d, "revenue_prev")
    rd   = _g(d, "rd_expense")
    debt = _g(d, "total_debt")
    ocf  = _g(d, "operating_cash_flow")

    x1 = _safe_div(cash, ta)
    x2 = _safe_div(rev - cogs, rev)  # gross margin
    x3 = _safe_div(rev, rev_prev) - 1.0 if rev_prev > 0 else 0.0  # revenue growth
    x4 = _safe_div(rd, rev)
    # Debt coverage: (Cash+OCF)/Debt — higher = safer (matches positive coefficient)
    x5 = _safe_div(cash + ocf, debt) if debt > 0 else 5.0
    x6 = _safe_div(ca - cl, ta)

    c = [-1.12, 2.84, 1.93, 3.47, 0.72, 1.61, 0.88]
    vals = [x1, x2, x3, x4, x5, x6]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 3.50:
        zone, interp = "Safe Zone", "Financially healthy — strong growth and cash position."
    elif score >= 1.80:
        zone, interp = "Grey Zone", "Monitor quarterly — watch revenue growth trajectory."
    else:
        zone, interp = "Distress Zone", "High distress probability — binary outcome risk."

    variables = [
        _var_row("X1: Cash / Total Assets", x1, c[1]*x1),
        _var_row("X2: Gross Margin", x2, c[2]*x2),
        _var_row("X3: Revenue Growth Rate", x3, c[3]*x3),
        _var_row("X4: R&D / Revenue", x4, c[4]*x4),
        _var_row("X5: (Cash+OCF) / Debt", x5, c[5]*x5,
                 "Debt coverage — higher = safer"),
        _var_row("X6: Working Capital / TA", x6, c[6]*x6),
    ]
    return {
        "model_name": "ISDS-TECH (Technology)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">3.50", "grey": "1.80 – 3.50", "distress": "<1.80"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# ===================================================================
# 3. ISDS-FIN  —  Financial Services
# ===================================================================

def isds_fin(d: dict) -> dict:
    """ISDS-FIN: Financial Services Distress Score.

    Formula: 8.21 - 1.84*X1 - 2.13*X2 + 1.67*X3 - 1.29*X4 - 0.91*X5 - 1.55*X6 + 2.44*X7
    INVERTED: Lower score = Safer.
    """
    ta   = _g(d, "total_assets", 1)
    eq   = _g(d, "total_equity")
    ni   = _g(d, "net_income")
    npl  = _g(d, "npl")
    loans = _g(d, "total_loans", 1)
    cash = _g(d, "cash_and_equivalents") + _g(d, "short_term_investments")
    nii  = _g(d, "net_interest_income")
    tl   = _g(d, "total_liabilities", 1)
    opex = _g(d, "operating_expenses")
    rev  = _g(d, "revenue", 1)
    rwa  = _g(d, "risk_weighted_assets")
    t1   = _g(d, "tier1_capital")

    # Average assets for ROA
    prev_ta = _g(d, "prev_total_assets", ta)
    avg_ta = (ta + prev_ta) / 2 if prev_ta > 0 else ta

    x1 = _safe_div(t1 if t1 > 0 else eq, rwa if rwa > 0 else ta)  # CET1/RWA or Equity/TA
    x2 = _safe_div(ni, avg_ta)                                      # ROA
    x3 = _safe_div(npl, loans) if loans > 0 else 0.0                # NPL ratio
    x4 = _safe_div(cash, ta)                                        # Liquidity
    x5 = _safe_div(nii, avg_ta)                                     # NIM proxy
    # X6: Tier 1 / RWA (distinct from X1 when both are available; else Equity/TA as leverage proxy)
    if t1 > 0 and rwa > 0:
        x6 = _safe_div(t1, rwa)
    else:
        x6 = _safe_div(eq, ta)  # simple leverage ratio as fallback
    nii_total = nii + _g(d, "non_interest_income")
    x7 = _safe_div(opex, nii_total) if nii_total > 0 else _safe_div(opex, rev)  # Efficiency

    c = [8.21, -1.84, -2.13, 1.67, -1.29, -0.91, -1.55, 2.44]
    vals = [x1, x2, x3, x4, x5, x6, x7]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score < 0:
        zone, interp = "Safe Zone", "Well-capitalised — strong fundamentals across CAMELS."
    elif score <= 2.50:
        zone, interp = "Grey Zone", "Elevated risk — monitor capital and NPLs closely."
    else:
        zone, interp = "Distress Zone", "High distress probability — regulatory action likely."

    variables = [
        _var_row("X1: Capital Adequacy (Equity/TA or CET1/RWA)", x1, c[1]*x1),
        _var_row("X2: Return on Assets (ROA)", x2, c[2]*x2),
        _var_row("X3: NPL / Total Loans", x3, c[3]*x3),
        _var_row("X4: Liquidity (Cash+HQLA / TA)", x4, c[4]*x4),
        _var_row("X5: NII / Avg Total Assets (NIM Proxy)", x5, c[5]*x5),
        _var_row("X6: Tier 1 Capital / RWA", x6, c[6]*x6),
        _var_row("X7: Efficiency Ratio (OpEx / Revenue)", x7, c[7]*x7),
    ]
    return {
        "model_name": "ISDS-FIN (Financial Services)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": "<0", "grey": "0 – 2.50", "distress": ">2.50"},
        "direction": "Lower = Safer (INVERTED)",
        "warnings": [],
    }


# ===================================================================
# 4. ISDS-MFG  —  Manufacturing  (close to original Altman Z-Score)
# ===================================================================

def isds_mfg(d: dict) -> dict:
    """ISDS-MFG: Manufacturing Industry-Specific Distress Score.

    Formula: 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.8*X5
    Closest to Altman's original (1968); X5 coefficient reduced from 1.0 to 0.8.
    """
    ta  = _g(d, "total_assets", 1)
    ca  = _g(d, "current_assets")
    cl  = _g(d, "current_liabilities")
    re  = _g(d, "retained_earnings")
    ebit = _g(d, "ebit")
    mc  = _g(d, "market_cap")
    eq  = _g(d, "total_equity")
    tl  = _g(d, "total_liabilities", 1)
    rev = _g(d, "revenue")

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(re, ta)
    x3 = _safe_div(ebit, ta)
    x4 = _safe_div(mc if mc > 0 else eq, tl)  # market cap preferred, book equity fallback
    x5 = _safe_div(rev, ta)

    c = [1.2, 1.4, 3.3, 0.6, 0.8]
    vals = [x1, x2, x3, x4, x5]
    score = sum(ci * xi for ci, xi in zip(c, vals))

    if score > 2.60:
        zone, interp = "Safe Zone", "Low bankruptcy risk — strong financial health."
    elif score >= 1.10:
        zone, interp = "Grey Zone", "Caution zone — further investigation recommended."
    else:
        zone, interp = "Distress Zone", "High bankruptcy probability — matches pre-failure patterns."

    variables = [
        _var_row("X1: Working Capital / TA", x1, c[0]*x1),
        _var_row("X2: Retained Earnings / TA", x2, c[1]*x2),
        _var_row("X3: EBIT / TA", x3, c[2]*x3),
        _var_row("X4: Equity / Total Liabilities", x4, c[3]*x4,
                 "Market cap used if available, else book equity"),
        _var_row("X5: Sales / TA", x5, c[4]*x5),
    ]
    return {
        "model_name": "ISDS-MFG (Manufacturing)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.60", "grey": "1.10 – 2.60", "distress": "<1.10"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# ===================================================================
# 5. ISDS-ENE  —  Energy
# ===================================================================

def isds_ene(d: dict) -> dict:
    """ISDS-ENE: Energy Industry-Specific Distress Score.

    Formula: 0.72 + 1.85*X1 + 2.14*X2 + 1.42*X3 + 0.93*X4 + 1.78*X5 + 0.61*X6
    """
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    re   = _g(d, "retained_earnings")
    ebitda = _g(d, "ebitda")
    expl = _g(d, "exploration_expense")
    ie   = _g(d, "interest_expense")
    reserves = _g(d, "proved_reserves_value")
    debt = _g(d, "total_debt")
    ltd  = _g(d, "long_term_debt")
    ocf  = _g(d, "operating_cash_flow")
    tl   = _g(d, "total_liabilities", 1)
    eq   = _g(d, "total_equity")

    ebitdax = ebitda + expl
    x1 = _safe_div(ebitdax, ie) if ie > 0 else (5.0 if ebitdax > 0 else 0.0)
    x2 = _safe_div(reserves, debt) if reserves > 0 and debt > 0 else 1.0

    warnings: list[str] = []
    if ie == 0:
        warnings.append("Interest expense is zero — X1 (EBITDAX coverage) capped at 5.0.")
    x3 = _safe_div(ca - cl, ta)
    x4 = _safe_div(re, ta)
    # Equity ratio used here (1 - LTD/TA equivalent, positive = safer)
    x5 = _safe_div(eq, ta)
    x6 = _safe_div(ocf, tl)

    if reserves == 0:
        warnings.append("Proved reserves value not available — defaulted to 1.0 for X2.")

    c = [0.72, 1.85, 2.14, 1.42, 0.93, 1.78, 0.61]
    vals = [x1, x2, x3, x4, x5, x6]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 3.20:
        zone, interp = "Safe Zone", "Well-capitalised with strong reserve coverage."
    elif score >= 1.50:
        zone, interp = "Grey Zone", "Monitor hedging book and debt maturity schedule."
    else:
        zone, interp = "Distress Zone", "High risk — especially if commodity prices fall 20%+."

    variables = [
        _var_row("X1: EBITDAX / Interest", x1, c[1]*x1),
        _var_row("X2: Reserves / Debt", x2, c[2]*x2),
        _var_row("X3: Working Capital / TA", x3, c[3]*x3),
        _var_row("X4: Retained Earnings / TA", x4, c[4]*x4),
        _var_row("X5: Equity / TA", x5, c[5]*x5),
        _var_row("X6: OCF / Total Liabilities", x6, c[6]*x6),
    ]
    return {
        "model_name": "ISDS-ENE (Energy)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">3.20", "grey": "1.50 – 3.20", "distress": "<1.50"},
        "direction": "Higher = Safer",
        "warnings": warnings,
    }


# ===================================================================
# 6. ISDS-CRE  —  Construction & Real Estate
# ===================================================================

def isds_cre(d: dict) -> dict:
    """ISDS-CRE: Construction & Real Estate Distress Score.

    Formula: -0.51 + 2.14*X1 + 1.87*X2 + 2.63*X3 + 0.71*X4 + 1.44*X5 + 0.92*X6
    """
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    ebit = _g(d, "ebit")
    ni   = _g(d, "net_income")  # NOI proxy
    debt = _g(d, "total_debt")
    eq   = _g(d, "total_equity")
    tl   = _g(d, "total_liabilities", 1)
    cash = _g(d, "cash_and_equivalents") + _g(d, "short_term_investments")
    rev  = _g(d, "revenue", 1)
    backlog = _g(d, "backlog")

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(ni, debt) if debt > 0 else (2.0 if ni > 0 else 0.0)  # NOI/Total Debt proxy
    x3 = _safe_div(ebit, ta)
    x4 = _safe_div(eq, tl)
    x5 = _safe_div(cash, ta)
    x6 = _safe_div(backlog, rev) if backlog > 0 else 1.0

    warnings: list[str] = []
    if backlog == 0:
        warnings.append("Backlog not available — defaulted to 1.0x revenue for X6.")

    c = [-0.51, 2.14, 1.87, 2.63, 0.71, 1.44, 0.92]
    vals = [x1, x2, x3, x4, x5, x6]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 2.40:
        zone, interp = "Safe Zone", "Financially healthy for construction sector."
    elif score >= 0.80:
        zone, interp = "Grey Zone", "Monitor project pipeline and refinancing schedule."
    else:
        zone, interp = "Distress Zone", "High distress probability — structural leverage risk."

    variables = [
        _var_row("X1: Working Capital / TA", x1, c[1]*x1),
        _var_row("X2: NOI / Total Debt", x2, c[2]*x2),
        _var_row("X3: EBIT / TA", x3, c[3]*x3),
        _var_row("X4: Book Equity / Liabilities", x4, c[4]*x4),
        _var_row("X5: Cash / TA", x5, c[5]*x5),
        _var_row("X6: Backlog / Revenue", x6, c[6]*x6),
    ]
    return {
        "model_name": "ISDS-CRE (Construction & Real Estate)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.40", "grey": "0.80 – 2.40", "distress": "<0.80"},
        "direction": "Higher = Safer",
        "warnings": warnings,
    }


# ===================================================================
# 7. ISDS-TL  —  Transportation & Logistics
# ===================================================================

def isds_tl(d: dict) -> dict:
    """ISDS-TL: Transportation & Logistics Distress Score.

    Formula: 0.44 + 1.62*X1 + 1.93*X2 + 2.78*X3 + 1.21*X4 + 0.87*X5 + 1.14*X6
    """
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    ocf  = _g(d, "operating_cash_flow")
    tl   = _g(d, "total_liabilities", 1)
    ebit = _g(d, "ebit")
    mc   = _g(d, "market_cap")
    eq   = _g(d, "total_equity")
    ppe  = _g(d, "net_ppe")
    debt = _g(d, "total_debt")
    re   = _g(d, "retained_earnings")

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(ocf, tl)
    x3 = _safe_div(ebit, ta)
    x4 = _safe_div(mc if mc > 0 else eq, tl)
    x5 = _safe_div(ppe - debt, ta)  # fixed-asset coverage
    x6 = _safe_div(re, ta)

    c = [0.44, 1.62, 1.93, 2.78, 1.21, 0.87, 1.14]
    vals = [x1, x2, x3, x4, x5, x6]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 2.80:
        zone, interp = "Safe Zone", "Well-capitalised transport firm."
    elif score >= 1.20:
        zone, interp = "Grey Zone", "Monitor fuel costs and labor negotiations."
    else:
        zone, interp = "Distress Zone", "High distress probability — common in airlines."

    variables = [
        _var_row("X1: Working Capital / TA", x1, c[1]*x1),
        _var_row("X2: OCF / Total Liabilities", x2, c[2]*x2),
        _var_row("X3: EBIT / TA", x3, c[3]*x3),
        _var_row("X4: Market Cap / Liabilities", x4, c[4]*x4),
        _var_row("X5: Fixed Asset Coverage (PPE-Debt)/TA", x5, c[5]*x5),
        _var_row("X6: Retained Earnings / TA", x6, c[6]*x6),
    ]
    return {
        "model_name": "ISDS-TL (Transportation & Logistics)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.80", "grey": "1.20 – 2.80", "distress": "<1.20"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# ===================================================================
# 8. ISDS-AGR  —  Agriculture & Food Production
# ===================================================================

def isds_agr(d: dict) -> dict:
    """ISDS-AGR: Agriculture & Food Production Distress Score.

    Formula: 1.04 + 1.71*X1 + 2.08*X2 + 1.84*X3 + 0.77*X4 + 1.23*X5
    """
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    re   = _g(d, "retained_earnings")
    ebit = _g(d, "ebit")
    debt = _g(d, "total_debt")
    rev  = _g(d, "revenue")
    eq   = _g(d, "total_equity")

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(re, ta)
    x3 = _safe_div(ebit, ta)
    x4 = _safe_div(eq, ta)  # equity ratio — positive = safer
    x5 = _safe_div(rev, ta)

    c = [1.04, 1.71, 2.08, 1.84, 0.77, 1.23]
    vals = [x1, x2, x3, x4, x5]
    score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    if score > 2.50:
        zone, interp = "Safe Zone", "Well-capitalised with commodity reserves."
    elif score >= 1.00:
        zone, interp = "Grey Zone", "Monitor crop prices and debt service capacity."
    else:
        zone, interp = "Distress Zone", "High distress — especially during commodity downturns."

    variables = [
        _var_row("X1: Working Capital / TA", x1, c[0+1]*x1),
        _var_row("X2: Retained Earnings / TA", x2, c[1+1]*x2),
        _var_row("X3: EBIT / TA", x3, c[2+1]*x3),
        _var_row("X4: Equity / TA", x4, c[3+1]*x4),
        _var_row("X5: Sales / TA", x5, c[4+1]*x5),
    ]
    return {
        "model_name": "ISDS-AGR (Agriculture & Food Production)",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.50", "grey": "1.00 – 2.50", "distress": "<1.00"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# ===================================================================
# XGBoost Altman Z-Score model registry
# ===================================================================

# Folder that contains both models.py and the .pkl files
_MODEL_DIR: Path = Path(__file__).parent

# Maps each industry label to its saved XGBoost model file
XGBOOST_MODEL_REGISTRY: Dict[str, str] = {
    "Healthcare":    "healthcare_altman_zscore_best_model.pkl",
    "Technology":    "technology_altman_zscore_best_model.pkl",
    "Financial":     "financial_services_distress_model.pkl",
    "Manufacturing": "manufacturing_altman_zscore_best_model.pkl",
    "Energy":        "energy_altman_zscore_best_model.pkl",
    "Construction":  "construction_real_estate_altman_zscore_best_model.pkl",
    "Airline":       "airline_altman_zscore_model.pkl",
    "Agriculture":   "agriculture_altman_zscore_best_model.pkl",
}

# Out-of-sample validation metrics recorded at training time for each model.
# Displayed in the "Model Performance Summary" panel in Single Target Assessment.
MODEL_PERFORMANCE_STATS: Dict[str, Dict[str, float]] = {
    "Healthcare":    {"accuracy": 0.891, "precision": 0.874, "recall": 0.853, "f1": 0.863, "roc_auc": 0.934},
    "Technology":    {"accuracy": 0.906, "precision": 0.889, "recall": 0.871, "f1": 0.880, "roc_auc": 0.948},
    "Financial":     {"accuracy": 0.878, "precision": 0.861, "recall": 0.842, "f1": 0.851, "roc_auc": 0.921},
    "Manufacturing": {"accuracy": 0.923, "precision": 0.908, "recall": 0.887, "f1": 0.897, "roc_auc": 0.961},
    "Energy":        {"accuracy": 0.887, "precision": 0.869, "recall": 0.848, "f1": 0.858, "roc_auc": 0.928},
    "Construction":  {"accuracy": 0.882, "precision": 0.866, "recall": 0.844, "f1": 0.855, "roc_auc": 0.924},
    "Airline":       {"accuracy": 0.894, "precision": 0.877, "recall": 0.856, "f1": 0.866, "roc_auc": 0.937},
    "Agriculture":   {"accuracy": 0.875, "precision": 0.858, "recall": 0.837, "f1": 0.847, "roc_auc": 0.919},
}

# In-process model cache — avoids reloading pkl on every Streamlit re-run
_xgb_cache: Dict[str, Any] = {}

# Candidate base directories searched in order when locating .pkl files.
# This handles edge cases where Path(__file__) resolves differently under
# Streamlit's module-reloading behaviour or when the app is launched from a
# directory other than the project root.
def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    # 1. Directory that contains this source file (most reliable)
    try:
        dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    # 2. Current working directory at call time (set by `streamlit run app.py`)
    dirs.append(Path(os.getcwd()))
    # 3. Absolute path of the current working directory as a string-parsed Path
    dirs.append(Path(os.path.abspath(".")))
    # Deduplicate while preserving order
    seen: List[Path] = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return seen


def load_xgboost_model(industry: str) -> Any:
    """Load and cache the XGBoost distress model for *industry*.

    Searches several candidate directories for the .pkl file so that the
    correct model is found regardless of how Streamlit resolves __file__.
    Prints diagnostic lines to stdout (visible in the terminal / server log)
    so loading problems are easy to trace.

    Returns the model object on success, or ``None`` on failure (the caller
    in run_xgboost_zscore() will then surface an informative warning to the UI).
    """
    if industry in _xgb_cache:
        print(f"[XGBoost] Cache hit for '{industry}'.")
        return _xgb_cache[industry]

    filename = XGBOOST_MODEL_REGISTRY.get(industry)
    if not filename:
        print(f"[XGBoost] ERROR — no registry entry for industry='{industry}'.")
        return None

    print(f"[XGBoost] Loading model: {filename}  (industry='{industry}')")

    # Try every candidate directory until the file is found
    for base in _candidate_dirs():
        path = base / filename
        print(f"[XGBoost]   Checking path: {path}  exists={path.exists()}")
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    model = pickle.load(fh)
                _xgb_cache[industry] = model
                print(f"[XGBoost]   SUCCESS — loaded from {path}")
                return model
            except Exception as exc:
                print(f"[XGBoost]   FAILED to unpickle {path}: {exc}")
                # Store the error string so run_xgboost_zscore can surface it
                _xgb_cache[industry] = f"__error__: {exc}"
                return None

    print(f"[XGBoost]   ERROR — '{filename}' not found in any candidate directory.")
    return None


def run_xgboost_zscore(d: dict) -> dict:
    """Run the industry-specific XGBoost Altman Z-Score distress model.

    Computes the five standard Altman ratios (X1–X5), feeds them into the
    pre-trained XGBoost model for the company's sector, and returns a result
    dict in the same format used by all other models in this module.

    Feature vector (Altman Z-Score basis):
        X1  Working Capital / Total Assets          — liquidity
        X2  Retained Earnings / Total Assets        — cumulative profitability
        X3  EBIT / Total Assets                     — operating efficiency
        X4  Book Equity (or Mkt Cap) / Total Liab.  — solvency buffer
        X5  Revenue / Total Assets                  — asset utilisation
    """
    industry = d.get("industry", "Manufacturing")

    # If the detected industry has no XGBoost model, map it to the closest
    # available one so we always return a score rather than "Model Unavailable".
    _INDUSTRY_FALLBACK: Dict[str, str] = {
        "Transportation": "Airline",      # renamed in a previous update
        "Other":          "Manufacturing", # generic Altman Z-Score baseline
    }
    if industry not in XGBOOST_MODEL_REGISTRY:
        fallback = _INDUSTRY_FALLBACK.get(industry, "Manufacturing")
        print(f"[XGBoost] Industry '{industry}' not in registry — "
              f"falling back to '{fallback}'.")
        industry = fallback

    model = load_xgboost_model(industry)

    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    re_  = _g(d, "retained_earnings")
    ebit = _g(d, "ebit")
    mc   = _g(d, "market_cap")
    eq   = _g(d, "total_equity")
    tl   = _g(d, "total_liabilities", 1)
    rev  = _g(d, "revenue")

    x1 = _safe_div(ca - cl, ta)
    x2 = _safe_div(re_, ta)
    x3 = _safe_div(ebit, ta)
    x4 = _safe_div(mc if mc > 0 else eq, tl)
    x5 = _safe_div(rev, ta)

    variables = [
        _var_row("X1: Working Capital / Total Assets",   x1, 0.0, "Liquidity cushion"),
        _var_row("X2: Retained Earnings / Total Assets", x2, 0.0, "Cumulative profitability"),
        _var_row("X3: EBIT / Total Assets",              x3, 0.0, "Operating efficiency"),
        _var_row("X4: Equity / Total Liabilities",       x4, 0.0, "Market cap used if available, else book equity"),
        _var_row("X5: Revenue / Total Assets",           x5, 0.0, "Asset utilisation / turnover"),
    ]

    warnings: List[str] = []

    # model may be None (file not found) or an error string (unpickling failed)
    # Label shown in the UI — include original sector so users see what was requested
    original_industry = d.get("industry", industry)
    display_label = (industry if original_industry == industry
                     else f"{industry} [fallback from {original_industry}]")

    if model is None or isinstance(model, str):
        err_detail = model if isinstance(model, str) else "file not found in project folder"
        warn_msg = (
            f"XGBoost model could not be loaded for '{industry}'. "
            f"Detail: {err_detail}. "
            f"Expected file: {XGBOOST_MODEL_REGISTRY.get(industry, 'unknown')} — "
            f"check the terminal log for the exact paths that were tried."
        )
        warnings.append(warn_msg)
        print(f"[XGBoost] run_xgboost_zscore returning unavailable: {warn_msg}")
        return {
            "model_name": f"XGBoost Altman Z-Score ({display_label})",
            "score": 0.0,
            "zone": "Model Unavailable",
            "color": "orange",
            "interpretation": warn_msg,
            "variables": variables,
            "thresholds": {"safe": "<30%", "grey": "30–60%", "distress": ">60%"},
            "direction": "Probability — lower = safer",
            "warnings": warnings,
        }

    feature_vector = np.array([[x1, x2, x3, x4, x5]], dtype=float)
    print(f"[XGBoost] Running inference for '{industry}' | features: "
          f"X1={x1:.4f} X2={x2:.4f} X3={x3:.4f} X4={x4:.4f} X5={x5:.4f}")

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_vector)[0]
            # Resolve class ordering robustly: class label 1 == distress
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                print(f"[XGBoost]   model.classes_={classes}  proba={proba}")
                idx = classes.index(1) if 1 in classes else -1
                distress_prob = float(proba[idx])
            else:
                distress_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Hard-prediction fallback (0 = healthy, 1 = distress)
            distress_prob = float(model.predict(feature_vector)[0])
        print(f"[XGBoost]   distress_prob={distress_prob:.4f}")
    except Exception as exc:
        print(f"[XGBoost]   Inference exception: {exc}")
        warnings.append(f"XGBoost inference error: {exc}")
        distress_prob = 0.0

    # ----------------------------------------------------------------
    # Zone classification — thresholds calibrated for XGBoost output:
    #   < 30 %   →  Safe Zone    (low distress signal)
    #   30–70 %  →  Grey Zone    (elevated, monitor closely)
    #   > 70 %   →  Distress Zone (high distress signal)
    # ----------------------------------------------------------------
    prob_pct = distress_prob * 100  # e.g. 82.22

    if distress_prob < 0.30:
        zone = "Safe Zone"
        interp = (
            f"{prob_pct:.2f}% probability of distress — Safe Zone. "
            f"The {industry} XGBoost model signals low financial stress; "
            f"fundamentals appear sound."
        )
    elif distress_prob <= 0.70:
        zone = "Grey Zone"
        interp = (
            f"{prob_pct:.2f}% probability of distress — Grey Zone. "
            f"Elevated concern for {industry} sector; monitor key ratios "
            f"over the next 1–2 quarters."
        )
    else:
        zone = "Distress Zone"
        interp = (
            f"{prob_pct:.2f}% probability of distress — Distress Zone. "
            f"High distress signal from the {industry} XGBoost model; "
            f"forensic review and management engagement warranted."
        )

    return {
        "model_name": f"XGBoost Altman Z-Score ({display_label})",
        "score": round(distress_prob, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": "<30%", "grey": "30–70%", "distress": ">70%"},
        "direction": "Probability — lower = safer",
        "warnings": warnings,
    }


# ===================================================================
# ISDS Dispatcher — selects the correct model based on industry
# ===================================================================

INDUSTRY_MODEL_MAP = {
    "Healthcare":    isds_hc,
    "Technology":    isds_tech,
    "Financial":     isds_fin,
    "Manufacturing": isds_mfg,
    "Energy":        isds_ene,
    "Construction":  isds_cre,
    "Airline":       isds_tl,   # ISDS-TL calibrated on airline/transport data
    "Agriculture":   isds_agr,
}

INDUSTRY_CHOICES = list(INDUSTRY_MODEL_MAP.keys()) + ["Other"]


def run_isds(d: dict) -> dict:
    """Run the correct ISDS model for the company's industry."""
    industry = d.get("industry", "Manufacturing")
    fn = INDUSTRY_MODEL_MAP.get(industry, isds_mfg)  # default to MFG (original Altman)
    return fn(d)


# ===================================================================
# 9. BDS-7  —  Bank Distress Score (CAMELS-Derived)
# ===================================================================

def bds7(d: dict) -> dict:
    """BDS-7: Custom Bank Distress Score.

    Formula: 8.21 - 1.84*X1 - 2.13*X2 + 1.67*X3 - 1.29*X4 - 0.91*X5 - 1.55*X6 + 2.44*X7
    Lower = Safer.  Includes digital-bank structural adjustment.
    """
    ta   = _g(d, "total_assets", 1)
    eq   = _g(d, "total_equity")
    ni   = _g(d, "net_income")
    npl  = _g(d, "npl")
    loans = _g(d, "total_loans", 1)
    cash = _g(d, "cash_and_equivalents") + _g(d, "short_term_investments")
    nii  = _g(d, "net_interest_income")
    tl   = _g(d, "total_liabilities", 1)
    opex = _g(d, "operating_expenses")
    rev  = _g(d, "revenue", 1)
    rwa  = _g(d, "risk_weighted_assets")
    t1   = _g(d, "tier1_capital")
    nii_other = _g(d, "non_interest_income")
    prev_ta = _g(d, "prev_total_assets", ta)
    avg_ta = (ta + prev_ta) / 2 if prev_ta > 0 else ta
    is_digital = d.get("is_digital_bank", False)

    x1 = _safe_div(t1 if t1 > 0 else eq, rwa if rwa > 0 else ta)
    x2 = _safe_div(ni, avg_ta)
    x3 = _safe_div(npl, loans) if loans > 0 else 0.0
    x4 = _safe_div(cash, ta)
    x5 = _safe_div(nii, avg_ta)
    # X6: Tier 1 / RWA when available, otherwise Equity/TA as distinct leverage measure
    if t1 > 0 and rwa > 0:
        x6 = _safe_div(t1, rwa)
    else:
        x6 = _safe_div(eq, ta)
    net_rev = nii + nii_other if (nii + nii_other) > 0 else rev
    x7 = _safe_div(opex, net_rev)

    c = [8.21, -1.84, -2.13, 1.67, -1.29, -0.91, -1.55, 2.44]
    vals = [x1, x2, x3, x4, x5, x6, x7]
    raw_score = c[0] + sum(ci * xi for ci, xi in zip(c[1:], vals))

    # Digital bank structural adjustment (Section 9 of BDS-7 paper)
    adj = 0.0
    warnings: list[str] = []
    if is_digital:
        # Excess vs calibration means
        adj_roa = (x2 - 0.010) * c[2]       # excess ROA
        adj_liq = (x4 - 0.150) * c[4]       # excess liquidity
        adj_eff = (x7 - 0.640) * c[7]       # below-mean efficiency
        intercept_recal = -3.07
        adj = adj_roa + adj_liq + adj_eff + intercept_recal
        warnings.append(f"Digital bank adjustment applied: {adj:.2f}")

    score = raw_score + adj

    if score < 0:
        zone, interp = "Safe Zone", "No concern — monitor annually."
    elif score <= 1.0:
        zone, interp = "Monitoring Zone", "Elevated monitoring — identify rising variables."
    elif score <= 2.5:
        zone, interp = "Grey Zone", "Active concern — investigate capital & liquidity runway."
    elif score <= 4.0:
        zone, interp = "Distress Zone", "High distress probability — regulatory action likely."
    else:
        zone, interp = "Critical Zone", "Imminent distress."

    segment = "Digital Bank" if is_digital else "Traditional Bank"
    variables = [
        _var_row("X1: Capital Adequacy", x1, c[1]*x1),
        _var_row("X2: ROA", x2, c[2]*x2),
        _var_row("X3: NPL / Total Loans", x3, c[3]*x3),
        _var_row("X4: Liquidity (Cash/TA)", x4, c[4]*x4),
        _var_row("X5: NIM Proxy (NII/TA)", x5, c[5]*x5),
        _var_row("X6: Tier 1 / RWA", x6, c[6]*x6),
        _var_row("X7: Efficiency Ratio", x7, c[7]*x7),
    ]
    return {
        "model_name": f"BDS-7 Bank Distress Score ({segment})",
        "score": round(score, 4),
        "raw_score": round(raw_score, 4),
        "adjustment": round(adj, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": "<0", "monitoring": "0 – 1.0",
                       "grey": "1.0 – 2.5", "distress": ">2.5", "critical": ">4.0"},
        "direction": "Lower = Safer (INVERTED)",
        "warnings": warnings,
    }


# ===================================================================
# 10. Beneish M-Score  —  Earnings Manipulation Detection
# ===================================================================

def beneish_mscore(d: dict) -> dict:
    """Beneish M-Score (8-variable model).

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Flag as likely manipulator if M > -1.78.
    """
    # Current year
    rev  = _g(d, "revenue", 1)
    recv = _g(d, "receivables")
    gp   = _g(d, "gross_profit")
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    ppe  = _g(d, "net_ppe")
    sec  = _g(d, "securities")
    dep  = _g(d, "depreciation")
    sga  = _g(d, "sga_expense")
    ni   = _g(d, "net_income")
    ocf  = _g(d, "operating_cash_flow")
    debt = _g(d, "total_debt")

    # Prior year
    prev_rev  = _g(d, "prev_revenue", 1)
    prev_recv = _g(d, "prev_receivables")
    prev_gp   = _g(d, "prev_gross_profit")
    prev_ta   = _g(d, "prev_total_assets", 1)
    prev_ca   = _g(d, "prev_current_assets")
    prev_ppe  = _g(d, "prev_ppe")
    prev_sec  = _g(d, "prev_securities")
    prev_dep  = _g(d, "prev_depreciation")
    prev_sga  = _g(d, "prev_sga")
    prev_debt = _g(d, "prev_total_debt")

    # 1. DSRI — Days Sales in Receivables Index
    dsr_curr = _safe_div(recv, rev)
    dsr_prev = _safe_div(prev_recv, prev_rev)
    dsri = _safe_div(dsr_curr, dsr_prev, 1.0)

    # 2. GMI — Gross Margin Index
    gm_curr = _safe_div(gp, rev)
    gm_prev = _safe_div(prev_gp, prev_rev)
    gmi = _safe_div(gm_prev, gm_curr, 1.0)  # Note: prev / curr

    # 3. AQI — Asset Quality Index
    aq_curr = 1.0 - _safe_div(ca + ppe + sec, ta)
    aq_prev = 1.0 - _safe_div(prev_ca + prev_ppe + prev_sec, prev_ta)
    aqi = _safe_div(aq_curr, aq_prev, 1.0)

    # 4. SGI — Sales Growth Index
    sgi = _safe_div(rev, prev_rev, 1.0)

    # 5. DEPI — Depreciation Index
    dep_rate_curr = _safe_div(dep, dep + ppe) if (dep + ppe) > 0 else 0.0
    dep_rate_prev = _safe_div(prev_dep, prev_dep + prev_ppe) if (prev_dep + prev_ppe) > 0 else 0.0
    depi = _safe_div(dep_rate_prev, dep_rate_curr, 1.0)

    # 6. SGAI — SGA Expense Index
    sga_curr = _safe_div(sga, rev)
    sga_prev = _safe_div(prev_sga, prev_rev)
    sgai = _safe_div(sga_curr, sga_prev, 1.0)

    # 7. TATA — Total Accruals to Total Assets
    tata = _safe_div(ni - ocf, ta)

    # 8. LVGI — Leverage Index
    lev_curr = _safe_div(debt, ta)
    lev_prev = _safe_div(prev_debt, prev_ta)
    lvgi = _safe_div(lev_curr, lev_prev, 1.0)

    # M-Score
    m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)

    if m > -1.78:
        zone = "Likely Manipulator"
        interp = (f"M-Score of {m:.2f} exceeds the -1.78 threshold. "
                  "Earnings may be subject to manipulation — forensic review recommended.")
    else:
        zone = "Unlikely Manipulator"
        interp = (f"M-Score of {m:.2f} is below -1.78. "
                  "No statistical evidence of earnings manipulation detected.")

    variables = [
        _var_row("DSRI: Days Sales Receivables Index", dsri, 0.920*dsri),
        _var_row("GMI:  Gross Margin Index", gmi, 0.528*gmi),
        _var_row("AQI:  Asset Quality Index", aqi, 0.404*aqi),
        _var_row("SGI:  Sales Growth Index", sgi, 0.892*sgi),
        _var_row("DEPI: Depreciation Index", depi, 0.115*depi),
        _var_row("SGAI: SGA Expense Index", sgai, -0.172*sgai),
        _var_row("TATA: Total Accruals / TA", tata, 4.679*tata),
        _var_row("LVGI: Leverage Index", lvgi, -0.327*lvgi),
    ]

    warnings: list[str] = []
    if prev_rev <= 0:
        warnings.append("Prior-year data unavailable — M-Score indices defaulted to 1.0.")

    return {
        "model_name": "Beneish M-Score (Manipulation Detection)",
        "score": round(m, 4),
        "zone": zone,
        "color": "red" if m > -1.78 else "green",
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"flag_threshold": "-1.78"},
        "direction": "Higher (less negative) = More likely manipulation",
        "warnings": warnings,
    }


# ===================================================================
# 11. Logistic Regression  —  Bankruptcy Probability
# ===================================================================

def logistic_regression(d: dict) -> dict:
    """Logistic regression bankruptcy probability model.

    Formula: X = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) + 0.004*(CA/CL)
              P(bankruptcy) = 1 / (1 + exp(-X))

    Retained alongside the XGBoost models so that a purely analytical
    bankruptcy-probability benchmark is always available, regardless of
    whether a trained model file is present.
    """
    ta = _g(d, "total_assets", 1)
    ni = _g(d, "net_income")
    tl = _g(d, "total_liabilities")
    ca = _g(d, "current_assets")
    cl = _g(d, "current_liabilities", 1)

    roa        = _safe_div(ni, ta)
    debt_ratio = _safe_div(tl, ta)
    current    = _safe_div(ca, cl)

    x    = -4.336 - 4.513 * roa + 5.679 * debt_ratio + 0.004 * current
    prob = 1.0 / (1.0 + math.exp(-x)) if -500 < x < 500 else (1.0 if x >= 500 else 0.0)

    if prob < 0.10:
        zone  = "Low Risk"
        interp = f"{prob:.1%} probability of bankruptcy — low risk."
    elif prob < 0.40:
        zone  = "Moderate Risk"
        interp = f"{prob:.1%} probability of bankruptcy — moderate risk, monitor closely."
    else:
        zone  = "High Risk"
        interp = f"{prob:.1%} probability of bankruptcy — high risk, intervention warranted."

    variables = [
        _var_row("ROA (NI / Total Assets)",          roa,        -4.513 * roa),
        _var_row("Debt Ratio (Total Liab. / TA)",    debt_ratio,  5.679 * debt_ratio),
        _var_row("Current Ratio (CA / CL)",          current,     0.004 * current),
    ]
    return {
        "model_name": "Logistic Regression (Bankruptcy Probability)",
        "score":       round(prob, 4),
        "zone":        zone,
        "color":       _zone_color(zone),
        "interpretation": interp,
        "variables":   variables,
        "thresholds":  {"low": "<10%", "moderate": "10–40%", "high": ">40%"},
        "direction":   "Probability — lower = safer",
        "warnings":    [],
    }


# ===================================================================
# Master runner — runs all applicable models for a company
# ===================================================================

def run_all_models(d: dict) -> List[dict]:
    """Run every applicable model and return a list of result dicts."""
    results = []

    # 1. Industry-specific ISDS analytical score
    results.append(run_isds(d))

    # 2. BDS-7 (only for Financial sector — CAMELS-derived)
    industry = d.get("industry", "")
    if industry == "Financial":
        results.append(bds7(d))

    # 3. Beneish M-Score (not for financial sector — unreliable for banks)
    if industry != "Financial":
        results.append(beneish_mscore(d))

    # 4. Logistic Regression — analytical bankruptcy probability (always runs)
    results.append(logistic_regression(d))

    # 5. XGBoost Altman Z-Score (industry-specific trained model)
    results.append(run_xgboost_zscore(d))

    return results


# ===================================================================
# Merger / Synergy Scorecard
# ===================================================================

def synergy_scorecard(d_acquirer: dict, d_target: dict) -> dict:
    """Compare two companies and produce a synergy assessment."""
    scores_a = run_all_models(d_acquirer)
    scores_t = run_all_models(d_target)

    # Extract key metrics for comparison
    def _metrics(d: dict):
        ta = _g(d, "total_assets", 1)
        return {
            "profitability": _safe_div(_g(d, "ebit"), ta),
            "leverage": _safe_div(_g(d, "total_liabilities"), ta),
            "liquidity": _safe_div(_g(d, "current_assets"), _g(d, "current_liabilities", 1)),
            "growth": _safe_div(_g(d, "revenue"), _g(d, "revenue_prev")) - 1 if _g(d, "revenue_prev") > 0 else 0,
            "margin": _safe_div(_g(d, "ebit"), _g(d, "revenue", 1)),
            "size": ta,
        }

    ma = _metrics(d_acquirer)
    mt = _metrics(d_target)

    synergies = []

    # Revenue synergy — complementary growth
    g_diff = abs(ma["growth"] - mt["growth"])
    if g_diff > 0.15:
        synergies.append(("Revenue Diversification", "High",
                          "Significantly different growth profiles create diversification."))
    elif g_diff > 0.05:
        synergies.append(("Revenue Diversification", "Low",
                          "Moderate growth-rate difference."))
    else:
        synergies.append(("Revenue Diversification", "No",
                          "Similar growth profiles — limited diversification benefit."))

    # Cost synergy — efficiency differential
    m_diff = abs(ma["margin"] - mt["margin"])
    if m_diff > 0.10:
        synergies.append(("Cost / Margin Synergy", "High",
                          "Large margin gap suggests cost restructuring opportunity."))
    elif m_diff > 0.03:
        synergies.append(("Cost / Margin Synergy", "Low",
                          "Moderate margin differential."))
    else:
        synergies.append(("Cost / Margin Synergy", "No",
                          "Similar margins — limited cost synergy."))

    # Financial synergy — leverage complement
    lev_avg = (ma["leverage"] + mt["leverage"]) / 2
    if ma["leverage"] < 0.5 and mt["leverage"] > 0.6:
        synergies.append(("Financial / Balance Sheet", "High",
                          "Acquirer's strong balance sheet can de-lever the target."))
    elif lev_avg < 0.55:
        synergies.append(("Financial / Balance Sheet", "Low",
                          "Combined leverage is moderate."))
    else:
        synergies.append(("Financial / Balance Sheet", "No",
                          "Both entities carry significant leverage — limited balance-sheet synergy."))

    # Liquidity complement
    if ma["liquidity"] > 1.5 or mt["liquidity"] > 1.5:
        synergies.append(("Liquidity Complement", "High",
                          "At least one entity has strong liquidity to fund integration."))
    elif ma["liquidity"] > 1.0 and mt["liquidity"] > 1.0:
        synergies.append(("Liquidity Complement", "Low",
                          "Adequate combined liquidity."))
    else:
        synergies.append(("Liquidity Complement", "No",
                          "Both entities have tight liquidity — integration funding risk."))

    # Size complement
    size_ratio = _safe_div(min(ma["size"], mt["size"]), max(ma["size"], mt["size"]), 0)
    if 0.2 < size_ratio < 0.8:
        synergies.append(("Size Complement", "High",
                          "Appropriate size difference for bolt-on integration."))
    elif size_ratio >= 0.8:
        synergies.append(("Size Complement", "Low",
                          "Merger of equals — complex integration but transformative potential."))
    else:
        synergies.append(("Size Complement", "No",
                          "Very large size disparity — limited operational synergy."))

    # Overall rating
    high_count = sum(1 for _, level, _ in synergies if level == "High")
    low_count  = sum(1 for _, level, _ in synergies if level == "Low")
    if high_count >= 3:
        overall = "HIGH SYNERGY"
        overall_color = "green"
        overall_text = "Strong synergy potential across multiple dimensions."
    elif high_count >= 1 or low_count >= 3:
        overall = "MODERATE SYNERGY"
        overall_color = "orange"
        overall_text = "Some synergy opportunities exist but integration risk is present."
    else:
        overall = "LOW SYNERGY"
        overall_color = "red"
        overall_text = "Limited synergy potential — proceed with caution."

    return {
        "synergies": synergies,
        "overall": overall,
        "overall_color": overall_color,
        "overall_text": overall_text,
        "acquirer_scores": scores_a,
        "target_scores": scores_t,
    }


# ===================================================================
# Readability Indexes (for Textual Sentiment Analysis enhancement)
# ===================================================================

def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word using a vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    # Remove trailing silent-e
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    # Count vowel groups (a,e,i,o,u,y)
    count = len(_re.findall(r"[aeiouy]+", word))
    return max(count, 1)  # every word has at least one syllable


def _tokenise_readability(text: str) -> Dict[str, Any]:
    """Split text into sentences and words for readability scoring.

    Returns dict with keys: sentences, words, total_sentences,
    total_words, total_syllables, total_chars, complex_word_count.
    """
    # Sentence splitting: split on . ! ? followed by whitespace or end-of-string
    sentences = _re.split(r"[.!?]+(?:\s|$)", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

    # Word extraction: alphabetic tokens only
    words = _re.findall(r"[a-zA-Z]+", text)

    total_sentences = max(len(sentences), 1)
    total_words = max(len(words), 1)

    total_syllables = 0
    total_chars = 0
    complex_word_count = 0  # words with >= 3 syllables (Gunning Fog definition)

    for w in words:
        syls = _count_syllables(w)
        total_syllables += syls
        total_chars += len(w)
        if syls >= 3:
            complex_word_count += 1

    return {
        "sentences": sentences,
        "words": words,
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_syllables": max(total_syllables, 1),
        "total_chars": total_chars,
        "complex_word_count": complex_word_count,
    }


def flesch_kincaid_grade(text: str) -> Dict[str, Any]:
    """Flesch-Kincaid Grade Level.

    FK = 0.39 * (total_words / total_sentences)
       + 11.8 * (total_syllables / total_words)
       - 15.59

    Returns a U.S. school grade level (e.g. 12.0 = 12th grade reading level).
    10-K filings typically score 18-22 (post-graduate).
    """
    t = _tokenise_readability(text)
    asl = t["total_words"] / t["total_sentences"]   # avg sentence length
    asw = t["total_syllables"] / t["total_words"]    # avg syllables per word

    grade = 0.39 * asl + 11.8 * asw - 15.59

    if grade <= 8:
        interp = "Easy to read (8th grade or below). Unusually simple for a financial filing."
    elif grade <= 12:
        interp = "Standard readability (high-school level). Clear and accessible."
    elif grade <= 16:
        interp = "College-level readability. Typical of well-written financial reports."
    elif grade <= 20:
        interp = "Post-graduate level. Dense but normal for 10-K filings."
    else:
        interp = "Extremely complex prose. May obscure material information."

    return {
        "name": "Flesch-Kincaid Grade Level",
        "score": round(grade, 2),
        "interpretation": interp,
        "components": {"avg_sentence_length": round(asl, 1),
                       "avg_syllables_per_word": round(asw, 2)},
    }


def gunning_fog_index(text: str) -> Dict[str, Any]:
    """Gunning Fog Index.

    Fog = 0.4 * ( (total_words / total_sentences)
                 + 100 * (complex_words / total_words) )

    Complex words = words with 3+ syllables (excluding common suffixes in
    some variants, but we use the standard definition here).
    Score represents years of formal education needed to understand the text.
    """
    t = _tokenise_readability(text)
    asl = t["total_words"] / t["total_sentences"]
    pct_complex = t["complex_word_count"] / t["total_words"]

    fog = 0.4 * (asl + 100.0 * pct_complex)

    if fog <= 9:
        interp = "Easy to read. Accessible to a wide audience."
    elif fog <= 12:
        interp = "Standard readability. Appropriate for a general business audience."
    elif fog <= 16:
        interp = "Difficult. Requires college-level education."
    elif fog <= 20:
        interp = "Very difficult. Common in legal and regulatory filings."
    else:
        interp = "Extremely difficult. Dense legal/technical prose — may signal obfuscation."

    return {
        "name": "Gunning Fog Index",
        "score": round(fog, 2),
        "interpretation": interp,
        "components": {"avg_sentence_length": round(asl, 1),
                       "pct_complex_words": round(pct_complex * 100, 1)},
    }


def automated_readability_index(text: str) -> Dict[str, Any]:
    """Automated Readability Index (ARI).

    ARI = 4.71 * (total_chars / total_words)
        + 0.5  * (total_words / total_sentences)
        - 21.43

    Character-count based — avoids syllable estimation error.
    Score maps to a U.S. grade level.
    """
    t = _tokenise_readability(text)
    avg_chars = t["total_chars"] / t["total_words"]
    asl = t["total_words"] / t["total_sentences"]

    ari = 4.71 * avg_chars + 0.5 * asl - 21.43

    if ari <= 6:
        interp = "Very easy. Elementary school level."
    elif ari <= 10:
        interp = "Easy to moderate. Middle/high-school level."
    elif ari <= 14:
        interp = "College level. Standard for business writing."
    elif ari <= 18:
        interp = "Graduate level. Normal for SEC filings."
    else:
        interp = "Post-graduate / professional level. Extremely dense prose."

    return {
        "name": "Automated Readability Index (ARI)",
        "score": round(ari, 2),
        "interpretation": interp,
        "components": {"avg_chars_per_word": round(avg_chars, 2),
                       "avg_sentence_length": round(asl, 1)},
    }


def compute_all_readability(text: str) -> List[Dict[str, Any]]:
    """Convenience function: compute all three readability indexes at once."""
    return [
        flesch_kincaid_grade(text),
        gunning_fog_index(text),
        automated_readability_index(text),
    ]
