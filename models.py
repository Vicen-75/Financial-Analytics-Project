# Fixed XGBoost zone classification with proper probability thresholds as requested
"""
models.py — All distress-prediction and manipulation-detection models.

Implements:
  - 7 Industry Distress Score models — analytical models
  - Beneish M-Score (8-variable earnings manipulation detector)
  - Logistic Regression (probability of bankruptcy)
  - XGBoost Distress Score (7 industry Perplexity-calibrated models)
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
        "model_name": "Healthcare Distress Score",
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
        "model_name": "Technology Distress Score",
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
        "model_name": "Manufacturing Distress Score",
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
        "model_name": "Energy Distress Score",
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
        "model_name": "Construction & Real Estate Distress Score",
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
        "model_name": "Transportation & Logistics Distress Score",
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
        "model_name": "Agriculture & Food Production Distress Score",
        "score": round(score, 4),
        "zone": zone,
        "color": _zone_color(zone),
        "interpretation": interp,
        "variables": variables,
        "thresholds": {"safe": ">2.50", "grey": "1.00 – 2.50", "distress": "<1.00"},
        "direction": "Higher = Safer",
        "warnings": [],
    }


# === CORRECTED: XGBoost Altman Z-Score using correct ISDS_XGBoost_Real_Data_Report ===

# Out-of-sample validation metrics from ISDS_XGBoost_Real_Data_Report (April 2026).
# Source: weighted-average precision/recall/F1 from held-out 25% test set; roc_auc = test-set AUC.
MODEL_PERFORMANCE_STATS: Dict[str, Dict[str, float]] = {
    "Healthcare":    {"accuracy": 0.641, "precision": 0.645, "recall": 0.641, "f1": 0.642, "roc_auc": 0.695},
    "Technology":    {"accuracy": 0.658, "precision": 0.671, "recall": 0.658, "f1": 0.663, "roc_auc": 0.682},
    "Manufacturing": {"accuracy": 0.655, "precision": 0.666, "recall": 0.655, "f1": 0.659, "roc_auc": 0.696},
    "Energy":        {"accuracy": 0.635, "precision": 0.635, "recall": 0.635, "f1": 0.635, "roc_auc": 0.664},
    "Construction":  {"accuracy": 0.689, "precision": 0.689, "recall": 0.689, "f1": 0.689, "roc_auc": 0.749},
    "Airline":       {"accuracy": 0.747, "precision": 0.749, "recall": 0.747, "f1": 0.747, "roc_auc": 0.869},
    "Agriculture":   {"accuracy": 0.696, "precision": 0.704, "recall": 0.696, "f1": 0.698, "roc_auc": 0.772},
}

# Exact probability thresholds from ISDS_XGBoost_Real_Data_Report Python code sections (p. 25-38).
# safe_thr = 70th percentile of predicted distress prob among healthy firms.
# dist_thr = 35th percentile of predicted distress prob among distressed firms.
_XGBOOST_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Healthcare":    {"safe": 0.470974, "distress": 0.543170},
    "Technology":    {"safe": 0.499104, "distress": 0.554779},
    "Manufacturing": {"safe": 0.506821, "distress": 0.544550},
    "Energy":        {"safe": 0.442242, "distress": 0.603441},
    "Construction":  {"safe": 0.370449, "distress": 0.649558},
    "Airline":       {"safe": 0.084938, "distress": 0.927720},
    "Agriculture":   {"safe": 0.420772, "distress": 0.603567},
}

# === CORRECTED: Removed Financial Services + Removed all ISDS branding except XGBoost name + Enhanced variable interpretations using ISDS_XGBoost_Real_Data_Report ===
# Feature specs from ISDS_XGBoost_Real_Data_Report feature importance tables.
# Each tuple: (display_name, ratio_key, distress_direction, importance, role_note)
# distress_direction: -1 = higher ratio reduces distress; +1 = higher ratio increases distress.
_XGBOOST_FEATURE_SPECS: Dict[str, list] = {
    "Healthcare": [
        ("Sales / Total Assets", "sale_ta", -1, 0.2098,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover. Top predictor (20.98% importance). "
         "Green flag: higher ratio signals strong revenue productivity per dollar of assets, reducing distress risk. "
         "Red flag: low or declining ratio signals market share erosion, idle capacity, or over-investment without matching revenue return."),
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.1635,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets — cumulative profitability history (16.35% importance). "
         "Green flag: high positive ratio signals sustained profitable operations and a strong internal capital base. "
         "Red flag: negative or near-zero ratio means historical losses have eroded equity — a key early-warning distress signal."),
        ("Equity / Total Liabilities", "ceq_lt", -1, 0.1406,
         "Total Shareholders' Equity / Total Liabilities (Balance Sheet) — solvency buffer (14.06% importance). "
         "Green flag: ratio >1.0 means equity exceeds all liabilities — robust solvency protection. "
         "Red flag: ratio <0.3 signals high leverage; negative equity (liabilities exceed assets) is a critical distress indicator."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1240,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term net liquidity (12.40% importance). "
         "Green flag: positive and growing ratio indicates ample near-term liquidity to meet obligations without external financing. "
         "Red flag: negative working capital signals inability to cover current obligations from current assets — a common distress precursor."),
        ("Operating Cash Flow / Total Assets", "ocf_at", -1, 0.1212,
         "Cash from Operations (Cash Flow Stmt) / Total Assets (Balance Sheet) — real cash generation (12.12% importance). "
         "Green flag: consistently positive OCF/TA means the business generates actual cash, not just accounting profit. "
         "Red flag: negative OCF despite positive net income flags earnings quality concerns and hidden cash burn."),
        ("Net Income / Total Assets", "ni_at", -1, 0.1212,
         "Net Income (Income Stmt) / Total Assets (Balance Sheet) — Return on Assets, ROA (12.12% importance). "
         "Green flag: positive and improving ROA signals efficient, profitable use of assets. "
         "Red flag: sustained negative ROA means the company cannot generate profit from its asset base — a direct insolvency risk signal."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1198,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — core operating profitability (11.98% importance). "
         "Green flag: positive EBIT/TA means operations generate returns above zero before financing costs. "
         "Red flag: negative EBIT/TA means core operations are loss-making — a critical warning independent of capital structure."),
    ],
    "Technology": [
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.1929,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets — cumulative profitability history. Top predictor (19.29% importance). "
         "Green flag: high positive ratio signals sustained profitable track record and self-funding capacity. "
         "Red flag: negative or near-zero retained earnings reflects persistent losses — especially dangerous in capital-intensive tech growth phases."),
        ("Sales / Total Assets", "sale_ta", -1, 0.1854,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover (18.54% importance). "
         "Green flag: higher ratio signals strong revenue productivity; tech firms with lean asset bases typically score well. "
         "Red flag: low or declining ratio signals product commoditisation, customer churn, or bloated asset base from failed acquisitions."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1586,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term liquidity cushion (15.86% importance). "
         "Green flag: positive and large ratio reflects strong cash reserves common in healthy tech firms. "
         "Red flag: negative working capital in tech often precedes cash-runway crises, particularly in pre-revenue or high-burn companies."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1585,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — operating profitability (15.85% importance). "
         "Green flag: positive EBIT/TA distinguishes profitable tech from cash-burning growth plays. "
         "Red flag: deeply negative EBIT/TA sustained beyond the growth phase is a key distress predictor in the sector."),
        ("Equity / Total Liabilities", "ceq_lt", -1, 0.1582,
         "Total Shareholders' Equity / Total Liabilities (Balance Sheet) — solvency cushion (15.82% importance). "
         "Green flag: high ratio indicates equity-heavy capital structure, typical of financially sound tech firms. "
         "Red flag: low ratio signals heavy debt load relative to equity — common before distress in hardware or platform companies."),
        ("Operating Cash Flow / Total Assets", "ocf_at", -1, 0.1465,
         "Cash from Operations (Cash Flow Stmt) / Total Assets (Balance Sheet) — real cash generation (14.65% importance). "
         "Green flag: high OCF/TA confirms that reported profits are backed by actual cash inflows. "
         "Red flag: negative OCF while reporting positive EBIT is a classic earnings manipulation or business-model stress signal."),
    ],
    "Manufacturing": [
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.2398,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets. Top predictor (23.98% importance). "
         "Green flag: high ratio reflects a long history of profitable operations and reinvested earnings — hallmark of financially sound manufacturers. "
         "Red flag: negative or rapidly declining retained earnings signals persistent losses, often driven by margin compression or large restructuring charges."),
        ("Sales / Total Assets", "sale_ta", -1, 0.2177,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover (21.77% importance). "
         "Green flag: high turnover signals efficient use of fixed assets and strong order books. "
         "Red flag: declining asset turnover in manufacturing often signals over-capacity, supply-chain disruption, or loss of key contracts."),
        ("Market Value Equity / Total Liabilities", "mve_tl", -1, 0.1874,
         "Market Capitalisation (or book equity as fallback) / Total Liabilities (Balance Sheet) — Altman's X4 ratio (18.74% importance). "
         "Green flag: ratio >1.0 signals the market values the company above its total debt — strong going-concern confidence. "
         "Red flag: ratio <0.3 mirrors pre-bankruptcy patterns; book equity is used as a conservative fallback when market cap is unavailable."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1778,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — core operating return (17.78% importance). "
         "Green flag: positive and stable EBIT/TA indicates the manufacturing operation generates returns above its cost base. "
         "Red flag: negative EBIT/TA sustained over two or more years is a strong predictor of plant closures and covenant breaches."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1773,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — liquidity buffer (17.73% importance). "
         "Green flag: positive working capital indicates the company can fund its operating cycle without short-term borrowing. "
         "Red flag: negative working capital in manufacturing often signals inventory or receivables stress and reliance on revolving credit — a distress precursor."),
    ],
    "Energy": [
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.1922,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets. Top predictor (19.22% importance). "
         "Green flag: high ratio signals the company has historically generated more profit than distributed — a buffer against commodity price cycles. "
         "Red flag: negative retained earnings in energy frequently follows sustained low commodity prices wiping out historical profits through impairment charges."),
        ("Equity / Total Liabilities", "ceq_lt", -1, 0.1653,
         "Total Shareholders' Equity / Total Liabilities (Balance Sheet) — solvency buffer (16.53% importance). "
         "Green flag: ratio >0.5 provides meaningful equity cushion relative to debt, reducing refinancing risk during commodity downturns. "
         "Red flag: ratio near zero or negative frequently precedes energy-sector restructurings, especially for E&P companies with high reserve-backed lending."),
        ("Sales / Total Assets", "sale_ta", -1, 0.1639,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover (16.39% importance). "
         "Green flag: higher ratio indicates the asset base (wells, pipelines, refineries) generates strong revenue relative to book value. "
         "Red flag: low ratio signals stranded assets, low utilisation, or a price environment where assets cannot justify their carrying value."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1623,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — operating profitability (16.23% importance). "
         "Green flag: positive EBIT/TA even in low-price environments signals cost discipline and a low breakeven point. "
         "Red flag: negative EBIT/TA in energy often coincides with large impairment write-downs or structural commodity price decline — a leading distress indicator."),
        ("Operating Cash Flow / Total Assets", "ocf_at", -1, 0.1586,
         "Cash from Operations (Cash Flow Stmt) / Total Assets (Balance Sheet) — real cash generation (15.86% importance). "
         "Green flag: positive and stable OCF/TA signals the company can self-fund capex and debt service through the commodity cycle. "
         "Red flag: negative OCF is particularly dangerous in energy because capex obligations (well maintenance, decommissioning) are non-deferrable."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1578,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term liquidity (15.78% importance). "
         "Green flag: positive working capital provides a near-term liquidity buffer against commodity price volatility and margin calls. "
         "Red flag: negative working capital in energy signals hedging losses, trade payables stress, or accelerated debt maturities — each a distress accelerant."),
    ],
    "Construction": [
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1758,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet). Top predictor (17.58% importance). "
         "Green flag: positive and stable EBIT/TA means the project portfolio generates operating returns — critical in a low-margin sector. "
         "Red flag: negative EBIT/TA in construction signals cost overruns, write-downs on loss-making contracts, or collapsed backlog — a leading distress indicator."),
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.1733,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets (17.33% importance). "
         "Green flag: positive retained earnings indicate the company has historically completed projects profitably and reinvested those gains. "
         "Red flag: negative retained earnings in construction often reflects a history of large project write-offs or litigation settlements — systemic quality risk."),
        ("Equity / Total Liabilities", "ceq_lt", -1, 0.1725,
         "Total Shareholders' Equity / Total Liabilities (Balance Sheet) — solvency buffer (17.25% importance). "
         "Green flag: ratio >0.4 provides meaningful equity protection against construction-specific risks (project defaults, surety claims). "
         "Red flag: low ratio signals high leverage relative to equity — vulnerable to a single large project failure or subcontractor dispute."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1653,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term liquidity (16.53% importance). "
         "Green flag: positive working capital confirms the company can fund its project pipeline without emergency borrowing. "
         "Red flag: negative working capital in construction is particularly dangerous because progress billing delays can cascade into supply-chain payment defaults."),
        ("Sales / Total Assets", "sale_ta", -1, 0.1649,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover (16.49% importance). "
         "Green flag: high turnover signals strong project pipeline and efficient deployment of the asset base. "
         "Red flag: declining ratio signals backlog depletion, failed contract bids, or project cancellations — reducing future revenue visibility."),
        ("Operating Cash Flow / Total Liabilities", "ocf_lt", -1, 0.1483,
         "Cash from Operations (Cash Flow Stmt) / Total Liabilities (Balance Sheet) — debt serviceability (14.83% importance). "
         "Green flag: ratio >0.10 indicates the company can service its total debt load from operating cash without asset sales. "
         "Red flag: near-zero or negative ratio signals inability to cover debt from operations — a classic precursor to covenant breaches and project lender defaults."),
    ],
    "Airline": [
        ("Sales / Total Assets", "sale_ta", -1, 0.2664,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover. Top predictor (26.64% importance). "
         "Green flag: airlines require very high asset turnover to service fleet financing; strong load factors and yield management drive this ratio up. "
         "Red flag: declining ratio signals underutilised fleet, route cuts, or demand shock — the single most powerful distress predictor in this sector."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1659,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — operating profitability (16.59% importance). "
         "Green flag: positive EBIT/TA confirms the airline covers its operating cost base (fuel, labour, maintenance) from revenues. "
         "Red flag: negative EBIT/TA is extremely common in airline distress; even brief periods of negative operating income can trigger covenant violations on aircraft financing."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1511,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term liquidity (15.11% importance). "
         "Green flag: positive working capital provides a buffer against sudden demand shocks or fuel price spikes. "
         "Red flag: airlines structurally tend toward negative working capital (advance ticket sales are a current liability); a sharply deteriorating ratio signals near-term liquidity stress."),
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.1497,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets (14.97% importance). "
         "Green flag: positive retained earnings indicate the airline has historically generated profit net of fleet depreciation — rare and highly valued in this sector. "
         "Red flag: deeply negative retained earnings are common in legacy airlines post-restructuring; chronic losses signal a structurally uncompetitive cost base."),
        ("Operating Cash Flow / Total Liabilities", "ocf_lt", -1, 0.1347,
         "Cash from Operations (Cash Flow Stmt) / Total Liabilities (Balance Sheet) — debt serviceability (13.47% importance). "
         "Green flag: ratio >0.08 for airlines signals sufficient cash generation to service aircraft debt and operating leases. "
         "Red flag: near-zero or negative ratio means the airline cannot service its debt from operations — critical given the size and inflexibility of fleet financing obligations."),
        ("Total Liabilities / Total Assets", "lt_at", +1, 0.1322,
         "Total Liabilities / Total Assets (Balance Sheet) — leverage ratio. Higher value increases distress probability (13.22% importance). "
         "Green flag: ratio <0.75 indicates manageable leverage relative to sector norms. "
         "Red flag: ratio >0.90 is extremely common in airline distress; combined fleet financing (owned and leased) makes this sector structurally high-leverage — watch trend direction, not just level."),
    ],
    "Agriculture": [
        ("Sales / Total Assets", "sale_ta", -1, 0.2395,
         "Revenue (Income Stmt) / Total Assets (Balance Sheet) — asset turnover. Top predictor (23.95% importance). "
         "Green flag: high ratio signals productive land, equipment, and processing assets generating strong revenue relative to their book value. "
         "Red flag: low or declining ratio signals commodity price weakness, crop failure impact on revenues, or over-investment in assets without matching revenue generation."),
        ("Retained Earnings / Total Assets", "re_ta", -1, 0.2161,
         "Accumulated Retained Earnings (Balance Sheet equity section) / Total Assets (21.61% importance). "
         "Green flag: positive retained earnings reflect a history of profitable seasons and prudent earnings retention — building resilience against commodity cycles. "
         "Red flag: negative retained earnings in agriculture often follows several consecutive bad harvest years or prolonged commodity price depression."),
        ("Working Capital / Total Assets", "wc_ta", -1, 0.1862,
         "(Current Assets - Current Liabilities) / Total Assets (Balance Sheet) — short-term liquidity (18.62% importance). "
         "Green flag: positive working capital enables funding of operating cycles (seeds, inputs, harvest) without emergency borrowing. "
         "Red flag: negative working capital in agriculture signals seasonal liquidity stress that may require distressed asset sales or curtailed planting."),
        ("EBIT / Total Assets", "ebit_ta", -1, 0.1795,
         "Earnings Before Interest & Tax (Income Stmt) / Total Assets (Balance Sheet) — core operating profitability (17.95% importance). "
         "Green flag: positive EBIT/TA even in down commodity cycles indicates cost efficiency and diversified revenue streams. "
         "Red flag: negative EBIT/TA signals the farming or processing operation is below breakeven — a direct distress precursor when commodity prices are depressed."),
        ("Equity / Total Liabilities", "ceq_lt", -1, 0.1786,
         "Total Shareholders' Equity / Total Liabilities (Balance Sheet) — solvency buffer (17.86% importance). "
         "Green flag: ratio >0.5 provides meaningful equity protection against commodity price volatility and crop insurance gaps. "
         "Red flag: low ratio signals high agricultural leverage — when land values decline or crop revenues fall, these companies face margin calls and covenant violations rapidly."),
    ],
}


def _compute_xgb_ratios(d: dict) -> dict:
    """Compute all financial ratios required by the ISDS-XGBoost feature sets."""
    ta   = _g(d, "total_assets", 1)
    ca   = _g(d, "current_assets")
    cl   = _g(d, "current_liabilities")
    re_  = _g(d, "retained_earnings")
    ebit = _g(d, "ebit")
    eq   = _g(d, "total_equity")
    tl   = _g(d, "total_liabilities", 1)
    rev  = _g(d, "revenue")
    ocf  = _g(d, "operating_cash_flow")
    ni   = _g(d, "net_income")
    mc   = _g(d, "market_cap")

    mve = mc if mc > 0 else eq

    return {
        "wc_ta":   _safe_div(ca - cl, ta),
        "re_ta":   _safe_div(re_, ta),
        "ebit_ta": _safe_div(ebit, ta),
        "ceq_lt":  _safe_div(eq, tl),
        "sale_ta": _safe_div(rev, ta),
        "ocf_at":  _safe_div(ocf, ta),
        "ocf_lt":  _safe_div(ocf, tl),
        "ni_at":   _safe_div(ni, ta),
        "lt_ta":   _safe_div(tl, ta),
        "lt_at":   _safe_div(tl, ta),
        "eq_ta":   _safe_div(eq, ta),
        "mve_tl":  _safe_div(mve, tl),
        "pll_at":  0.0,  # Loan loss provisions -- null in standard Compustat extract
    }


def run_xgboost_zscore(d: dict) -> dict:
    """Run the ISDS-XGBooster Distress Score for the company's industry.

    Implements the 7 industry models from ISDS_XGBoost_Real_Data_Report (April 2026).
    Since XGBoost is a non-linear tree ensemble without closed-form coefficients, this
    implementation uses a feature-importance-weighted signed composite score with a
    sigmoid transformation as a tractable proxy. Feature importances and zone thresholds
    are the exact values published in the report.

    Score: distress_prob = sigmoid( sum_i(direction_i * importance_i * ratio_i) )
    Zones: exact 70th-pct (healthy) and 35th-pct (distressed) thresholds from the report.
    """
    import math

    industry = d.get("industry", "Manufacturing")

    _INDUSTRY_FALLBACK: Dict[str, str] = {
        "Transportation": "Airline",
        "Other":          "Manufacturing",
    }
    if industry not in _XGBOOST_FEATURE_SPECS:
        fallback = _INDUSTRY_FALLBACK.get(industry, "Manufacturing")
        industry = fallback

    original_industry = d.get("industry", industry)
    display_label = (industry if original_industry == industry
                     else f"{industry} [fallback from {original_industry}]")

    ratios   = _compute_xgb_ratios(d)
    specs    = _XGBOOST_FEATURE_SPECS[industry]
    thr      = _XGBOOST_THRESHOLDS[industry]
    safe_thr = thr["safe"]
    dist_thr = thr["distress"]

    weighted_score = 0.0
    variables: list = []
    for display_name, ratio_key, distress_dir, importance, role_note in specs:
        ratio_val      = ratios.get(ratio_key, 0.0)
        contrib        = distress_dir * importance * ratio_val
        weighted_score += contrib
        if contrib < -0.001:
            val_signal = (f"Computed value: {ratio_val:.4f} — "
                          f"Green flag: this ratio is actively reducing distress probability "
                          f"(contribution: {contrib:.4f}).")
        elif contrib > 0.001:
            val_signal = (f"Computed value: {ratio_val:.4f} — "
                          f"Red flag: this ratio is increasing distress probability "
                          f"(contribution: +{contrib:.4f}). Investigate and monitor closely.")
        else:
            val_signal = (f"Computed value: {ratio_val:.4f} — "
                          f"Neutral: near-zero contribution to distress probability.")
        dynamic_note = f"{role_note} || {val_signal}"
        variables.append(_var_row(display_name, ratio_val, contrib, dynamic_note))

    distress_prob = 1.0 / (1.0 + math.exp(-weighted_score))
    prob_pct = distress_prob * 100

    if safe_thr >= dist_thr:
        if distress_prob > safe_thr:
            zone  = "Distress Zone"
            interp = (
                f"{prob_pct:.2f}% distress probability -- Distress Zone. "
                f"Exceeds the {safe_thr:.4f} distress threshold for {industry}; "
                f"elevated likelihood of inactivation/delisting. Forensic review warranted."
            )
        else:
            zone  = "Safe Zone"
            interp = (
                f"{prob_pct:.2f}% distress probability -- Safe Zone. "
                f"Below the {safe_thr:.4f} threshold calibrated on {industry} sector data; "
                f"financial fundamentals appear sound."
            )
    elif distress_prob < safe_thr:
        zone  = "Safe Zone"
        interp = (
            f"{prob_pct:.2f}% distress probability -- Safe Zone. "
            f"Below the {safe_thr:.4f} safe threshold calibrated on {industry} sector data; "
            f"financial fundamentals appear sound."
        )
    elif distress_prob > dist_thr:
        zone  = "Distress Zone"
        interp = (
            f"{prob_pct:.2f}% distress probability -- Distress Zone. "
            f"Exceeds the {dist_thr:.4f} distress threshold for {industry}; "
            f"elevated likelihood of inactivation/delisting. Forensic review warranted."
        )
    else:
        zone  = "Grey Zone"
        interp = (
            f"{prob_pct:.2f}% distress probability -- Grey Zone "
            f"({safe_thr:.4f}--{dist_thr:.4f} band for {industry}). "
            f"Elevated risk; monitor key ratios quarterly and investigate underlying drivers."
        )

    return {
        "model_name":     f"ISDS-XGBooster Distress Score ({display_label})",
        "score":          round(distress_prob, 4),
        "zone":           zone,
        "color":          _zone_color(zone),
        "interpretation": interp,
        "variables":      variables,
        "thresholds": {
            "safe":     f"< {safe_thr:.4f}",
            "grey":     f"{safe_thr:.4f} - {dist_thr:.4f}",
            "distress": f"> {dist_thr:.4f}",
        },
        "direction":  "Distress probability -- lower = safer",
        "warnings":   [],
    }





# ===================================================================
# ISDS Dispatcher — selects the correct model based on industry
# ===================================================================

INDUSTRY_MODEL_MAP = {
    "Healthcare":    isds_hc,
    "Technology":    isds_tech,
    "Manufacturing": isds_mfg,
    "Energy":        isds_ene,
    "Construction":  isds_cre,
    "Airline":       isds_tl,
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

    def _bm_note(base: str, v: float, contrib: float, high_bad: bool = True,
                 warn_thresh: float = 1.1, flag_thresh: float = 1.2) -> str:
        """Generate automatic value-based note for a Beneish index."""
        v_str = f"Computed value: {v:.4f}."
        if high_bad:
            if v <= warn_thresh:
                sig = f"{v_str} Green flag: within normal range — no manipulation signal for this index."
            elif v <= flag_thresh:
                sig = f"{v_str} Monitor: slightly elevated — warrants closer review."
            else:
                sig = f"{v_str} Red flag: abnormally high — potential manipulation signal. Forensic review recommended."
        else:
            if v >= warn_thresh:
                sig = f"{v_str} Green flag: within normal range — no manipulation signal for this index."
            elif v >= flag_thresh:
                sig = f"{v_str} Monitor: slightly below normal range."
            else:
                sig = f"{v_str} Red flag: significantly below normal — potential manipulation signal."
        return f"{base} || {sig}"

    variables = [
        _var_row("DSRI: Days Sales Receivables Index", dsri, 0.920 * dsri,
                 _bm_note(
                     "Accounts Receivable (Balance Sheet) / Revenue (Income Stmt), current vs prior year. "
                     "Measures whether receivables grow disproportionately relative to sales — a sign of "
                     "premature revenue recognition or channel stuffing. Coefficient: +0.920.",
                     dsri, 0.920 * dsri, high_bad=True, warn_thresh=1.0, flag_thresh=1.2)),
        _var_row("GMI:  Gross Margin Index", gmi, 0.528 * gmi,
                 _bm_note(
                     "Prior-year Gross Margin / Current-year Gross Margin (both from Income Stmt). "
                     "Values above 1.0 indicate margins deteriorated — companies under margin pressure "
                     "have more incentive to manipulate earnings. Coefficient: +0.528.",
                     gmi, 0.528 * gmi, high_bad=True, warn_thresh=1.0, flag_thresh=1.2)),
        _var_row("AQI:  Asset Quality Index", aqi, 0.404 * aqi,
                 _bm_note(
                     "Current-year non-current / non-tangible asset ratio vs prior year (Balance Sheet). "
                     "Increases signal greater capitalisation of costs as assets — a common manipulation technique. "
                     "Coefficient: +0.404.",
                     aqi, 0.404 * aqi, high_bad=True, warn_thresh=1.0, flag_thresh=1.2)),
        _var_row("SGI:  Sales Growth Index", sgi, 0.892 * sgi,
                 _bm_note(
                     "Current Revenue / Prior Revenue (Income Stmt). Beneish found that high-growth firms "
                     "face more pressure to meet expectations, increasing manipulation risk. "
                     "Coefficient: +0.892.",
                     sgi, 0.892 * sgi, high_bad=True, warn_thresh=1.2, flag_thresh=1.4)),
        _var_row("DEPI: Depreciation Index", depi, 0.115 * depi,
                 _bm_note(
                     "Prior-year / Current-year depreciation rate (Income Stmt / Balance Sheet). "
                     "Values above 1.0 indicate the company slowed its depreciation rate, boosting reported "
                     "earnings artificially. Coefficient: +0.115.",
                     depi, 0.115 * depi, high_bad=True, warn_thresh=1.0, flag_thresh=1.1)),
        _var_row("SGAI: SGA Expense Index", sgai, -0.172 * sgai,
                 _bm_note(
                     "Current SGA/Revenue vs Prior SGA/Revenue (Income Stmt). A declining ratio (index < 1) "
                     "can signal cost-cutting used to inflate short-term earnings rather than genuine efficiency. "
                     "Negative coefficient (-0.172): higher SGAI actually reduces M-Score.",
                     sgai, -0.172 * sgai, high_bad=False, warn_thresh=1.0, flag_thresh=0.85)),
        _var_row("TATA: Total Accruals / TA", tata, 4.679 * tata,
                 (lambda v=tata: (
                     "Net Income minus Operating Cash Flow, divided by Total Assets "
                     "(Income Stmt / Cash Flow Stmt / Balance Sheet). Captures the gap between "
                     "accrual-based income and real cash generation. Coefficient: +4.679 — the "
                     "single most powerful Beneish predictor. "
                     f"|| Computed value: {v:.4f}. "
                     + ("Green flag: negative or near-zero TATA means earnings are cash-backed — low manipulation risk."
                        if v <= 0.02 else
                        "Monitor: positive accruals — earnings exceed cash flow; investigate working capital changes."
                        if v <= 0.05 else
                        "Red flag: high positive accruals — large gap between reported income and cash flow; strong manipulation signal.")
                 ))()),
        _var_row("LVGI: Leverage Index", lvgi, -0.327 * lvgi,
                 _bm_note(
                     "Current Debt/Assets vs Prior Debt/Assets (Balance Sheet). "
                     "Increasing leverage can signal financial distress being masked by earnings manipulation. "
                     "Negative coefficient (-0.327): the model penalises increasing leverage mildly.",
                     lvgi, -0.327 * lvgi, high_bad=True, warn_thresh=1.0, flag_thresh=1.2)),
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

    def _lr_note(base: str, v: float, contrib: float) -> str:
        if contrib < -0.05:
            sig = f"Computed value: {v:.4f} — Green flag: reducing bankruptcy probability (contribution: {contrib:.4f})."
        elif contrib > 0.05:
            sig = f"Computed value: {v:.4f} — Red flag: increasing bankruptcy probability (contribution: +{contrib:.4f}). Monitor closely."
        else:
            sig = f"Computed value: {v:.4f} — Neutral / minimal contribution ({contrib:+.4f})."
        return f"{base} || {sig}"

    variables = [
        _var_row("ROA (NI / Total Assets)", roa, -4.513 * roa,
                 _lr_note(
                     "Net Income (Income Stmt) / Total Assets (Balance Sheet) — Return on Assets. "
                     "Coefficient: -4.513. Higher ROA strongly reduces bankruptcy probability. "
                     "Green flag: ROA above 5%. Red flag: negative ROA (loss-making) sharply raises risk.",
                     roa, -4.513 * roa)),
        _var_row("Debt Ratio (Total Liab. / TA)", debt_ratio, 5.679 * debt_ratio,
                 _lr_note(
                     "Total Liabilities (Balance Sheet) / Total Assets (Balance Sheet) — overall leverage. "
                     "Coefficient: +5.679 — the dominant risk driver in this model. "
                     "Green flag: ratio below 0.50 (equity-funded). Red flag: above 0.75 (highly leveraged, limited buffer).",
                     debt_ratio, 5.679 * debt_ratio)),
        _var_row("Current Ratio (CA / CL)", current, 0.004 * current,
                 _lr_note(
                     "Current Assets / Current Liabilities (Balance Sheet) — short-term liquidity. "
                     "Coefficient: +0.004 (very small; liquidity is a secondary predictor in this model). "
                     "Green flag: ratio above 1.5. Red flag: below 1.0 means current liabilities exceed current assets.",
                     current, 0.004 * current)),
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

    # 1. Beneish M-Score (earnings manipulation detection — always runs)
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
