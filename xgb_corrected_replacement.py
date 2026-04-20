# === CORRECTED: XGBoost Altman Z-Score using correct ISDS_XGBoost_Real_Data_Report ===

# Out-of-sample validation metrics from ISDS_XGBoost_Real_Data_Report (April 2026).
# Source: weighted-average precision/recall/F1 from held-out 25% test set; roc_auc = test-set AUC.
MODEL_PERFORMANCE_STATS: Dict[str, Dict[str, float]] = {
    "Healthcare":    {"accuracy": 0.641, "precision": 0.645, "recall": 0.641, "f1": 0.642, "roc_auc": 0.695},
    "Technology":    {"accuracy": 0.658, "precision": 0.671, "recall": 0.658, "f1": 0.663, "roc_auc": 0.682},
    "Financial":     {"accuracy": 0.601, "precision": 0.606, "recall": 0.601, "f1": 0.602, "roc_auc": 0.642},
    "Manufacturing": {"accuracy": 0.655, "precision": 0.666, "recall": 0.655, "f1": 0.659, "roc_auc": 0.696},
    "Energy":        {"accuracy": 0.635, "precision": 0.635, "recall": 0.635, "f1": 0.635, "roc_auc": 0.664},
    "Construction":  {"accuracy": 0.689, "precision": 0.689, "recall": 0.689, "f1": 0.689, "roc_auc": 0.749},
    "Airline":       {"accuracy": 0.747, "precision": 0.749, "recall": 0.747, "f1": 0.747, "roc_auc": 0.869},
    "Agriculture":   {"accuracy": 0.696, "precision": 0.704, "recall": 0.696, "f1": 0.698, "roc_auc": 0.772},
}

# Exact probability thresholds from ISDS_XGBoost_Real_Data_Report Python code sections (p. 25-38).
# safe_thr = 70th percentile of predicted distress prob among healthy firms.
# dist_thr = 35th percentile of predicted distress prob among distressed firms.
# Note: ISDS-FIN has inverted thresholds (safe > dist) per the report — this is correct and
# reflects that Financial Services predictions cluster in a narrow band with effectively no grey zone.
_XGBOOST_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Healthcare":    {"safe": 0.470974, "distress": 0.543170},
    "Technology":    {"safe": 0.499104, "distress": 0.554779},
    "Financial":     {"safe": 0.514148, "distress": 0.510250},
    "Manufacturing": {"safe": 0.506821, "distress": 0.544550},
    "Energy":        {"safe": 0.442242, "distress": 0.603441},
    "Construction":  {"safe": 0.370449, "distress": 0.649558},
    "Airline":       {"safe": 0.084938, "distress": 0.927720},
    "Agriculture":   {"safe": 0.420772, "distress": 0.603567},
}

# Feature specs from ISDS_XGBoost_Real_Data_Report feature importance tables.
# Each tuple: (display_name, ratio_key, distress_direction, importance, role_note)
# distress_direction: -1 = higher ratio reduces distress; +1 = higher ratio increases distress.
_XGBOOST_FEATURE_SPECS: Dict[str, list] = {
    "Healthcare": [
        ("Sales / Total Assets",               "sale_ta", -1, 0.2098, "Asset turnover -- top predictor (20.98%)"),
        ("Retained Earnings / Total Assets",   "re_ta",   -1, 0.1635, "Cumulative profitability (16.35%)"),
        ("Equity / Total Liabilities",         "ceq_lt",  -1, 0.1406, "Solvency buffer (14.06%)"),
        ("Working Capital / Total Assets",     "wc_ta",   -1, 0.1240, "Liquidity cushion (12.40%)"),
        ("Operating Cash Flow / Total Assets", "ocf_at",  -1, 0.1212, "Cash generation (12.12%)"),
        ("Net Income / Total Assets",          "ni_at",   -1, 0.1212, "Net profitability (12.12%)"),
        ("EBIT / Total Assets",                "ebit_ta", -1, 0.1198, "Operating earnings (11.98%)"),
    ],
    "Technology": [
        ("Retained Earnings / Total Assets",   "re_ta",   -1, 0.1929, "Cumulative profitability -- top predictor (19.29%)"),
        ("Sales / Total Assets",               "sale_ta", -1, 0.1854, "Asset turnover (18.54%)"),
        ("Working Capital / Total Assets",     "wc_ta",   -1, 0.1586, "Liquidity cushion (15.86%)"),
        ("EBIT / Total Assets",                "ebit_ta", -1, 0.1585, "Operating earnings (15.85%)"),
        ("Equity / Total Liabilities",         "ceq_lt",  -1, 0.1582, "Solvency buffer (15.82%)"),
        ("Operating Cash Flow / Total Assets", "ocf_at",  -1, 0.1465, "Cash generation (14.65%)"),
    ],
    "Financial": [
        ("Total Liabilities / Total Assets",        "lt_ta",  +1, 0.2950, "Leverage -- top predictor (29.50%)"),
        ("Retained Earnings / Total Assets",        "re_ta",  -1, 0.2629, "Cumulative profitability (26.29%)"),
        ("Equity / Total Liabilities",              "ceq_lt", -1, 0.2307, "Solvency buffer (23.07%)"),
        ("Equity / Total Assets",                   "eq_ta",  -1, 0.2114, "Capital adequacy (21.14%)"),
        ("Loan Loss Provisions / Total Assets",     "pll_at", +1, 0.0000, "Credit risk -- null in Compustat extract (0.00%)"),
    ],
    "Manufacturing": [
        ("Retained Earnings / Total Assets",            "re_ta",  -1, 0.2398, "Cumulative profitability -- top predictor (23.98%)"),
        ("Sales / Total Assets",                        "sale_ta",-1, 0.2177, "Asset turnover (21.77%)"),
        ("Market Value Equity / Total Liabilities",     "mve_tl", -1, 0.1874, "Market cap used if available, else book equity (18.74%)"),
        ("EBIT / Total Assets",                         "ebit_ta",-1, 0.1778, "Operating earnings (17.78%)"),
        ("Working Capital / Total Assets",              "wc_ta",  -1, 0.1773, "Liquidity cushion (17.73%)"),
    ],
    "Energy": [
        ("Retained Earnings / Total Assets",   "re_ta",   -1, 0.1922, "Cumulative profitability -- top predictor (19.22%)"),
        ("Equity / Total Liabilities",         "ceq_lt",  -1, 0.1653, "Solvency buffer (16.53%)"),
        ("Sales / Total Assets",               "sale_ta", -1, 0.1639, "Asset turnover (16.39%)"),
        ("EBIT / Total Assets",                "ebit_ta", -1, 0.1623, "Operating earnings (16.23%)"),
        ("Operating Cash Flow / Total Assets", "ocf_at",  -1, 0.1586, "Cash generation (15.86%)"),
        ("Working Capital / Total Assets",     "wc_ta",   -1, 0.1578, "Liquidity cushion (15.78%)"),
    ],
    "Construction": [
        ("EBIT / Total Assets",                         "ebit_ta",-1, 0.1758, "Operating earnings -- top predictor (17.58%)"),
        ("Retained Earnings / Total Assets",            "re_ta",  -1, 0.1733, "Cumulative profitability (17.33%)"),
        ("Equity / Total Liabilities",                  "ceq_lt", -1, 0.1725, "Solvency buffer (17.25%)"),
        ("Working Capital / Total Assets",              "wc_ta",  -1, 0.1653, "Liquidity cushion (16.53%)"),
        ("Sales / Total Assets",                        "sale_ta",-1, 0.1649, "Asset turnover (16.49%)"),
        ("Operating Cash Flow / Total Liabilities",     "ocf_lt", -1, 0.1483, "Debt serviceability (14.83%)"),
    ],
    "Airline": [
        ("Sales / Total Assets",                        "sale_ta",-1, 0.2664, "Asset turnover -- top predictor (26.64%)"),
        ("EBIT / Total Assets",                         "ebit_ta",-1, 0.1659, "Operating earnings (16.59%)"),
        ("Working Capital / Total Assets",              "wc_ta",  -1, 0.1511, "Liquidity cushion (15.11%)"),
        ("Retained Earnings / Total Assets",            "re_ta",  -1, 0.1497, "Cumulative profitability (14.97%)"),
        ("Operating Cash Flow / Total Liabilities",     "ocf_lt", -1, 0.1347, "Debt serviceability (13.47%)"),
        ("Total Liabilities / Total Assets",            "lt_at",  +1, 0.1322, "Leverage (13.22%)"),
    ],
    "Agriculture": [
        ("Sales / Total Assets",               "sale_ta", -1, 0.2395, "Asset turnover -- top predictor (23.95%)"),
        ("Retained Earnings / Total Assets",   "re_ta",   -1, 0.2161, "Cumulative profitability (21.61%)"),
        ("Working Capital / Total Assets",     "wc_ta",   -1, 0.1862, "Liquidity cushion (18.62%)"),
        ("EBIT / Total Assets",                "ebit_ta", -1, 0.1795, "Operating earnings (17.95%)"),
        ("Equity / Total Liabilities",         "ceq_lt",  -1, 0.1786, "Solvency buffer (17.86%)"),
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
    """Run the ISDS-XGBoost Distress Score for the company's industry.

    Implements the 8 industry models from ISDS_XGBoost_Real_Data_Report (April 2026).
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
        variables.append(_var_row(display_name, ratio_val, contrib,
                                  f"XGB importance={importance:.4f}"))

    distress_prob = 1.0 / (1.0 + math.exp(-weighted_score))
    prob_pct = distress_prob * 100

    # ISDS-FIN has inverted thresholds (safe_thr > dist_thr) — correct per the report;
    # effectively no grey zone exists: below safe_thr = SAFE, above safe_thr = DISTRESS.
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
        "model_name":     f"ISDS-XGBoost Distress Score ({display_label})",
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
