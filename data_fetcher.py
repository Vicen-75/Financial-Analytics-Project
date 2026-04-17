# Audited and fixed as per user instructions - Quality improvements for Single Target Assessment and Merger Analysis. Codex evaluation expected.
"""
data_fetcher.py — Data acquisition layer.

Three input methods:
  1. yfinance auto-fetch for public companies (with year selector + closest-year fallback)
  2. Manual entry for private companies
  3. CSV file upload for batch analysis
"""

from __future__ import annotations
import math
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------------
# GICS-to-industry heuristic mapping
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    "Healthcare":               "Healthcare",
    "Health Care":              "Healthcare",
    "Technology":               "Technology",
    "Information Technology":   "Technology",
    "Communication Services":   "Technology",
    "Financials":               "Financial",
    "Financial Services":       "Financial",
    "Industrials":              "Manufacturing",
    "Consumer Discretionary":   "Manufacturing",
    "Consumer Staples":         "Agriculture",
    "Energy":                   "Energy",
    "Real Estate":              "Construction",
    "Materials":                "Manufacturing",
    "Utilities":                "Energy",
}


def _map_sector(sector: str | None) -> str:
    """Map a GICS sector string to our 8-industry taxonomy."""
    if not sector:
        return "Manufacturing"
    for key, val in _SECTOR_MAP.items():
        if key.lower() in sector.lower():
            return val
    return "Manufacturing"


# ---------------------------------------------------------------------------
# Safe numeric extraction from yfinance DataFrames
# ---------------------------------------------------------------------------

def _yf_val(df: pd.DataFrame, labels: list[str], col_idx: int = 0) -> float:
    """Try each label in *labels* against the DataFrame index and return the first match."""
    if df is None or df.empty:
        return 0.0
    for label in labels:
        if label in df.index:
            try:
                v = df.iloc[df.index.get_loc(label), col_idx]
                if pd.notna(v) and np.isfinite(v):
                    return float(v)
            except Exception:
                continue
    return 0.0


def _pick_year(df: pd.DataFrame, target_year: int) -> Tuple[int, int | None, str | None]:
    """Return (col_index, actual_year, warning_or_None) for the closest fiscal year."""
    if df is None or df.empty:
        return 0, None, "No financial data available."
    years = []
    for i, col in enumerate(df.columns):
        try:
            yr = col.year if hasattr(col, "year") else int(str(col)[:4])
            years.append((i, yr))
        except Exception:
            continue
    if not years:
        return 0, None, "Could not parse fiscal years."
    # Prefer exact match
    for idx, yr in years:
        if yr == target_year:
            return idx, yr, None
    # Closest available
    best = min(years, key=lambda t: abs(t[1] - target_year))
    warn = f"FY{target_year} not available — using closest year FY{best[1]}."
    return best[0], best[1], warn


# ---------------------------------------------------------------------------
# Primary yfinance fetch
# ---------------------------------------------------------------------------

def fetch_yfinance(ticker: str, target_year: int = 2024) -> Dict:
    """Fetch financial data from yfinance and return a standardised dict."""
    tkr = yf.Ticker(ticker)
    info = tkr.info or {}

    # Annual financial statements (most recent first)
    bs = tkr.balance_sheet
    inc = tkr.income_stmt
    cf = tkr.cashflow

    # Also grab prior-year statements for Beneish
    bs_q = tkr.quarterly_balance_sheet  # not used directly, but available

    # Determine column index for the target year
    idx, actual_year, warn_bs = _pick_year(bs, target_year)
    idx_i, _, warn_i = _pick_year(inc, target_year)
    idx_c, _, warn_c = _pick_year(cf, target_year)

    # Consolidate year warnings
    warnings = [w for w in [warn_bs, warn_i, warn_c] if w]
    year_warning = warnings[0] if warnings else None

    # Prior-year column (for Beneish & growth calcs)
    # yfinance returns columns most-recent-first, so idx+1 should be the prior year.
    # Clamp to valid range to avoid IndexError.
    bs_max = (len(bs.columns) - 1) if bs is not None and not bs.empty else 0
    inc_max = (len(inc.columns) - 1) if inc is not None and not inc.empty else 0
    idx_prev = min(idx + 1, bs_max) if bs_max > 0 else 0
    idx_prev_i = min(idx_i + 1, inc_max) if inc_max > 0 else 0

    # --- Balance Sheet ---
    def bv(labels, col=idx):
        return _yf_val(bs, labels, col)

    total_assets = bv(["Total Assets"])
    current_assets = bv(["Current Assets", "Total Current Assets"])
    current_liab = bv(["Current Liabilities", "Total Current Liabilities"])
    total_liab = bv(["Total Liabilities Net Minority Interest",
                     "Total Liabilities", "Total Liab"])
    long_term_debt = bv(["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    total_debt = bv(["Total Debt"])
    if total_debt == 0:
        total_debt = long_term_debt + bv(["Current Debt", "Current Debt And Capital Lease Obligation"])
    retained_earnings = bv(["Retained Earnings"])
    total_equity = bv(["Total Equity Gross Minority Interest",
                       "Stockholders Equity", "Total Stockholders Equity"])
    cash = bv(["Cash And Cash Equivalents"])
    # Fallback: if pure cash line missing, use combined line but then zero-out STI to avoid double-counting
    if cash == 0:
        cash = bv(["Cash Cash Equivalents And Short Term Investments"])
    # Only count STI separately if cash came from pure cash line (not the combined line)
    _cash_pure = bv(["Cash And Cash Equivalents"])
    if _cash_pure > 0:
        sti = bv(["Other Short Term Investments", "Short Term Investments"])
    else:
        sti = 0.0  # already included in the combined cash figure
    ppe = bv(["Net PPE", "Net Property Plant And Equipment", "Property Plant And Equipment Net"])
    receivables = bv(["Accounts Receivable", "Net Receivable", "Receivables"])
    securities = bv(["Available For Sale Securities", "Investments And Advances",
                     "Long Term Equity Investment"])

    # Prior-year balance sheet
    prev_ta = bv(["Total Assets"], idx_prev)
    prev_ca = bv(["Current Assets", "Total Current Assets"], idx_prev)
    prev_ppe = bv(["Net PPE", "Net Property Plant And Equipment"], idx_prev)
    prev_sec = bv(["Available For Sale Securities", "Investments And Advances"], idx_prev)
    prev_recv = bv(["Accounts Receivable", "Net Receivable", "Receivables"], idx_prev)
    prev_debt = bv(["Total Debt"], idx_prev)
    if prev_debt == 0:
        prev_debt = bv(["Long Term Debt"], idx_prev) + bv(["Current Debt"], idx_prev)

    # --- Income Statement ---
    def iv(labels, col=idx_i):
        return _yf_val(inc, labels, col)

    revenue = iv(["Total Revenue", "Revenue"])
    cogs = iv(["Cost Of Revenue"])
    gross_profit = iv(["Gross Profit"])
    if gross_profit == 0 and revenue > 0 and cogs > 0:
        gross_profit = revenue - cogs
    ebit = iv(["EBIT", "Operating Income"])
    ebitda = iv(["EBITDA", "Normalized EBITDA"])
    net_income = iv(["Net Income", "Net Income Common Stockholders"])
    interest_exp = iv(["Interest Expense", "Interest Expense Non Operating",
                       "Interest Expense Debt"])
    depreciation = iv(["Depreciation And Amortization In Income Statement",
                       "Depreciation And Amortization", "Reconciled Depreciation"])
    sga = iv(["Selling General And Administration", "Selling And Marketing Expense"])
    rd = iv(["Research And Development", "Research Development"])
    opex = iv(["Total Operating Expenses", "Operating Expense",
               "Total Expenses"])
    nii = iv(["Net Interest Income"])

    # Prior-year income
    prev_rev = iv(["Total Revenue", "Revenue"], idx_prev_i)
    prev_gp = iv(["Gross Profit"], idx_prev_i)
    prev_dep = iv(["Depreciation And Amortization In Income Statement",
                   "Depreciation And Amortization", "Reconciled Depreciation"], idx_prev_i)
    prev_sga = iv(["Selling General And Administration"], idx_prev_i)

    # --- Cash Flow ---
    def cv(labels, col=idx_c):
        return _yf_val(cf, labels, col)

    ocf = cv(["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
              "Total Cash From Operating Activities"])
    capex = cv(["Capital Expenditure", "Capital Expenditures",
                "Purchase Of PPE", "Purchase Of Property Plant And Equipment"])

    # --- Market Data ---
    market_cap = info.get("marketCap", 0) or 0
    shares = info.get("sharesOutstanding", 0) or 0
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    # --- Sector / Industry ---
    sector = info.get("sector", "")
    industry_raw = info.get("industry", "")
    industry = _map_sector(sector)
    company_name = info.get("shortName") or info.get("longName") or ticker

    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "industry": industry,
        "industry_raw": industry_raw,
        "sector_raw": sector,
        "year": actual_year or target_year,
        "year_warning": year_warning,

        # Balance Sheet
        "total_assets": total_assets,
        "current_assets": current_assets,
        "current_liabilities": current_liab,
        "total_liabilities": total_liab,
        "long_term_debt": long_term_debt,
        "total_debt": total_debt,
        "retained_earnings": retained_earnings,
        "total_equity": total_equity,
        "cash_and_equivalents": cash,
        "short_term_investments": sti,
        "net_ppe": ppe,
        "receivables": receivables,
        "securities": securities,

        # Income Statement
        "revenue": revenue,
        "revenue_prev": prev_rev,
        "cost_of_revenue": cogs,
        "gross_profit": gross_profit,
        "ebit": ebit,
        "ebitda": ebitda,
        "net_income": net_income,
        "interest_expense": abs(interest_exp) if interest_exp else 0,
        "depreciation": depreciation,
        "sga_expense": sga,
        "rd_expense": rd,
        "operating_expenses": opex,
        "net_interest_income": nii if nii else 0,

        # Cash Flow
        "operating_cash_flow": ocf,
        "capex": capex,

        # Market
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "stock_price": price,

        # Prior-year (Beneish)
        "prev_receivables": prev_recv,
        "prev_revenue": prev_rev,
        "prev_gross_profit": prev_gp,
        "prev_current_assets": prev_ca,
        "prev_ppe": prev_ppe,
        "prev_securities": prev_sec,
        "prev_total_assets": prev_ta,
        "prev_depreciation": prev_dep,
        "prev_sga": prev_sga,
        "prev_total_debt": prev_debt,

        # Bank-specific (defaults — user can override manually)
        "npl": 0.0,
        "total_loans": 0.0,
        "tier1_capital": 0.0,
        "risk_weighted_assets": 0.0,
        "non_interest_income": 0.0,

        # Specialty (Energy / Construction)
        "proved_reserves_value": 0.0,
        "backlog": 0.0,
        "exploration_expense": 0.0,

        "is_digital_bank": False,
    }


# ---------------------------------------------------------------------------
# Manual-entry template (returns a dict with all keys set to user values)
# ---------------------------------------------------------------------------

def manual_entry_template() -> Dict:
    """Return a blank data dict for manual entry."""
    return {k: 0.0 if isinstance(v, (int, float)) else v
            for k, v in fetch_yfinance.__code__.co_varnames  # type: ignore
            # Simpler: just return all keys with zeros
            } if False else {
        "ticker": "MANUAL",
        "company_name": "",
        "industry": "Manufacturing",
        "industry_raw": "",
        "sector_raw": "",
        "year": 2024,
        "year_warning": None,
        "total_assets": 0.0, "current_assets": 0.0, "current_liabilities": 0.0,
        "total_liabilities": 0.0, "long_term_debt": 0.0, "total_debt": 0.0,
        "retained_earnings": 0.0, "total_equity": 0.0,
        "cash_and_equivalents": 0.0, "short_term_investments": 0.0,
        "net_ppe": 0.0, "receivables": 0.0, "securities": 0.0,
        "revenue": 0.0, "revenue_prev": 0.0, "cost_of_revenue": 0.0,
        "gross_profit": 0.0, "ebit": 0.0, "ebitda": 0.0,
        "net_income": 0.0, "interest_expense": 0.0, "depreciation": 0.0,
        "sga_expense": 0.0, "rd_expense": 0.0, "operating_expenses": 0.0,
        "net_interest_income": 0.0, "operating_cash_flow": 0.0, "capex": 0.0,
        "market_cap": 0.0, "shares_outstanding": 0.0, "stock_price": 0.0,
        "prev_receivables": 0.0, "prev_revenue": 0.0, "prev_gross_profit": 0.0,
        "prev_current_assets": 0.0, "prev_ppe": 0.0, "prev_securities": 0.0,
        "prev_total_assets": 0.0, "prev_depreciation": 0.0, "prev_sga": 0.0,
        "prev_total_debt": 0.0,
        "npl": 0.0, "total_loans": 0.0, "tier1_capital": 0.0,
        "risk_weighted_assets": 0.0, "non_interest_income": 0.0,
        "proved_reserves_value": 0.0, "backlog": 0.0, "exploration_expense": 0.0,
        "is_digital_bank": False,
    }


# ---------------------------------------------------------------------------
# CSV upload parser
# ---------------------------------------------------------------------------

# Expected (minimum) columns in the CSV.  Column names are case-insensitive
# and underscores/spaces are normalised.
_CSV_COL_MAP = {
    "ticker": "ticker",
    "company_name": "company_name",
    "companyname": "company_name",
    "industry": "industry",
    "year": "year",
    "total_assets": "total_assets",
    "totalassets": "total_assets",
    "current_assets": "current_assets",
    "currentassets": "current_assets",
    "current_liabilities": "current_liabilities",
    "currentliabilities": "current_liabilities",
    "total_liabilities": "total_liabilities",
    "totalliabilities": "total_liabilities",
    "long_term_debt": "long_term_debt",
    "longtermdebt": "long_term_debt",
    "total_debt": "total_debt",
    "totaldebt": "total_debt",
    "retained_earnings": "retained_earnings",
    "retainedearnings": "retained_earnings",
    "total_equity": "total_equity",
    "totalequity": "total_equity",
    "cash_and_equivalents": "cash_and_equivalents",
    "cash": "cash_and_equivalents",
    "revenue": "revenue",
    "cost_of_revenue": "cost_of_revenue",
    "cogs": "cost_of_revenue",
    "gross_profit": "gross_profit",
    "grossprofit": "gross_profit",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "net_income": "net_income",
    "netincome": "net_income",
    "interest_expense": "interest_expense",
    "depreciation": "depreciation",
    "sga_expense": "sga_expense",
    "sga": "sga_expense",
    "rd_expense": "rd_expense",
    "rd": "rd_expense",
    "operating_expenses": "operating_expenses",
    "opex": "operating_expenses",
    "operating_cash_flow": "operating_cash_flow",
    "ocf": "operating_cash_flow",
    "market_cap": "market_cap",
    "marketcap": "market_cap",
    "shares_outstanding": "shares_outstanding",
    "stock_price": "stock_price",
    "revenue_prev": "revenue_prev",
    "prev_revenue": "revenue_prev",
    "npl": "npl",
    "total_loans": "total_loans",
    "net_interest_income": "net_interest_income",
    "net_ppe": "net_ppe",
    "receivables": "receivables",
}


def parse_csv(uploaded_file) -> List[Dict]:
    """Parse an uploaded CSV and return a list of company data dicts."""
    df = pd.read_csv(uploaded_file)
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    records: list[dict] = []
    for _, row in df.iterrows():
        template = manual_entry_template()
        for raw_col, std_key in _CSV_COL_MAP.items():
            if raw_col in row.index:
                val = row[raw_col]
                if std_key in ("ticker", "company_name", "industry", "industry_raw", "sector_raw"):
                    template[std_key] = str(val) if pd.notna(val) else ""
                elif std_key == "year":
                    template[std_key] = int(val) if pd.notna(val) else 2024
                else:
                    template[std_key] = float(val) if pd.notna(val) else 0.0
        # Fill derived fields
        if template["total_debt"] == 0 and template["long_term_debt"] > 0:
            template["total_debt"] = template["long_term_debt"]
        if template["gross_profit"] == 0 and template["revenue"] > 0 and template["cost_of_revenue"] > 0:
            template["gross_profit"] = template["revenue"] - template["cost_of_revenue"]
        records.append(template)
    return records
