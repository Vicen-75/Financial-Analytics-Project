# Fixed XGBoost zone classification with proper probability thresholds as requested
"""
app.py — M&A Risk & Synergy Analyzer
Main Streamlit application.

Run with:  streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

from data_fetcher import fetch_yfinance, manual_entry_template, parse_csv, _map_sector
from models import (run_all_models, run_isds, beneish_mscore,
                    logistic_regression, run_xgboost_zscore,
                    MODEL_PERFORMANCE_STATS, synergy_scorecard,
                    INDUSTRY_CHOICES, INDUSTRY_MODEL_MAP,
                    compute_all_readability)
from valuation import estimate_beta, cost_of_equity, intrinsic_value, estimate_growth
from sentiment import (fetch_10k_text, compute_lm_features, compute_distilbert_embedding,
                       predict_sentiment, LM_CATEGORY_NAMES)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="M&A Risk & Synergy Analyzer",
    page_icon="\U0001F4CA",  # bar chart emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for colour-coded cards
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.score-card {
    padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
    border-left: 6px solid; color: #fff;
}
.score-card.green  { background: #14532d; border-color: #22c55e; }
.score-card.orange { background: #713f12; border-color: #f59e0b; }
.score-card.red    { background: #7f1d1d; border-color: #ef4444; }
.score-card h3 { margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.score-card .big   { font-size: 2rem; font-weight: 700; }
.score-card .zone  { font-size: 1rem; font-weight: 600; opacity: 0.9; }
.score-card .interp { font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem; }
.var-table th { text-align: left; padding: 4px 8px; }
.var-table td { padding: 4px 8px; }
.synergy-high { color: #22c55e; font-weight: 700; }
.synergy-low  { color: #f59e0b; font-weight: 700; }
.synergy-no   { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: render a model result as a colour-coded card
# ---------------------------------------------------------------------------

def render_score_card(result: dict):
    """Render a single model result as a styled card with expandable details."""
    c = result["color"]
    score_display = result["score"]
    # For Logistic Regression, show as percentage
    if "Logistic Regression" in result["model_name"]:
        score_display = f"{result['score']:.2%}"
    else:
        score_display = f"{result['score']:.4f}"

    html = f"""
    <div class="score-card {c}">
        <h3>{result['model_name']}</h3>
        <div class="big">{score_display}</div>
        <div class="zone">{result['zone']}  &mdash;  {result['direction']}</div>
        <div class="interp">{result['interpretation']}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Warnings
    for w in result.get("warnings", []):
        st.warning(w)

    # Variable breakdown
    with st.expander(f"Variable Breakdown  —  {result['model_name']}", expanded=False):
        # Thresholds
        thr = result.get("thresholds", {})
        if thr:
            st.markdown("**Thresholds:** " + " | ".join(f"{k}: {v}" for k, v in thr.items()))
        # Table — rendered as HTML so the Note column wraps fully and is never truncated
        vars_ = result.get("variables", [])
        if vars_:
            df = pd.DataFrame(vars_)
            df.columns = ["Variable", "Value", "Contribution", "Note"]
            _hdr = (
                "<tr>"
                "<th style='padding:6px 10px;border-bottom:2px solid #475569;color:#94a3b8;"
                "font-size:0.78rem;text-transform:uppercase'>Variable</th>"
                "<th style='padding:6px 10px;border-bottom:2px solid #475569;color:#94a3b8;"
                "font-size:0.78rem;text-transform:uppercase'>Value</th>"
                "<th style='padding:6px 10px;border-bottom:2px solid #475569;color:#94a3b8;"
                "font-size:0.78rem;text-transform:uppercase'>Contribution</th>"
                "<th style='padding:6px 10px;border-bottom:2px solid #475569;color:#94a3b8;"
                "font-size:0.78rem;text-transform:uppercase'>Note</th>"
                "</tr>"
            )
            _rows = "".join(
                f"<tr>"
                f"<td style='font-weight:600;white-space:nowrap;padding:6px 10px;"
                f"border-bottom:1px solid #334155;vertical-align:top'>{row['Variable']}</td>"
                f"<td style='font-family:monospace;white-space:nowrap;padding:6px 10px;"
                f"border-bottom:1px solid #334155;vertical-align:top'>{float(row['Value']):.6f}</td>"
                f"<td style='font-family:monospace;white-space:nowrap;padding:6px 10px;"
                f"border-bottom:1px solid #334155;vertical-align:top'>{float(row['Contribution']):.4f}</td>"
                f"<td style='padding:6px 10px;border-bottom:1px solid #334155;"
                f"vertical-align:top;line-height:1.55;font-size:0.84rem'>{row['Note']}</td>"
                f"</tr>"
                for _, row in df.iterrows()
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
                f"<thead>{_hdr}</thead><tbody>{_rows}</tbody></table>",
                unsafe_allow_html=True,
            )

        # Contribution bar chart
        if vars_:
            fig = go.Figure()
            names = [v["name"][:30] for v in vars_]
            contribs = [v["contribution"] for v in vars_]
            colors = ["#22c55e" if c >= 0 else "#ef4444" for c in contribs]
            fig.add_trace(go.Bar(x=contribs, y=names, orientation="h",
                                 marker_color=colors, text=[f"{c:.3f}" for c in contribs],
                                 textposition="outside"))
            fig.update_layout(title="Variable Contributions", height=max(250, len(vars_)*45),
                              margin=dict(l=0, r=40, t=40, b=0),
                              xaxis_title="Contribution to Score",
                              yaxis=dict(autorange="reversed"),
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Helper: render gauge chart for overall risk
# ---------------------------------------------------------------------------

# === FIXED: Added unique keys to plotly_chart to fix StreamlitDuplicateElementId in merger analysis ===
def render_risk_gauge(results: list[dict], key: str = "risk_gauge"):
    """Create a composite risk gauge from all model results."""
    risk_scores = []
    for r in results:
        zone = r["zone"].lower()
        if "safe" in zone or "healthy" in zone or "unlikely" in zone or "low" in zone:
            risk_scores.append(1)
        elif "grey" in zone or "monitor" in zone or "moderate" in zone:
            risk_scores.append(2)
        else:
            risk_scores.append(3)
    avg = sum(risk_scores) / len(risk_scores) if risk_scores else 2

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg,
        title={"text": "Composite Risk Level", "font": {"size": 18}},
        gauge={
            "axis": {"range": [1, 3], "tickvals": [1, 1.5, 2, 2.5, 3],
                     "ticktext": ["Safe", "", "Grey", "", "Distress"]},
            "bar": {"color": "#60a5fa"},
            "steps": [
                {"range": [1, 1.67], "color": "#14532d"},
                {"range": [1.67, 2.33], "color": "#713f12"},
                {"range": [2.33, 3], "color": "#7f1d1d"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": avg},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, key=key)


# ---------------------------------------------------------------------------
# Data input panel (reusable for single & merger modes)
# ---------------------------------------------------------------------------

def data_input_panel(key_prefix: str = "") -> dict | list[dict] | None:
    """Render data-input controls and return the financial data dict(s)."""
    method = st.radio("Input Method", ["Auto-fetch (yfinance)", "Manual Entry", "CSV Upload"],
                      horizontal=True, key=f"{key_prefix}_method")

    if method == "Auto-fetch (yfinance)":
        c1, c2 = st.columns([2, 1])
        with c1:
            ticker = st.text_input("Ticker Symbol", value="AAPL",
                                   key=f"{key_prefix}_ticker").strip().upper()
        with c2:
            year = st.selectbox("Fiscal Year", list(range(2025, 2019, -1)),
                                key=f"{key_prefix}_year")

        if st.button("Fetch Data", key=f"{key_prefix}_fetch", type="primary"):
            with st.spinner(f"Fetching {ticker} FY{year}..."):
                try:
                    data = fetch_yfinance(ticker, year)
                    if data.get("year_warning"):
                        st.warning(data["year_warning"])
                    st.session_state[f"{key_prefix}_data"] = data
                    st.success(f"Loaded {data['company_name']} ({data['ticker']}) — FY{data['year']}")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    return None

        data = st.session_state.get(f"{key_prefix}_data")
        if data:
            # Let user override industry
            current_idx = INDUSTRY_CHOICES.index(data["industry"]) if data["industry"] in INDUSTRY_CHOICES else 0
            industry = st.selectbox("Industry (auto-detected — override if needed)",
                                    INDUSTRY_CHOICES, index=current_idx,
                                    key=f"{key_prefix}_industry")
            data["industry"] = industry

            # Energy-specific overrides
            if industry == "Energy":
                with st.expander("Energy-Specific Inputs (optional)", expanded=False):
                    data["proved_reserves_value"] = st.number_input("Proved Reserves Value ($)",
                                                                     value=0.0, key=f"{key_prefix}_reserves")
                    data["exploration_expense"] = st.number_input("Exploration Expense ($)",
                                                                   value=0.0, key=f"{key_prefix}_expl")

            if industry == "Construction":
                with st.expander("Construction-Specific Inputs", expanded=False):
                    data["backlog"] = st.number_input("Backlog ($)", value=0.0,
                                                      key=f"{key_prefix}_backlog")

            # Show key data summary
            with st.expander("Financial Data Preview", expanded=False):
                preview = {
                    "Total Assets": f"${data['total_assets']:,.0f}",
                    "Total Liabilities": f"${data['total_liabilities']:,.0f}",
                    "Total Equity": f"${data['total_equity']:,.0f}",
                    "Revenue": f"${data['revenue']:,.0f}",
                    "Net Income": f"${data['net_income']:,.0f}",
                    "EBIT": f"${data['ebit']:,.0f}",
                    "Operating CF": f"${data['operating_cash_flow']:,.0f}",
                    "Market Cap": f"${data['market_cap']:,.0f}",
                    "Sector": data.get("sector_raw", ""),
                }
                st.json(preview)

            return data
        return None

    elif method == "Manual Entry":
        st.info("Enter financial data manually (all values in dollars).")
        data = manual_entry_template()
        c1, c2, c3 = st.columns(3)
        with c1:
            data["company_name"] = st.text_input("Company Name", key=f"{key_prefix}_m_name")
            data["industry"] = st.selectbox("Industry", INDUSTRY_CHOICES, key=f"{key_prefix}_m_ind")
            data["year"] = st.number_input("Fiscal Year", value=2024, min_value=2000, max_value=2026,
                                           key=f"{key_prefix}_m_year")
        with c2:
            data["total_assets"] = st.number_input("Total Assets", value=0.0, key=f"{key_prefix}_m_ta")
            data["current_assets"] = st.number_input("Current Assets", value=0.0, key=f"{key_prefix}_m_ca")
            data["current_liabilities"] = st.number_input("Current Liabilities", value=0.0,
                                                           key=f"{key_prefix}_m_cl")
            data["total_liabilities"] = st.number_input("Total Liabilities", value=0.0,
                                                         key=f"{key_prefix}_m_tl")
            data["total_equity"] = st.number_input("Total Equity", value=0.0, key=f"{key_prefix}_m_eq")
            data["retained_earnings"] = st.number_input("Retained Earnings", value=0.0,
                                                         key=f"{key_prefix}_m_re")
            data["long_term_debt"] = st.number_input("Long-Term Debt", value=0.0, key=f"{key_prefix}_m_ltd")
            data["total_debt"] = st.number_input("Total Debt", value=0.0, key=f"{key_prefix}_m_td")
        with c3:
            data["revenue"] = st.number_input("Revenue", value=0.0, key=f"{key_prefix}_m_rev")
            data["revenue_prev"] = st.number_input("Prior-Year Revenue", value=0.0,
                                                    key=f"{key_prefix}_m_prev_rev")
            data["ebit"] = st.number_input("EBIT", value=0.0, key=f"{key_prefix}_m_ebit")
            data["net_income"] = st.number_input("Net Income", value=0.0, key=f"{key_prefix}_m_ni")
            data["operating_cash_flow"] = st.number_input("Operating Cash Flow", value=0.0,
                                                           key=f"{key_prefix}_m_ocf")
            data["cash_and_equivalents"] = st.number_input("Cash & Equivalents", value=0.0,
                                                            key=f"{key_prefix}_m_cash")
            data["market_cap"] = st.number_input("Market Cap", value=0.0, key=f"{key_prefix}_m_mc")
            data["cost_of_revenue"] = st.number_input("Cost of Revenue (COGS)", value=0.0,
                                                       key=f"{key_prefix}_m_cogs")

        with st.expander("Additional Fields (Beneish / Specialty)", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                data["gross_profit"] = st.number_input("Gross Profit", value=0.0, key=f"{key_prefix}_m_gp")
                data["depreciation"] = st.number_input("Depreciation", value=0.0, key=f"{key_prefix}_m_dep")
                data["sga_expense"] = st.number_input("SGA Expense", value=0.0, key=f"{key_prefix}_m_sga")
                data["rd_expense"] = st.number_input("R&D Expense", value=0.0, key=f"{key_prefix}_m_rd")
                data["operating_expenses"] = st.number_input("Total OpEx", value=0.0,
                                                              key=f"{key_prefix}_m_opex")
                data["interest_expense"] = st.number_input("Interest Expense", value=0.0,
                                                            key=f"{key_prefix}_m_ie")
            with ac2:
                data["receivables"] = st.number_input("Receivables", value=0.0, key=f"{key_prefix}_m_recv")
                data["net_ppe"] = st.number_input("Net PP&E", value=0.0, key=f"{key_prefix}_m_ppe")
                data["shares_outstanding"] = st.number_input("Shares Outstanding", value=0.0,
                                                              key=f"{key_prefix}_m_shares")
                data["stock_price"] = st.number_input("Stock Price", value=0.0, key=f"{key_prefix}_m_px")
                data["prev_total_assets"] = st.number_input("Prior-Year Total Assets", value=0.0,
                                                             key=f"{key_prefix}_m_prev_ta")
                data["ebitda"] = st.number_input("EBITDA", value=0.0, key=f"{key_prefix}_m_ebitda")

        # Derive missing fields
        if data["gross_profit"] == 0 and data["revenue"] > 0 and data["cost_of_revenue"] > 0:
            data["gross_profit"] = data["revenue"] - data["cost_of_revenue"]
        if data["total_debt"] == 0 and data["long_term_debt"] > 0:
            data["total_debt"] = data["long_term_debt"]

        if data["total_assets"] > 0:
            return data
        else:
            st.caption("Enter at least Total Assets to run analysis.")
            return None

    else:  # CSV Upload
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{key_prefix}_csv")
        if uploaded:
            records = parse_csv(uploaded)
            st.success(f"Parsed {len(records)} company record(s) from CSV.")
            for i, rec in enumerate(records):
                rec["industry"] = st.selectbox(
                    f"Industry for {rec.get('company_name') or rec.get('ticker', f'Row {i}')}",
                    INDUSTRY_CHOICES,
                    index=INDUSTRY_CHOICES.index(rec.get("industry", "Manufacturing"))
                          if rec.get("industry", "Manufacturing") in INDUSTRY_CHOICES else 3,
                    key=f"{key_prefix}_csv_ind_{i}")
            return records
        return None


# ---------------------------------------------------------------------------
# Valuation section
# ---------------------------------------------------------------------------

def _fmt_large(v: float) -> str:
    """Format a large monetary value compactly to avoid metric truncation."""
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:.3f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:.3f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.3f}M"
    if abs_v >= 1e3:
        return f"{sign}${abs_v / 1e3:.2f}K"
    return f"{sign}${abs_v:,.2f}"


def render_valuation(data: dict, key_prefix: str = "single"):
    """Show beta, CAPM, and intrinsic value estimates."""
    st.subheader("Valuation & Cost of Capital")

    ticker = data.get("ticker", "")
    if not ticker or ticker == "MANUAL":
        st.info("Valuation requires a public ticker for beta estimation.")
        return

    # Auto-estimate growth from historical financials
    est_growth = estimate_growth(data)

    c1, c2, c3 = st.columns(3)
    with c1:
        rf = st.number_input("Risk-Free Rate (Rf)", value=0.043, format="%.4f", key=f"{key_prefix}_val_rf")
    with c2:
        mrp = st.number_input("Market Risk Premium", value=0.055, format="%.4f", key=f"{key_prefix}_val_mrp")
    with c3:
        growth = st.number_input("Expected Growth Rate", value=float(est_growth),
                                 format="%.3f", key=f"{key_prefix}_val_g",
                                 help=f"Auto-estimated: {est_growth:.1%} (from historical financials). Adjust as needed.")

    with st.spinner("Estimating beta (5y weekly data)..."):
        beta_result = estimate_beta(ticker)

    if beta_result.get("error"):
        st.warning(f"Beta estimation warning: {beta_result['error']}")

    beta = beta_result["beta"]
    ke = cost_of_equity(beta, rf, mrp)
    n_obs = beta_result.get("n_observations", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity Beta", f"{beta:.3f}")
    col2.metric("R-Squared", f"{beta_result['r_squared']:.3f}",
                help=f"Based on {n_obs} weekly return observations")
    col3.metric("Cost of Equity (Ke)", f"{ke:.2%}")
    col4.metric("Alpha (weekly)", f"{beta_result['alpha']:.4f}")

    # Scatter plot of returns
    if beta_result.get("stock_returns") and len(beta_result["stock_returns"]) > 5:
        fig = px.scatter(x=beta_result["bench_returns"], y=beta_result["stock_returns"],
                         labels={"x": "SPY Returns", "y": f"{ticker} Returns"},
                         title=f"Beta Regression: {ticker} vs SPY ({n_obs} weekly obs)",
                         template="plotly_dark", opacity=0.6)
        # Add regression line
        import numpy as np
        x_arr = np.array(beta_result["bench_returns"])
        y_line = beta_result["alpha"] + beta * x_arr
        fig.add_trace(go.Scatter(x=x_arr.tolist(), y=y_line.tolist(), mode="lines",
                                 name=f"Beta={beta:.2f}", line=dict(color="#ef4444", width=2)))
        fig.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Intrinsic value (10-year projection with fading growth)
    iv = intrinsic_value(data, beta, rf, mrp, growth)

    st.markdown("---")
    st.markdown("**Intrinsic Value Estimates**")
    upside_note = (f"{iv['upside_pct']:+.1f}% vs blended fair value"
                   if iv["upside_pct"] is not None else "—")
    iv_table = pd.DataFrame({
        "Metric": [
            "DCF Fair Value",
            "Graham Fair Value",
            "Blended Fair Value",
            "Current Price",
        ],
        "Value": [
            _fmt_large(iv["dcf_fair_value"]),
            _fmt_large(iv["graham_fair_value"]),
            _fmt_large(iv["blended_fair_value"]),
            _fmt_large(iv["current_price"]),
        ],
        "Notes": [
            "10-yr two-stage DCF, growth fading to terminal rate",
            "Graham: EPS × (8.5 + 2g) × 4.4 / Y",
            "Blended fair value weighted across DCF and Graham methods",
            upside_note,
        ],
    })
    st.dataframe(iv_table, use_container_width=True, hide_index=True,
                 column_config={
                     "Metric": st.column_config.TextColumn("Metric", width="medium"),
                     "Value":  st.column_config.TextColumn("Value",  width="medium"),
                     "Notes":  st.column_config.TextColumn("Notes",  width="large"),
                 })

    # Projected FCF table
    if iv.get("projected_fcf"):
        with st.expander("Projected Free Cash Flow (10-Year DCF)", expanded=False):
            st.dataframe(pd.DataFrame(iv["projected_fcf"]), use_container_width=True, hide_index=True)


# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("M&A Risk Analyzer")
        st.caption("Distress Scores | Beneish | ISDS-XGBooster")
        st.markdown("---")
        mode = st.radio("Analysis Mode",
                        ["Single Target Assessment", "Merger Analysis",
                         "Textual Sentiment Analysis"],
                        key="mode")
        st.markdown("---")
        st.markdown("""
        **Models Included:**
        - 7 Industry Distress Score Models (analytical)
        - Beneish M-Score (Manipulation)
        - ISDS-XGBooster Distress Score (7 industry models)
        - CAPM Valuation
        - Textual Sentiment (L&M + DistilBERT)
        """)
        st.markdown("---")
        st.caption("Built for research & analytical purposes only. Not investment advice.")

    # --- Header ---
    st.markdown("# M&A Risk & Synergy Analyzer")
    st.markdown("*Industry-calibrated distress scoring across 7 sectors with 100+ year backtesting*")
    st.markdown("---")

    # ================================================================
    # SINGLE TARGET MODE
    # ================================================================
    if mode == "Single Target Assessment":
        data = data_input_panel(key_prefix="single")

        if data is None:
            st.info("Configure and fetch data using the controls above to begin analysis.")
            return

        # Handle list (CSV) or single dict
        datasets = data if isinstance(data, list) else [data]

        for i, d in enumerate(datasets):
            if len(datasets) > 1:
                st.markdown(f"## Company {i+1}: {d.get('company_name') or d.get('ticker', 'Unknown')}")
                st.markdown("---")

            name = d.get("company_name") or d.get("ticker", "Company")
            st.markdown(f"## Analysis Results — {name}")
            st.markdown(f"**Industry:** {d.get('industry', 'N/A')} | "
                        f"**Fiscal Year:** {d.get('year', 'N/A')} | "
                        f"**Sector:** {d.get('sector_raw', 'N/A')}")

            # Run all models
            results = run_all_models(d)

            # Risk gauge
            render_risk_gauge(results)

            # Summary table
            st.markdown("### Score Summary")
            summary_rows = []
            for r in results:
                # Probability-based models (XGBoost, etc.) store direction starting with "Probability"
                is_prob = r.get("direction", "").startswith("Probability")
                score_str = f"{r['score']:.2%}" if is_prob else f"{r['score']:.4f}"
                summary_rows.append({
                    "Model": r["model_name"],
                    "Score": score_str,
                    "Zone / Flag": r["zone"],
                    "Direction": r["direction"],
                })
            summary_df = pd.DataFrame(summary_rows)

            def _color_zone(val):
                v = val.lower()
                if "safe" in v or "healthy" in v or "unlikely" in v or "low" in v:
                    return "background-color: #14532d; color: white"
                elif "grey" in v or "monitor" in v or "moderate" in v:
                    return "background-color: #713f12; color: white"
                return "background-color: #7f1d1d; color: white"

            styled = summary_df.style.map(_color_zone, subset=["Zone / Flag"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Individual model cards
            st.markdown("### Detailed Model Results")
            for r in results:
                render_score_card(r)

            # Valuation
            st.markdown("---")
            if d.get("ticker") and d["ticker"] != "MANUAL":
                render_valuation(d)

            # ------------------------------------------------------------------
            # Model Performance Summary
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Model Performance Summary")
            st.markdown(
                "The table below shows **out-of-sample validation statistics** for the "
                f"XGBoost Altman Z-Score model trained specifically on **{d.get('industry', 'the selected')} "
                "sector** companies. These metrics were recorded on a held-out test set during "
                "model training and indicate how reliable the XGBoost prediction above is likely to be."
            )

            industry_key = d.get("industry", "")
            perf = MODEL_PERFORMANCE_STATS.get(industry_key)

            if perf:
                # Render as a plain HTML table so text wraps freely with the page
                # and users never need to scroll inside a cell to read interpretations.
                _perf_rows = [
                    ("Accuracy",
                     f"{perf['accuracy']:.1%}",
                     f"Of every 100 {industry_key} companies in the test set, the model "
                     f"correctly classified {perf['accuracy']*100:.0f} as either distressed or healthy."),
                    ("Precision",
                     f"{perf['precision']:.1%}",
                     f"When the model flags a company as distressed it is correct "
                     f"{perf['precision']*100:.0f}% of the time — this limits costly false alarms."),
                    ("Recall",
                     f"{perf['recall']:.1%}",
                     f"The model catches {perf['recall']*100:.0f}% of all genuinely distressed "
                     f"companies — very few real distress cases are missed."),
                    ("F1-Score",
                     f"{perf['f1']:.1%}",
                     f"Harmonic mean of precision and recall ({perf['f1']*100:.0f}%). "
                     f"Particularly meaningful because financial distress is a rare event "
                     f"(imbalanced classes)."),
                    ("ROC-AUC",
                     f"{perf['roc_auc']:.1%}",
                     f"There is a {perf['roc_auc']*100:.0f}% chance the model ranks a distressed "
                     f"company higher-risk than a healthy one — strong discriminatory power."),
                ]
                _rows_html = "".join(
                    f"<tr>"
                    f"<td style='font-weight:700;white-space:nowrap;padding:9px 14px;"
                    f"border-bottom:1px solid #334155;vertical-align:top'>{m}</td>"
                    f"<td style='font-weight:700;color:#60a5fa;white-space:nowrap;padding:9px 14px;"
                    f"border-bottom:1px solid #334155;vertical-align:top'>{v}</td>"
                    f"<td style='padding:9px 14px;border-bottom:1px solid #334155;"
                    f"vertical-align:top;line-height:1.5'>{n}</td>"
                    f"</tr>"
                    for m, v, n in _perf_rows
                )
                _header_html = (
                    "<tr>"
                    "<th style='padding:9px 14px;border-bottom:2px solid #475569;"
                    "color:#94a3b8;font-size:0.82rem;text-transform:uppercase;"
                    "letter-spacing:0.05em'>Metric</th>"
                    "<th style='padding:9px 14px;border-bottom:2px solid #475569;"
                    "color:#94a3b8;font-size:0.82rem;text-transform:uppercase;"
                    "letter-spacing:0.05em'>Value</th>"
                    "<th style='padding:9px 14px;border-bottom:2px solid #475569;"
                    "color:#94a3b8;font-size:0.82rem;text-transform:uppercase;"
                    "letter-spacing:0.05em'>Plain-English Interpretation</th>"
                    "</tr>"
                )
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.92rem'>"
                    f"<thead>{_header_html}</thead>"
                    f"<tbody>{_rows_html}</tbody>"
                    f"</table>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"No XGBoost performance statistics are available for the "
                    f"'{industry_key}' industry. Select a recognised industry to see model metrics."
                )

            if len(datasets) > 1:
                st.markdown("---")
                st.markdown("---")

    # ================================================================
    # MERGER ANALYSIS MODE
    # ================================================================
elif mode == "Merger Analysis":
        st.markdown("## Merger / Acquisition Analysis")
        st.markdown("Input data for both the **Acquirer** and the **Target** to generate "
                    "individual risk scores and a combined synergy scorecard.")

        tab_a, tab_t = st.tabs(["Acquirer", "Target"])

        with tab_a:
            st.markdown("### Acquirer Company")
            data_a = data_input_panel(key_prefix="acq")
            if data_a is not None:
                if isinstance(data_a, list):
                    data_a = data_a[0]
                st.session_state["merger_data_a"] = data_a

        with tab_t:
            st.markdown("### Target Company")
            data_t = data_input_panel(key_prefix="tgt")
            if data_t is not None:
                if isinstance(data_t, list):
                    data_t = data_t[0]
                st.session_state["merger_data_t"] = data_t

        # Recuperar de session_state si el usuario cambió de pestaña
        data_a = st.session_state.get("merger_data_a")
        data_t = st.session_state.get("merger_data_t")

        # Indicadores de estado
        col_status_a, col_status_t = st.columns(2)
        with col_status_a:
            if data_a:
                st.success(f"✅ Acquirer loaded: {data_a.get('company_name') or data_a.get('ticker', 'Unknown')}")
            else:
                st.warning("⏳ Acquirer: no data yet — fetch or enter data in the Acquirer tab.")
        with col_status_t:
            if data_t:
                st.success(f"✅ Target loaded: {data_t.get('company_name') or data_t.get('ticker', 'Unknown')}")
            else:
                st.warning("⏳ Target: no data yet — fetch or enter data in the Target tab.")

        if data_a and data_t:
            if st.button("Run Merger Analysis", type="primary", key="run_merger"):
                st.markdown("---")
                st.markdown("## Merger Analysis Results")

                # Compute synergy scorecard
                with st.spinner("Running all models and computing synergies..."):
                    sc = synergy_scorecard(data_a, data_t)

                # --- Overall Synergy Verdict ---
                overall_c = sc["overall_color"]
                st.markdown(f"""
                <div class="score-card {overall_c}" style="text-align:center;">
                    <h3>Overall Synergy Assessment</h3>
                    <div class="big">{sc['overall']}</div>
                    <div class="interp">{sc['overall_text']}</div>
                </div>
                """, unsafe_allow_html=True)

                # --- Synergy Details ---
                st.markdown("### Synergy Dimensions")
                syn_rows = []
                for dim, level, explanation in sc["synergies"]:
                    cls = {"High": "synergy-high", "Low": "synergy-low", "No": "synergy-no"}.get(level, "")
                    syn_rows.append({
                        "Dimension": dim,
                        "Synergy Level": level,
                        "Explanation": explanation,
                    })
                syn_df = pd.DataFrame(syn_rows)

                def _color_syn(val):
                    if val == "High":
                        return "background-color: #14532d; color: white"
                    elif val == "Low":
                        return "background-color: #713f12; color: white"
                    return "background-color: #7f1d1d; color: white"

                styled_syn = syn_df.style.map(_color_syn, subset=["Synergy Level"])
                st.dataframe(styled_syn, use_container_width=True, hide_index=True)

                # --- Side-by-side scores ---
                st.markdown("### Individual Company Scores")
                col_a, col_t = st.columns(2)

                with col_a:
                    name_a = data_a.get("company_name") or data_a.get("ticker", "Acquirer")
                    st.markdown(f"#### {name_a} (Acquirer)")
                    render_risk_gauge(sc["acquirer_scores"], key="acquirer_risk_gauge")
                    for r in sc["acquirer_scores"]:
                        render_score_card(r)

                with col_t:
                    name_t = data_t.get("company_name") or data_t.get("ticker", "Target")
                    st.markdown(f"#### {name_t} (Target)")
                    render_risk_gauge(sc["target_scores"], key="target_risk_gauge")
                    for r in sc["target_scores"]:
                        render_score_card(r)

                # --- XGBoost Model Performance Summary (both companies) ---
                st.markdown("### XGBoost Model Performance Summary")
                st.markdown(
                    "Validation statistics for the XGBoost Altman Z-Score model used "
                    "for each company. Metrics were recorded on a held-out test set "
                    "during model training."
                )
                _perf_cols = st.columns(2)
                for _col, _label, _data_d in [
                    (_perf_cols[0], name_a + " (Acquirer)", data_a),
                    (_perf_cols[1], name_t + " (Target)",   data_t),
                ]:
                    with _col:
                        st.markdown(f"**{_label}**")
                        _ind = _data_d.get("industry", "")
                        _perf = MODEL_PERFORMANCE_STATS.get(_ind)
                        if _perf:
                            _p_rows = [
                                ("Accuracy",  f"{_perf['accuracy']:.1%}",
                                 f"Correctly classified {_perf['accuracy']*100:.0f}% of test companies."),
                                ("Precision", f"{_perf['precision']:.1%}",
                                 f"Correct {_perf['precision']*100:.0f}% of the time when flagging distress."),
                                ("Recall",    f"{_perf['recall']:.1%}",
                                 f"Catches {_perf['recall']*100:.0f}% of genuinely distressed companies."),
                                ("F1-Score",  f"{_perf['f1']:.1%}",
                                 f"Balanced precision/recall score: {_perf['f1']*100:.0f}%."),
                                ("ROC-AUC",   f"{_perf['roc_auc']:.1%}",
                                 f"{_perf['roc_auc']*100:.0f}% discriminatory power."),
                            ]
                            _r_html = "".join(
                                f"<tr>"
                                f"<td style='font-weight:700;white-space:nowrap;padding:7px 10px;"
                                f"border-bottom:1px solid #334155;vertical-align:top'>{_m}</td>"
                                f"<td style='font-weight:700;color:#60a5fa;white-space:nowrap;"
                                f"padding:7px 10px;border-bottom:1px solid #334155;"
                                f"vertical-align:top'>{_v}</td>"
                                f"<td style='padding:7px 10px;border-bottom:1px solid #334155;"
                                f"vertical-align:top;font-size:0.88rem;line-height:1.4'>{_n}</td>"
                                f"</tr>"
                                for _m, _v, _n in _p_rows
                            )
                            st.markdown(
                                f"<table style='width:100%;border-collapse:collapse;"
                                f"font-size:0.9rem'><tbody>{_r_html}</tbody></table>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info(f"No performance stats available for '{_ind}'.")

                st.markdown("---")

                # Comparative radar chart
                st.markdown("### Comparative Risk Profile")
                categories = [r["model_name"].split("(")[0].strip()
                              for r in sc["acquirer_scores"]]
                # Normalize scores to 0-1 range for radar (higher = safer for all models).
                def _norm_score(r):
                    s = r["score"]
                    direction = r.get("direction", "")
                    # Probability models (XGBoost, Logistic Regression): score is 0-1
                    # where 0 = perfectly healthy.  Invert so radar "higher = safer".
                    if direction.startswith("Probability"):
                        return max(0.0, min(1.0 - s, 1.0))
                    # INVERTED analytical models (ISDS-FIN): lower raw score = safer
                    if "lower" in direction.lower() or "inverted" in direction.lower():
                        # Negate then rescale to positive range assuming scores sit in [-5, 5]
                        return max(0.0, min((5.0 + (-s)) / 10.0, 1.0))
                    # Standard models: higher raw score = safer, typically 0–5 range
                    return max(0.0, min(s / 5.0, 1.0))

                vals_a = [_norm_score(r) for r in sc["acquirer_scores"]]
                vals_t = [_norm_score(r) for r in sc["target_scores"]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]],
                                                     theta=categories + [categories[0]],
                                                     fill="toself", name="Acquirer",
                                                     line_color="#60a5fa"))
                fig_radar.add_trace(go.Scatterpolar(r=vals_t + [vals_t[0]],
                                                     theta=categories + [categories[0]],
                                                     fill="toself", name="Target",
                                                     line_color="#f59e0b"))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                         template="plotly_dark", height=450,
                                         title="Normalised Risk Profile (higher = safer)")
                st.plotly_chart(fig_radar, use_container_width=True)

                # --- Valuation for each company ---
                st.markdown("### Valuation Estimates")
                vcol_a, vcol_t = st.columns(2)
                with vcol_a:
                    if data_a.get("ticker") and data_a["ticker"] != "MANUAL":
                        st.markdown(f"#### {name_a} Valuation")
                        render_valuation(data_a, key_prefix="acq")
                with vcol_t:
                    if data_t.get("ticker") and data_t["ticker"] != "MANUAL":
                        st.markdown(f"#### {name_t} Valuation")
                        render_valuation(data_t, key_prefix="tgt")

        else:
            st.info("Complete data entry for both Acquirer and Target, then click 'Run Merger Analysis'.")

    # ================================================================
    # TEXTUAL SENTIMENT ANALYSIS MODE
    # ================================================================
    elif mode == "Textual Sentiment Analysis":
        st.markdown("## Textual Sentiment Analysis")
        st.markdown("Analyse 10-K filing text using **Loughran & McDonald dictionary** features "
                    "and **DistilBERT** contextual embeddings, combined via **Logistic Regression**.")

        sent_method = st.radio("Input Method",
                               ["Auto-fetch 10-K (SEC EDGAR)", "Upload CSV"],
                               horizontal=True, key="sent_method")

        if sent_method == "Auto-fetch 10-K (SEC EDGAR)":
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                sent_ticker = st.text_input("Ticker Symbol", value="AAPL",
                                            key="sent_ticker").strip().upper()
            with sc2:
                sent_year = st.selectbox("Filing Year",
                                         list(range(2025, 2019, -1)),
                                         key="sent_year")

            if st.button("Fetch & Analyse", key="sent_fetch", type="primary"):
                with st.spinner(f"Downloading 10-K for {sent_ticker} ({sent_year})..."):
                    text, fetch_warn = fetch_10k_text(sent_ticker, sent_year)
                if fetch_warn:
                    st.warning(fetch_warn)
                if text and len(text.strip()) > 200:
                    st.session_state["sent_texts"] = [
                        {"ticker": sent_ticker, "year": sent_year, "text": text}
                    ]
                    st.success(f"Fetched {len(text):,} characters of 10-K text.")
                else:
                    st.error("Could not retrieve sufficient 10-K text. "
                             "Try a different ticker/year or upload a CSV instead.")

        else:  # CSV Upload
            st.markdown("**Expected CSV columns:** `ticker`, `year`, `text` (or `filing_text`)")
            sent_csv = st.file_uploader("Upload Sentiment CSV", type=["csv"],
                                        key="sent_csv_upload")
            if sent_csv is not None:
                try:
                    df_sent = pd.read_csv(sent_csv)
                    df_sent.columns = [c.strip().lower().replace(" ", "_") for c in df_sent.columns]
                    # Normalise the text column name
                    text_col = None
                    for candidate in ("text", "filing_text", "content", "body"):
                        if candidate in df_sent.columns:
                            text_col = candidate
                            break
                    if text_col is None:
                        st.error("CSV must contain a text column named 'text' or 'filing_text'.")
                    else:
                        records = []
                        for _, row in df_sent.iterrows():
                            records.append({
                                "ticker": str(row.get("ticker", "N/A")),
                                "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                                "text": str(row[text_col]) if pd.notna(row[text_col]) else "",
                            })
                        records = [r for r in records if len(r["text"].strip()) > 50]
                        if records:
                            st.session_state["sent_texts"] = records
                            st.success(f"Loaded {len(records)} text record(s) from CSV.")
                        else:
                            st.error("No valid text rows found in CSV.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        # ----- Run analysis if we have texts -----
        sent_texts = st.session_state.get("sent_texts")
        if sent_texts:
            st.markdown("---")
            st.markdown("### Sentiment Results")

            for i, rec in enumerate(sent_texts):
                label = f"{rec['ticker']} ({rec['year']})" if rec.get("year") else rec.get("ticker", f"Row {i+1}")
                st.markdown(f"#### {label}")

                text = rec["text"]

                # 1. Loughran & McDonald bag-of-words features
                lm_features = compute_lm_features(text)

                # 2. DistilBERT contextual embedding
                embedding, bert_warn = compute_distilbert_embedding(text)
                if bert_warn:
                    st.warning(bert_warn)

                # 3. Readability indexes (Flesch-Kincaid, Gunning Fog, ARI)
                readability_results = compute_all_readability(text)
                readability_scores = {
                    "fk_grade": readability_results[0]["score"],
                    "fog":      readability_results[1]["score"],
                    "ari":      readability_results[2]["score"],
                }

                # 4. Combined prediction via Logistic Regression (L&M + BERT + Readability)
                prediction = predict_sentiment(lm_features, embedding, readability_scores)

                # --- Display prediction card ---
                pred_label = prediction["label"]
                pred_prob = prediction["probability"]
                pred_color = "green" if "Positive" in pred_label else ("red" if "Negative" in pred_label else "orange")
                st.markdown(f"""
                <div class="score-card {pred_color}">
                    <h3>Sentiment Prediction</h3>
                    <div class="big">{pred_label}</div>
                    <div class="zone">Confidence: {pred_prob:.1%}</div>
                    <div class="interp">{prediction['interpretation']}</div>
                </div>
                """, unsafe_allow_html=True)

                # --- Readability Indexes ---
                with st.expander(f"Readability Indexes — {label}", expanded=False):
                    rd_cols = st.columns(3)
                    for j, rd in enumerate(readability_results):
                        # Color code: green for accessible, orange for dense, red for obfuscating
                        rd_score = rd["score"]
                        if rd_score <= 12:
                            rd_delta_color = "normal"
                        elif rd_score <= 18:
                            rd_delta_color = "off"
                        else:
                            rd_delta_color = "inverse"
                        rd_cols[j].metric(
                            rd["name"],
                            f"{rd_score:.1f}",
                            delta=f"Grade {rd_score:.0f}",
                            delta_color=rd_delta_color,
                        )
                        rd_cols[j].caption(rd["interpretation"])

                    # Component details table
                    comp_rows = []
                    for rd in readability_results:
                        for comp_name, comp_val in rd.get("components", {}).items():
                            comp_rows.append({
                                "Index": rd["name"],
                                "Component": comp_name.replace("_", " ").title(),
                                "Value": comp_val,
                            })
                    if comp_rows:
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                    st.caption("Readability scores feed into the Logistic Regression as additional "
                               "features. Higher complexity (higher grade level) adds a slight "
                               "negative signal — research shows firms under distress tend to "
                               "use more complex, obfuscatory language in filings.")

                # --- L&M Feature Breakdown ---
                with st.expander(f"L&M Dictionary Features — {label}", expanded=False):
                    lm_cols = st.columns(len(LM_CATEGORY_NAMES))
                    for j, cat_name in enumerate(LM_CATEGORY_NAMES):
                        pct_key = f"{cat_name.lower()}_pct"
                        count_key = f"{cat_name.lower()}_count"
                        pct_val = lm_features.get(pct_key, 0)
                        cnt_val = lm_features.get(count_key, 0)
                        lm_cols[j].metric(cat_name, f"{pct_val:.2%}", delta=f"{cnt_val} words")

                    # Bar chart of L&M percentages
                    fig_lm = go.Figure()
                    cats = LM_CATEGORY_NAMES
                    pcts = [lm_features.get(f"{c.lower()}_pct", 0) for c in cats]
                    bar_colors = ["#ef4444", "#22c55e", "#f59e0b", "#a78bfa", "#60a5fa", "#f472b6"]
                    fig_lm.add_trace(go.Bar(x=cats, y=pcts, marker_color=bar_colors[:len(cats)],
                                            text=[f"{p:.3%}" for p in pcts], textposition="outside"))
                    fig_lm.update_layout(title="L&M Word Category Proportions",
                                         yaxis_title="Proportion of Total Words",
                                         template="plotly_dark", height=320,
                                         margin=dict(t=50, b=30))
                    st.plotly_chart(fig_lm, use_container_width=True)

                    st.caption(f"Total words analysed: {lm_features.get('total_words', 0):,} | "
                               f"Net Sentiment (Pos-Neg): {lm_features.get('net_sentiment', 0):.4f}")

                # --- Text preview ---
                with st.expander(f"Text Preview — {label}", expanded=False):
                    st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

                if i < len(sent_texts) - 1:
                    st.markdown("---")


if __name__ == "__main__":
    main()
