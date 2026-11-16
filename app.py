"""
app.py - Streamlit dashboard for the Nassau Candy factory-reassignment project.
Run (after `python pipeline.py` has created the outputs/ folder):
    streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nassau_core import (FACTORY_NAMES, REGIONS, SHIP_MODES, PRODUCT_TO_FACTORY, FREIGHT_USD_PER_UNIT_PER_1000KM,
                         simulate, add_scores, compute_kpis)

OUT = "outputs/"
VERDICT_COLORS = {"Recommended": "#31a354", "High risk": "#de2d26", "Profit loss": "#fd8d3c",
                  "Slower": "#969696", "Marginal gain": "#bdbdbd", "Current assignment": "#3182bd"}


# ---------------------------------------------------------------- cached loaders
@st.cache_resource
def load_model():
    return joblib.load(OUT + "model.joblib")


@st.cache_data
def load_orders():
    return pd.read_csv(OUT + "cleaned_data.csv")


@st.cache_data(show_spinner="Simulating scenarios...")
def run_simulation(region, modes, freight_rate):
    """Simulate every product x factory on the orders that match the sidebar filters."""
    bundle = load_model(); orders = load_orders()
    if region != "All regions":
        orders = orders[orders["Region"] == region]
    orders = orders[orders["Ship Mode"].isin(modes)]
    if orders.empty:
        return pd.DataFrame(), {}
    sim = simulate(bundle["model"], orders, freight_rate=freight_rate, tree_model=bundle["rf"], sample_cap=400)
    volume = load_orders()["Factory"].value_counts().to_dict()        # capacity is judged on ALL orders, not the filter
    return sim, volume


def scored_table(region, modes, freight_rate, priority):
    sim, volume = run_simulation(region, tuple(modes), freight_rate)
    if sim.empty:
        return sim
    return add_scores(sim, volume, priority=priority)


# ---------------------------------------------------------------- small helpers
def bar_by_factory(df, value_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = [VERDICT_COLORS["Current assignment"] if cur else "#9ecae1" for cur in df["Is Current"]]
    ax.bar(df["Candidate Factory"], df[value_col], color=colors)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(df[value_col]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def show_cols(df):
    cols = ["Product", "Current Factory", "Candidate Factory", "Base Lead Time (days)", "New Lead Time (days)",
            "Lead Time Reduction (%)", "Avg Distance After (km)", "Profit Impact ($)", "Profit Impact (%)",
            "Risk Score", "Confidence Score", "Evidence", "Priority Score", "Verdict"]
    return df[cols].round(2)


# ---------------------------------------------------------------- the app
def main():
    st.set_page_config(page_title="Nassau Candy - Factory Optimization", layout="wide")
    st.title("Nassau Candy - Factory Reassignment Decision Support")
    st.caption("Predicts shipping lead time under different factory assignments and recommends moves that balance speed, profit and risk.")

    bundle = load_model(); orders = load_orders()
    products = sorted(orders["Product Name"].unique())

    # ---- sidebar controls (the "User Capabilities" from the brief)
    st.sidebar.header("Controls")
    product = st.sidebar.selectbox("Product", products, index=products.index("Wonka Bar - Milk Chocolate"))
    region = st.sidebar.selectbox("Region", ["All regions"] + REGIONS)
    modes = st.sidebar.multiselect("Ship mode", SHIP_MODES, default=SHIP_MODES)
    if not modes:
        st.sidebar.warning("Pick at least one ship mode."); modes = SHIP_MODES
    priority = st.sidebar.slider("Optimization priority  (0 = profit, 100 = speed)", 0, 100, 50) / 100
    freight = st.sidebar.number_input("Assumed freight cost ($ per unit per 1,000 km)", 0.0, 2.0, FREIGHT_USD_PER_UNIT_PER_1000KM, 0.05,
                                      help="Not in the dataset - an assumption that drives the profit side of the simulation.")
    top_n = st.sidebar.number_input("Top-N recommendations", 1, 30, 10)
    st.sidebar.markdown(f"**Model used:** {bundle['model_name']}")

    scored = scored_table(region, modes, freight, priority)
    if scored.empty:
        st.warning("No orders match these filters."); return
    n_products = orders["Product Name"].nunique()

    tab1, tab2, tab3, tab4 = st.tabs(["Factory Optimization Simulator", "What-If Scenario Analysis",
                                      "Recommendation Dashboard", "Risk & Impact Panel"])

    # ---------------- TAB 1
    with tab1:
        p = scored[scored["Product"] == product].reset_index(drop=True)
        st.subheader(f"{product}  -  predicted performance by factory")
        st.write(f"Currently made at **{PRODUCT_TO_FACTORY[product]}**. Based on {int(p['Orders'].iloc[0]):,} historical orders matching your filters.")
        c1, c2 = st.columns(2)
        c1.pyplot(bar_by_factory(p, "New Lead Time (days)", "Predicted average lead time (days)", "days"))
        c2.pyplot(bar_by_factory(p, "Avg Distance After (km)", "Average distance to customers (km)", "km"))
        st.dataframe(show_cols(p), use_container_width=True, hide_index=True)

    # ---------------- TAB 2
    with tab2:
        st.subheader("Current vs recommended assignment")
        alts = scored[(scored["Product"] == product) & (~scored["Is Current"])].sort_values("Priority Score", ascending=False)
        cand = st.selectbox("Compare against factory", list(alts["Candidate Factory"]), index=0)
        cur = scored[(scored["Product"] == product) & (scored["Is Current"])].iloc[0]
        new = alts[alts["Candidate Factory"] == cand].iloc[0]
        m = st.columns(4)
        m[0].metric("Lead time (days)", f"{new['New Lead Time (days)']:.2f}", f"{new['Lead Time Change (days)']:+.2f}", delta_color="inverse")
        m[1].metric("Avg distance (km)", f"{new['Avg Distance After (km)']:.0f}", f"{new['Avg Distance After (km)'] - cur['Avg Distance Before (km)']:+.0f}", delta_color="inverse")
        m[2].metric("Profit impact ($)", f"{new['Profit Impact ($)']:+,.0f}", f"{new['Profit Impact (%)']:+.2f}%")
        m[3].metric("Risk score", f"{new['Risk Score']:.0f}/100", new["Verdict"], delta_color="off")
        # lead-time comparison by region for this product
        bundle = load_model(); from nassau_core import build_features, distance_to_state
        o = orders[(orders["Product Name"] == product) & (orders["Ship Mode"].isin(modes))]
        rows = []
        for r in REGIONS:
            g = o[o["Region"] == r]
            if g.empty: continue
            for label, fac in [("Current", PRODUCT_TO_FACTORY[product]), (f"{cand}", cand)]:
                X = build_features(g["Ship Mode"], g["Region"], [fac] * len(g), g["Division"], g["Units"], distance_to_state(fac, g["State/Province"]))
                rows.append({"Region": r, "Scenario": label, "Lead": bundle["model"].predict(X).mean()})
        rr = pd.DataFrame(rows)
        if not rr.empty:
            piv = rr.pivot(index="Region", columns="Scenario", values="Lead")
            fig, ax = plt.subplots(figsize=(7, 3.4)); piv.plot.bar(ax=ax, rot=0, color=["#9ecae1", "#3182bd"][:piv.shape[1]])
            ax.set_ylabel("Predicted lead time (days)"); ax.set_title("Lead time by region: current vs candidate factory", fontsize=10)
            fig.tight_layout(); st.pyplot(fig)
        st.caption("Differences are small because lead time in this data is driven mainly by ship mode - see the Risk & Impact Panel.")

    # ---------------- TAB 3
    with tab3:
        kp = compute_kpis(scored, n_products)
        k = st.columns(4)
        k[0].metric("Lead Time Reduction (%)", f"{kp['Lead Time Reduction (%)']:.2f}%")
        k[1].metric("Profit Impact Stability", f"{kp['Profit Impact Stability (%)']:.0f}%")
        k[2].metric("Scenario Confidence Score", f"{kp['Scenario Confidence Score']:.0f}/100")
        k[3].metric("Recommendation Coverage", f"{kp['Recommendation Coverage (%)']:.0f}%")
        st.subheader(f"Top {int(top_n)} reassignment suggestions")
        alts_all = scored[~scored["Is Current"]].copy()
        rec = alts_all[alts_all["Verdict"] == "Recommended"].sort_values("Priority Score", ascending=False).head(int(top_n))
        if rec.empty:
            st.info("No scenario passes all checks with these settings. Try a higher freight rate or a wider filter.")
        else:
            st.dataframe(show_cols(rec), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(rec) + 1.5))
            lab = rec["Product"] + "  ->  " + rec["Candidate Factory"]
            ax.barh(lab[::-1], rec["Priority Score"][::-1], color="#31a354"); ax.set_xlabel("Priority score"); ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout(); st.pyplot(fig)

    # ---------------- TAB 4
    with tab4:
        st.subheader("Risk & impact alerts")
        alts_all = scored[~scored["Is Current"]]
        hi = alts_all[alts_all["Verdict"] == "High risk"]
        loss = alts_all[alts_all["Verdict"] == "Profit loss"]
        weak = alts_all[(alts_all["Verdict"] == "Recommended") & (alts_all["Evidence"] == "Weak")]
        a, b, c = st.columns(3)
        a.metric("High-risk reassignments", len(hi)); b.metric("Scenarios that lose profit", len(loss)); c.metric("Recommended but weak evidence", len(weak))
        st.error("High-risk: the target factory would have to absorb a product with far more volume than it handles today (capacity strain), or the model is unsure.")
        st.dataframe(show_cols(hi.sort_values("Risk Score", ascending=False)), use_container_width=True, hide_index=True)
        st.warning("Profit-loss alerts: the new factory is farther from customers, so estimated freight cost rises.")
        st.dataframe(show_cols(loss.sort_values("Profit Impact ($)")), use_container_width=True, hide_index=True)
        st.info("Reading the results: the model's typical error is about 1 day, while simulated lead-time changes are a fraction of a day. "
                "Treat speed gains as directional; the freight/distance saving is the more reliable benefit, and it depends on the freight-cost assumption in the sidebar.")


if __name__ == "__main__":
    main()
