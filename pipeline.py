"""
pipeline.py  -  runs the whole analysis end to end:
  STEP 1  load raw data and rebuild a trustworthy lead time
  STEP 2  clean / remove outliers / add factory + distance
  STEP 3  exploratory data analysis (charts)
  STEP 4  encode + scale -> train 3 models -> compare (RMSE, MAE, R2)
  STEP 5  cluster routes (factory -> state) and find slow / congested combinations
  STEP 6  scenario simulation + scoring + top-N recommendations + KPIs
Run:  python pipeline.py        (takes about a minute)
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nassau_core import *

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)
OUT = "outputs/"
RNG = 42

# =====================================================================================
# STEP 1 - LOAD + REBUILD LEAD TIME
# =====================================================================================
raw = pd.read_csv("data/Nassau_Candy_Distributor.csv")
raw["od"] = pd.to_datetime(raw["Order Date"], format="%d-%m-%Y")
raw["sd"] = pd.to_datetime(raw["Ship Date"], format="%d-%m-%Y")
raw["raw_gap"] = (raw["sd"] - raw["od"]).dt.days                      # 904 ... 1642 days: clearly wrong!
raw["order_year"] = raw["Order ID"].str.split("-").str[1].astype(int)  # the year inside the Order ID is trustworthy
# The date columns were shifted by a different fixed amount per year. A "Same Day" order must have a
# true lead time of 0, so the typical Same-Day gap of each year IS that year's shift. Subtract it.
shift = raw[raw["Ship Mode"] == "Same Day"].groupby("order_year")["raw_gap"].median()
raw["Lead Time (days)"] = raw["raw_gap"] - raw["order_year"].map(shift)
print("Year shifts removed (days):", shift.to_dict())

# =====================================================================================
# STEP 2 - CLEAN: factory, distance, outliers
# =====================================================================================
df = raw.copy()
df["Factory"] = df["Product Name"].map(PRODUCT_TO_FACTORY)
df["Distance (km)"] = np.nan
for f in FACTORY_NAMES:
    m = df["Factory"] == f
    df.loc[m, "Distance (km)"] = distance_to_state(f, df.loc[m, "State/Province"])
df["Margin (%)"] = 100 * df["Gross Profit"] / df["Sales"]
df["Freight Est ($)"] = FREIGHT_USD_PER_UNIT_PER_1000KM * df["Units"] * df["Distance (km)"] / 1000

n0 = len(df)
q1, q3 = df["Sales"].quantile([.25, .75]); iqr = q3 - q1
sales_hi = q3 + 3 * iqr                                                 # "extreme" outlier fence (3 x IQR)
g = df.groupby("Ship Mode")["Lead Time (days)"]
lt_hi = g.transform(lambda s: s.quantile(.75) + 3 * (s.quantile(.75) - s.quantile(.25)))
mask = (df["Sales"] <= sales_hi) & (df["Lead Time (days)"] <= lt_hi) & (df["Lead Time (days)"] >= 0)
removed = int((~mask).sum()); df = df[mask].reset_index(drop=True)
print(f"Outliers removed: {removed} of {n0} rows ({100*removed/n0:.2f}%)")
df.drop(columns=["od", "sd"]).to_csv(OUT + "cleaned_data.csv", index=False)

# =====================================================================================
# STEP 3 - EDA CHARTS
# =====================================================================================
LT = "Lead Time (days)"
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x="Ship Mode", y=LT, order=SHIP_MODES, ax=ax[0], color="#9ecae1")
ax[0].set_title("Lead time by ship mode"); ax[0].set_xlabel("")
sns.barplot(data=df, x="Region", y=LT, order=REGIONS, ax=ax[1], color="#9ecae1", errorbar=None)
ax[1].set_title("Average lead time by region"); ax[1].set_ylim(0, 6)
for p in ax[1].patches: ax[1].annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(OUT + "fig1_leadtime_modes_regions.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
prod = df.groupby("Product Name").agg(Sales=("Sales", "sum"), GP=("Gross Profit", "sum"), Orders=("Row ID", "count")).sort_values("Sales")
ax[0].barh(prod.index, prod["Sales"], color="#fdae6b"); ax[0].set_title("Sales by product ($)"); ax[0].tick_params(axis="y", labelsize=7)
fac = df.groupby("Factory").agg(Orders=("Row ID", "count"), Lead=(LT, "mean"), Dist=("Distance (km)", "mean")).sort_values("Orders")
ax[1].barh(fac.index, fac["Orders"], color="#a1d99b"); ax[1].set_title("Orders handled by factory"); 
for i, (o, l) in enumerate(zip(fac["Orders"], fac["Lead"])): ax[1].text(o, i, f"  {o:,} orders | {l:.2f}d", va="center", fontsize=8)
ax[1].set_xlim(0, fac["Orders"].max() * 1.45)
plt.tight_layout(); plt.savefig(OUT + "fig2_products_factories.png", dpi=150); plt.close()

df["Distance band"] = pd.cut(df["Distance (km)"], [0, 1000, 1800, 2600, 6000], labels=["<1000 km", "1000-1800", "1800-2600", ">2600"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.pointplot(data=df, x="Distance band", y=LT, hue="Ship Mode", hue_order=SHIP_MODES, errorbar=None, ax=ax)
ax.set_title("Lead time barely changes with distance, but ship mode shifts the whole line")
ax.title.set_fontsize(10)
plt.tight_layout(); plt.savefig(OUT + "fig3_distance_vs_leadtime.png", dpi=150); plt.close()
corr_by_mode = {m: float(np.corrcoef(d["Distance (km)"], d[LT])[0, 1]) for m, d in df.groupby("Ship Mode")}
print("Correlation distance vs lead time within each ship mode:", {k: round(v, 3) for k, v in corr_by_mode.items()})

div = df.groupby("Division").agg(Sales=("Sales", "sum"), GP=("Gross Profit", "sum"))
reg = df.groupby("Region").agg(Sales=("Sales", "sum"), GP=("Gross Profit", "sum"))
eda_tables = {"division": div.round(2).to_dict("index"), "region": reg.round(2).to_dict("index"),
              "margin_overall_pct": float(100 * df["Gross Profit"].sum() / df["Sales"].sum()),
              "margin_by_mode_pct": (100 * df.groupby("Ship Mode")["Gross Profit"].sum() / df.groupby("Ship Mode")["Sales"].sum()).round(2).to_dict(),
              "lead_by_mode": df.groupby("Ship Mode")[LT].mean().round(2).to_dict(),
              "lead_by_region": df.groupby("Region")[LT].mean().round(2).to_dict(),
              "mode_share_pct": (100 * df["Ship Mode"].value_counts(normalize=True)).round(1).to_dict(),
              "orders_by_factory": df["Factory"].value_counts().to_dict(),
              "avg_distance_by_factory": df.groupby("Factory")["Distance (km)"].mean().round(0).to_dict(),
              "corr_distance_leadtime_by_mode": corr_by_mode, "rows_after_cleaning": len(df), "outliers_removed": removed}

# =====================================================================================
# STEP 4 - FEATURES + MODELS
# =====================================================================================
X = build_features(df["Ship Mode"], df["Region"], df["Factory"], df["Division"], df["Units"], df["Distance (km)"])
y = df[LT].to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG)
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),            # scaling matters for linear models
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_depth=12, random_state=RNG, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RNG),
}
res = []
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res.append({"Model": name, "RMSE": np.sqrt(mean_squared_error(yte, p)), "MAE": mean_absolute_error(yte, p), "R2": r2_score(yte, p)})
res = pd.DataFrame(res).round(4); res.to_csv(OUT + "model_metrics.csv", index=False)
print(res.to_string(index=False))
# Baseline for context: always predict the average
base_rmse = float(np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean(), dtype=float)))); print("Naive (mean) RMSE:", round(base_rmse, 4))

# Does the factory / distance information add anything beyond ship mode + region?  (ablation)
def quick_r2(cols):
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RNG).fit(Xtr[cols], ytr)
    return float(r2_score(yte, m.predict(Xte[cols])))
mode_cols = [c for c in X.columns if c.startswith("mode_") or c == "ship_mode_rank"]
loc_cols = ["distance_km"] + [c for c in X.columns if c.startswith("factory_") or c.startswith("region_")]
ablation = {"ship mode only": quick_r2(mode_cols), "ship mode + units + division": quick_r2(mode_cols + ["units"] + [c for c in X.columns if c.startswith("division_")]),
            "ship mode + distance/factory/region": quick_r2(mode_cols + loc_cols), "all features": quick_r2(list(X.columns))}
print("Ablation R2:", {k: round(v, 4) for k, v in ablation.items()})

best_name = res.sort_values(["RMSE"]).iloc[0]["Model"]
print("Best by RMSE:", best_name)
# Interpretability: Linear Regression coefficients and Random Forest importances
lr = models["Linear Regression"]; rf = models["Random Forest"]; gb = models["Gradient Boosting"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp_gb = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
res.set_index("Model")[["RMSE", "MAE"]].plot.bar(ax=ax[0], rot=15, color=["#6baed6", "#fd8d3c"]); ax[0].set_title("Error (lower is better)"); ax[0].set_xlabel("")
ax[1].bar(res["Model"], res["R2"], color="#74c476"); ax[1].set_title("R-squared (higher is better)"); ax[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(res["R2"]): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
imp.head(8)[::-1].plot.barh(ax=ax[2], color="#9e9ac8"); ax[2].set_title("Random Forest: top features")
plt.tight_layout(); plt.savefig(OUT + "fig4_model_comparison.png", dpi=150); plt.close()
pred = (rf if best_name == "Random Forest" else gb if best_name == "Gradient Boosting" else lr).predict(Xte)
fig, ax = plt.subplots(figsize=(5, 4.5)); ax.scatter(yte + np.random.RandomState(0).uniform(-.15, .15, len(yte)), pred, s=6, alpha=.3)
ax.plot([0, 11], [0, 11], "r--", lw=1); ax.set_xlabel("Actual lead time (days)"); ax.set_ylabel("Predicted"); ax.set_title(f"{best_name}: predicted vs actual (test set)")
plt.tight_layout(); plt.savefig(OUT + "fig5_pred_vs_actual.png", dpi=150); plt.close()

# Final model used for simulation = best by RMSE; Random Forest is also kept for the per-tree agreement score.
final = {"Linear Regression": lr, "Random Forest": rf, "Gradient Boosting": gb}[best_name]
joblib.dump({"model": final, "rf": rf, "model_name": best_name, "columns": list(X.columns)}, OUT + "model.joblib")

# =====================================================================================
# STEP 5 - ROUTE CLUSTERING
# =====================================================================================
# "Slowness" must be judged fairly: a route that mostly uses Standard Class is slow only because of the ship mode.
# So we also compute EXCESS lead time = actual lead time minus the average for that ship mode.
df["Excess Lead (days)"] = df[LT] - df.groupby("Ship Mode")[LT].transform("mean")
route = df.groupby(["Factory", "State/Province"]).agg(
    Orders=("Row ID", "count"), AvgLead=(LT, "mean"), StdLead=(LT, "std"), AvgExcess=("Excess Lead (days)", "mean"),
    AvgDist=("Distance (km)", "mean"), StandardShare=("Ship Mode", lambda s: (s == "Standard Class").mean())).reset_index()
route["StdLead"] = route["StdLead"].fillna(0)
route = route[route["Orders"] >= 5].reset_index(drop=True)             # tiny routes are too noisy to cluster
route["LogOrders"] = np.log10(route["Orders"])
feat = ["AvgLead", "StdLead", "AvgExcess", "AvgDist", "LogOrders"]    # performance only (margin is a product trait, not a route trait)
Z = StandardScaler().fit_transform(route[feat])
sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=RNG).fit_predict(Z)) for k in range(3, 7)}   # at least 3 groups so the result is useful
k = max(sil, key=sil.get); route["Cluster"] = KMeans(k, n_init=10, random_state=RNG).fit_predict(Z)
cs = route.groupby("Cluster")[feat].mean().round(2); cs["Routes"] = route.groupby("Cluster").size()
print("Silhouette by k:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k); print(cs)
order = cs["AvgExcess"].rank(method="first").astype(int)
labels = {c: ("Faster than expected" if order[c] == 1 else "Slower than expected" if order[c] == len(cs) else f"Typical (group {order[c]})") for c in cs.index}
route["Cluster Label"] = route["Cluster"].map(labels); route.round(3).to_csv(OUT + "route_clusters.csv", index=False)
cs["Label"] = cs.index.map(labels); cs.to_csv(OUT + "cluster_summary.csv")
fig, ax = plt.subplots(figsize=(7.5, 5))
sns.scatterplot(data=route, x="AvgDist", y="AvgLead", hue="Cluster Label", size="Orders", sizes=(20, 250), alpha=.7, ax=ax)
ax.set_xlabel("Average distance factory -> state (km)"); ax.set_ylabel("Average lead time (days)"); ax.set_title("Route clusters (each dot = factory -> state)")
plt.tight_layout(); plt.savefig(OUT + "fig6_route_clusters.png", dpi=150); plt.close()
slow_routes = route.sort_values("AvgExcess", ascending=False).query("Orders >= 30").head(8)
# Congestion: product x region combinations with high volume AND above-average lead time
pr = df.groupby(["Product Name", "Region"]).agg(Orders=("Row ID", "count"), AvgLead=(LT, "mean")).reset_index()
pr["Congested"] = (pr["Orders"] >= pr["Orders"].quantile(.75)) & (pr["AvgLead"] > df[LT].mean())
pr.round(3).to_csv(OUT + "product_region_congestion.csv", index=False)

# =====================================================================================
# STEP 6 - SIMULATION, SCORING, RECOMMENDATIONS, KPIs
# =====================================================================================
vol = df["Factory"].value_counts().to_dict()
sim = simulate(final, df, tree_model=rf)
scored = add_scores(sim, vol, priority=0.5)
scored.round(4).to_csv(OUT + "simulation_all_scenarios.csv", index=False)
alts = scored[~scored["Is Current"]].copy()
alts["Rank in Product"] = alts.groupby("Product")["Priority Score"].rank(ascending=False, method="first").astype(int)
top = alts[alts["Rank in Product"] <= 3].sort_values(["Product", "Rank in Product"])
top.round(4).to_csv(OUT + "top3_per_product.csv", index=False)
rec = alts[alts["Verdict"] == "Recommended"].sort_values("Priority Score", ascending=False)
rec.round(4).to_csv(OUT + "recommended_reassignments.csv", index=False)
best_per_prod = alts.sort_values("Priority Score", ascending=False).groupby("Product").head(1)
kp = compute_kpis(scored, df["Product Name"].nunique())
kpi = {
    "avg_lead_time_reduction_pct_recommended": kp["Lead Time Reduction (%)"],
    "profit_impact_stability_pct": kp["Profit Impact Stability (%)"],
    "scenario_confidence_score": kp["Scenario Confidence Score"],
    "recommendation_coverage_pct": kp["Recommendation Coverage (%)"],
    "products_with_recommendation": int(rec["Product"].nunique()), "products_total": int(df["Product Name"].nunique()),
    "scenarios_simulated": int(len(alts)), "scenarios_recommended": int(len(rec)),
}
print(json.dumps(kpi, indent=1))
fig, ax = plt.subplots(figsize=(9, 5))
bp = best_per_prod.sort_values("Profit Impact ($)")
lab = bp["Product"] + "  ->  " + bp["Candidate Factory"]
colors = bp["Verdict"].map({"Recommended": "#31a354", "High risk": "#de2d26", "Profit loss": "#fd8d3c", "Slower": "#969696", "Marginal gain": "#bdbdbd"}).fillna("#bdbdbd")
ax.barh(lab, bp["Lead Time Reduction (%)"], color=colors); ax.set_xlabel("Predicted lead-time reduction (%)"); ax.tick_params(axis="y", labelsize=7)
ax.set_title("Best-scoring alternative factory per product (green = recommended, red = high risk)", fontsize=10)
plt.tight_layout(); plt.savefig(OUT + "fig7_best_alternative.png", dpi=150); plt.close()

# Sensitivity of the profit side to the freight-rate assumption (so readers see it is an assumption)
sens = []
for rate in [0.05, 0.15, 0.30]:
    s2 = add_scores(simulate(final, df, freight_rate=rate, tree_model=rf), vol, 0.5); a2 = s2[~s2["Is Current"]]
    r2 = a2[a2["Verdict"] == "Recommended"]
    sens.append({"freight_rate": rate, "recommended_scenarios": int(len(r2)), "products_covered": int(r2["Product"].nunique()),
                 "total_profit_impact_best_recs": float(r2.sort_values("Priority Score", ascending=False).groupby("Product").head(1)["Profit Impact ($)"].sum())})
pd.DataFrame(sens).to_csv(OUT + "freight_sensitivity.csv", index=False)

json.dump({"eda": eda_tables, "ablation": ablation, "best_model": best_name, "naive_rmse": base_rmse, "kpi": kpi,
           "silhouette": sil, "k": int(k), "year_shifts": {str(a): int(b) for a, b in shift.items()}, "freight_sensitivity": sens,
           "slow_routes": slow_routes[["Factory", "State/Province", "Orders", "AvgLead", "AvgExcess", "AvgDist"]].round(2).to_dict("records"),
           "congested": pr[pr["Congested"]].round(2).to_dict("records"),
           "top_features_rf": imp.head(6).round(4).to_dict()}, open(OUT + "results_summary.json", "w"), indent=1, default=float)
print("Done. See the outputs/ folder.")
