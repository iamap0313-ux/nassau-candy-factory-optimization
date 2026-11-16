"""
nassau_core.py  -  shared building blocks used by BOTH pipeline.py and app.py
------------------------------------------------------------------------------
Keeping the reusable logic (constants, distance formula, feature builder, simulation)
in one file means the training script and the Streamlit dashboard can never disagree.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# 1. Business constants taken straight from the project brief
# ----------------------------------------------------------------------------------
FACTORIES = {                       # name -> (latitude, longitude)
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.119140,  -96.181150),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.117500,  -89.971107),
}

PRODUCT_TO_FACTORY = {              # current (legacy) assignment from the brief
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# ----------------------------------------------------------------------------------
# 2. Destination coordinates.  The dataset has no latitude/longitude for customers, only
#    State/Province, so each state is represented by ONE approximate point (its geographic
#    centre; for Canadian provinces the main population centre).  Approximation is OK
#    because we only need *relative* distances between factories.
# ----------------------------------------------------------------------------------
STATE_COORDS = {
 "Alabama":(32.80,-86.79),"Arizona":(34.20,-111.60),"Arkansas":(34.90,-92.40),
 "California":(37.20,-119.50),"Colorado":(39.00,-105.50),"Connecticut":(41.60,-72.70),
 "Delaware":(39.00,-75.50),"District of Columbia":(38.90,-77.00),"Florida":(28.60,-82.40),
 "Georgia":(32.70,-83.40),"Idaho":(44.40,-114.60),"Illinois":(40.00,-89.20),
 "Indiana":(39.90,-86.30),"Iowa":(42.10,-93.50),"Kansas":(38.50,-98.40),
 "Kentucky":(37.50,-85.30),"Louisiana":(31.00,-92.00),"Maine":(45.40,-69.20),
 "Maryland":(39.00,-76.80),"Massachusetts":(42.30,-71.80),"Michigan":(44.30,-85.40),
 "Minnesota":(46.30,-94.30),"Mississippi":(32.70,-89.70),"Missouri":(38.40,-92.50),
 "Montana":(47.00,-109.60),"Nebraska":(41.50,-99.80),"Nevada":(39.30,-116.60),
 "New Hampshire":(43.70,-71.60),"New Jersey":(40.10,-74.70),"New Mexico":(34.40,-106.10),
 "New York":(42.90,-75.50),"North Carolina":(35.60,-79.40),"North Dakota":(47.50,-100.50),
 "Ohio":(40.30,-82.80),"Oklahoma":(35.60,-97.50),"Oregon":(43.90,-120.60),
 "Pennsylvania":(40.90,-77.80),"Rhode Island":(41.70,-71.50),"South Carolina":(33.90,-80.90),
 "South Dakota":(44.40,-100.20),"Tennessee":(35.80,-86.30),"Texas":(31.50,-99.30),
 "Utah":(39.30,-111.70),"Vermont":(44.00,-72.70),"Virginia":(37.50,-78.80),
 "Washington":(47.40,-120.50),"West Virginia":(38.60,-80.60),"Wisconsin":(44.60,-90.00),
 "Wyoming":(43.00,-107.50),
 # Canada (main population centre of each province)
 "Alberta":(51.00,-114.10),"British Columbia":(49.30,-123.10),"Manitoba":(49.90,-97.10),
 "New Brunswick":(46.10,-64.80),"Newfoundland and Labrador":(47.60,-52.70),
 "Nova Scotia":(44.60,-63.60),"Ontario":(43.70,-79.40),"Prince Edward Island":(46.20,-63.10),
 "Quebec":(45.50,-73.60),"Saskatchewan":(52.10,-106.70),
}

SHIP_MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]   # fastest -> slowest
REGIONS = ["Atlantic", "Gulf", "Interior", "Pacific"]
DIVISIONS = ["Chocolate", "Other", "Sugar"]
FACTORY_NAMES = list(FACTORIES.keys())

# Assumption (NOT in the data): freight cost per unit per 1,000 km.  Used only for the
# profit-sensitivity part of the simulation.  The dashboard lets the user change it.
FREIGHT_USD_PER_UNIT_PER_1000KM = 0.15


# ----------------------------------------------------------------------------------
# 3. Distance: great-circle (haversine) distance in km between two lat/lon points
# ----------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_to_state(factory, states):
    """Distance (km) from `factory` to each state in the Series/array `states`."""
    flat, flon = FACTORIES[factory]
    coords = pd.Series(states).map(STATE_COORDS)
    lat = coords.map(lambda t: t[0]).to_numpy()
    lon = coords.map(lambda t: t[1]).to_numpy()
    return haversine_km(flat, flon, lat, lon)


# ----------------------------------------------------------------------------------
# 4. Feature matrix.  The model sees: ship mode, destination region, origin factory,
#    division, units and the factory->customer distance.  Categorical columns are
#    one-hot encoded with FIXED category lists so training and the dashboard always
#    produce exactly the same columns.
# ----------------------------------------------------------------------------------
def build_features(ship_mode, region, factory, division, units, distance_km):
    X = pd.DataFrame({
        "ship_mode_rank": pd.Series(ship_mode).map({m: i for i, m in enumerate(SHIP_MODES)}).to_numpy(),
        "units": np.asarray(units, dtype=float),
        "distance_km": np.asarray(distance_km, dtype=float),
    })
    for r in REGIONS:
        X[f"region_{r}"] = (pd.Series(region).to_numpy() == r).astype(int)
    for f in FACTORY_NAMES:
        X[f"factory_{f}"] = (pd.Series(factory).to_numpy() == f).astype(int)
    for d in DIVISIONS:
        X[f"division_{d}"] = (pd.Series(division).to_numpy() == d).astype(int)
    for m in SHIP_MODES:
        X[f"mode_{m}"] = (pd.Series(ship_mode).to_numpy() == m).astype(int)
    return X


# ----------------------------------------------------------------------------------
# 5. Scenario simulation: "what if this product were made at factory F instead?"
# ----------------------------------------------------------------------------------
def simulate(model, orders, freight_rate=FREIGHT_USD_PER_UNIT_PER_1000KM, tree_model=None, sample_cap=None):
    """
    `orders`  : DataFrame of historical orders (needs Product Name, Ship Mode, Region,
                State/Province, Division, Units, Gross Profit, Factory).
    For every order and every candidate factory we predict lead time and estimate freight.
    Returns one row per (product, candidate factory) with the aggregate effect.
    """
    rows = []
    for product, g in orders.groupby("Product Name"):
        n_full = len(g)
        # `sample_cap` keeps the live dashboard fast: simulate on a random sample, then scale dollar totals back up
        if sample_cap and n_full > sample_cap:
            g = g.sample(sample_cap, random_state=0)
        scale = n_full / len(g)
        current = PRODUCT_TO_FACTORY[product]
        base_d = distance_to_state(current, g["State/Province"])
        base_X = build_features(g["Ship Mode"], g["Region"], [current] * len(g), g["Division"], g["Units"], base_d)
        base_lt = model.predict(base_X)
        base_freight = freight_rate * g["Units"].to_numpy() * base_d / 1000.0
        for cand in FACTORY_NAMES:
            d = distance_to_state(cand, g["State/Province"])
            X = build_features(g["Ship Mode"], g["Region"], [cand] * len(g), g["Division"], g["Units"], d)
            new_lt = model.predict(X)
            freight = freight_rate * g["Units"].to_numpy() * d / 1000.0
            d_profit = base_freight - freight                       # + means profit goes UP
            gp = g["Gross Profit"].to_numpy()
            # agreement between individual trees = how sure the forest is about the sign
            agree = np.nan
            if tree_model is not None:
                tb = np.array([t.predict(base_X.to_numpy()) for t in tree_model.estimators_])
                tn = np.array([t.predict(X.to_numpy()) for t in tree_model.estimators_])
                per_tree = (tb - tn).mean(axis=1)                  # lead-time days saved, per tree
                mean_delta = per_tree.mean()
                agree = float(np.mean(np.sign(per_tree) == np.sign(mean_delta))) if mean_delta != 0 else 0.5
            rows.append({
                "Product": product, "Division": g["Division"].iloc[0],
                "Current Factory": current, "Candidate Factory": cand,
                "Is Current": cand == current, "Orders": n_full,
                "Base Lead Time (days)": base_lt.mean(), "New Lead Time (days)": new_lt.mean(),
                "Lead Time Change (days)": new_lt.mean() - base_lt.mean(),
                "Lead Time Reduction (%)": 100 * (base_lt.mean() - new_lt.mean()) / base_lt.mean(),
                "Avg Distance Before (km)": base_d.mean(), "Avg Distance After (km)": d.mean(),
                "Profit Impact ($)": d_profit.sum() * scale,
                "Profit Impact (%)": 100 * d_profit.sum() / gp.sum(),
                "Share of Orders Not Hurt (%)": 100 * np.mean(d_profit >= -0.005 * gp),
                "Tree Agreement": agree,
            })
    return pd.DataFrame(rows)


def compute_kpis(scored, n_products_total):
    """The four KPIs from the brief, computed on a scored simulation table."""
    alts = scored[~scored["Is Current"]]
    rec = alts[alts["Verdict"] == "Recommended"]
    if rec.empty:
        return {"Lead Time Reduction (%)": 0.0, "Profit Impact Stability (%)": 0.0,
                "Scenario Confidence Score": 0.0, "Recommendation Coverage (%)": 0.0}
    best = rec.sort_values("Priority Score", ascending=False).groupby("Product").head(1)
    return {"Lead Time Reduction (%)": float(best["Lead Time Reduction (%)"].mean()),
            "Profit Impact Stability (%)": float(best["Share of Orders Not Hurt (%)"].mean()),
            "Scenario Confidence Score": float(best["Confidence Score"].mean()),
            "Recommendation Coverage (%)": 100 * best["Product"].nunique() / n_products_total}


def add_scores(sim, factory_volume, priority=0.5, min_agreement_floor=0.5):
    """
    Turns raw simulation rows into KPIs, a risk score and a ranking score.
    priority : 0 = care only about profit ... 1 = care only about speed.
    """
    s = sim.copy()
    # --- Risk (0-100): capacity strain + model uncertainty + thin data -----------------
    cand_vol = s["Candidate Factory"].map(factory_volume).clip(lower=1)
    ratio = np.where(s["Is Current"], 0.0, s["Orders"] / cand_vol)           # volume we'd push on the factory
    strain = np.clip(np.log10(1 + ratio) / np.log10(1 + 10), 0, 1)
    uncertainty = (1 - s["Tree Agreement"].fillna(0.5)).clip(0, 0.5) * 2    # 0 (sure) .. 1 (coin-flip)
    support = np.clip(s["Orders"] / 200.0, 0, 1)                            # <200 orders = thin evidence
    s["Capacity Strain"] = strain
    s["Risk Score"] = 100 * (0.5 * strain + 0.3 * uncertainty + 0.2 * (1 - support))
    s["Confidence Score"] = 100 * (0.6 * s["Tree Agreement"].fillna(0.5) + 0.4 * support)
    s["Evidence"] = np.where(s["Confidence Score"] >= 70, "Strong", np.where(s["Confidence Score"] >= 45, "Moderate", "Weak"))
    # --- Ranking score -------------------------------------------------------------
    lt = s["Lead Time Reduction (%)"]; pr = s["Profit Impact (%)"]
    lt_n = (lt - lt.min()) / (lt.max() - lt.min() + 1e-9)
    pr_n = (pr - pr.min()) / (pr.max() - pr.min() + 1e-9)
    benefit = priority * lt_n + (1 - priority) * pr_n
    s["Priority Score"] = 100 * benefit * (1 - 0.5 * s["Risk Score"] / 100)
    # --- Verdict --------------------------------------------------------------------
    # Thresholds (all adjustable): a lead-time change smaller than 0.05 day (~1 hour) is far below the model's
    # ~1-day typical error, so it is treated as "no real change" rather than as a gain or a loss.
    def verdict(r):
        if r["Is Current"]: return "Current assignment"
        if r["Risk Score"] >= 60: return "High risk"                 # capacity strain / uncertain
        if r["Profit Impact (%)"] < 0: return "Profit loss"
        if r["Lead Time Change (days)"] > 0.05: return "Slower"
        if r["Profit Impact (%)"] < 0.5: return "Marginal gain"
        return "Recommended"
    s["Verdict"] = s.apply(verdict, axis=1)
    return s
