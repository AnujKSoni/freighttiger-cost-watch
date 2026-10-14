"""Core numbers: weekly cost per tonne-km per route, the two graded baselines, and the flag rule.

Pure pandas, no LLM. Every number the output CSV reports comes from here.
"""
import pandas as pd

HISTORY_WEEKS = 8     # own-history window: trailing 8 weeks, strictly before the current week
OWN_RISE_PCT = 10.0   # anomalous if this far above the route's own trailing average...
PEER_GAP_PCT = 20.0   # ...or this far above same-type peers in the same week (and rising at all)


def load_shipments(path):
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    # Weeks run Monday-Sunday; week_of is that week's Monday.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_costs(df):
    """One row per (route, week): cost per tonne-km, both baselines, % gaps and the anomaly flag."""
    w = (df.groupby(["route", "route_type", "week_of"], as_index=False)
           .agg(shipments=("shipment_id", "size"),
                freight_cost=("freight_cost_inr", "sum"),
                tonne_km=("tonne_km", "sum")))
    # Pooled: total cost / total tonne-km. Reproduces sample_output exactly; a mean of
    # per-shipment ratios does not.
    w["cost_per_tonne_km"] = w["freight_cost"] / w["tonne_km"]
    w = w.sort_values(["route", "week_of"], ignore_index=True)

    # vs own history: mean of this route's weekly values in the 8 calendar weeks strictly
    # before this one. Early weeks use whatever history exists; week 1 has none.
    for _, g in w.groupby("route"):
        past = g.set_index("week_of")["cost_per_tonne_km"].rolling(f"{7 * HISTORY_WEEKS}D", closed="left")
        w.loc[g.index, "own_baseline"] = past.mean().to_numpy()
        w.loc[g.index, "history_weeks"] = past.count().to_numpy()
    w["history_weeks"] = w["history_weeks"].fillna(0).astype(int)

    # vs similar routes: same route_type, same week, excluding the route itself.
    same = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    n = same.transform("count")
    w["peer_baseline"] = ((same.transform("sum") - w["cost_per_tonne_km"]) / (n - 1)).where(n > 1)

    w["own_pct"] = (w["cost_per_tonne_km"] / w["own_baseline"] - 1) * 100
    w["peer_pct"] = (w["cost_per_tonne_km"] / w["peer_baseline"] - 1) * 100
    # NaN comparisons are False, so a week without a baseline is never flagged.
    w["anomalous"] = (w["own_pct"] > 0) & ((w["own_pct"] >= OWN_RISE_PCT) | (w["peer_pct"] >= PEER_GAP_PCT))
    return w
