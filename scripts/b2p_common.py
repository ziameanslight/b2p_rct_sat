"""B2P panel helpers shared by b2p_features.py, b2p_dhs_ridge.py and b2p_item_scatter.py."""
import numpy as np
import pandas as pd

from config import DATA, SURVEY

ROUNDS = ["Baseline", "Follow-up round 1", "Follow-up round 2", "Follow-up round 3"]
SHORT = dict(zip(ROUNDS, ["B", "R1", "R2", "R3"]))
WAVES = ["R1", "R2", "R3"]
# river-crossing questions (food market, bank, government office, health facility)
XQ = ["To get from your home to that location, would you need to cross a river? ",
      "Do you have to cross a river to reach that bank?", "Do you have to cross a river to reach that office?",
      "To get from your home to that health facility, would you need to cross a river?"]


def imagery_year(date):
    """AEF annual composite matched to a visit: calendar year of (interview date - 6 months), i.e. the composite that
    best covers the 12 months before the interview; makes the imagery gap equal the survey gap (13-month rounds that
    straddle New Year otherwise span 0 or 2 composites)."""
    return (date - pd.DateOffset(months=6)).dt.year.clip(2020, 2024)


def load():
    """Household x round panel (data/hh_rounds.csv) joined to the bridge match (data/hh_points.parquet); matched
    households only; r = B / R1 / R2 / R3, year = imagery year."""
    d = pd.read_csv(DATA / "hh_rounds.csv", parse_dates=["date"])
    d = d[d["round"].isin(ROUNDS)].drop(columns=["site", "district"])
    H = pd.read_parquet(DATA / "hh_points.parquet")[["uid", "site", "bridge", "arm", "close"]]
    d = d.merge(H, on="uid")
    d["year"] = imagery_year(d.date)
    d["log_cons"] = np.log1p(d.cons_week)
    d["r"] = d["round"].map(SHORT)
    return d[d.arm != "unmatched"].copy()


def village_sides():
    """uid -> village (unit = site | village name) and river side. Crossing share = baseline share of households that
    must cross a river to reach the market, bank, government office or health facility (mean of the 4 answers).
    Within each bridge the 2 villages with the highest share are the cut-off side (+1), the others the other side
    (-1), following the study design (2 more-affected + 1 less-affected village per site)."""
    raw = pd.read_csv(SURVEY / "B2P Scale-up Household Survey(3).csv", low_memory=False,
                      usecols=["Unique ID", "Deployment", "Village", *XQ])
    raw = raw[raw.Deployment == "Baseline"].rename(columns={"Unique ID": "uid"}).drop_duplicates("uid")
    raw["cross"] = raw[XQ].apply(lambda c: c.astype(str).str.lower().map({"yes": 1, "no": 0})).mean(axis=1)
    H = pd.read_parquet(DATA / "hh_points.parquet")[["uid", "site", "bridge"]]
    m = raw[["uid", "Village", "cross"]].merge(H, on="uid")
    m["unit"] = m.site + " | " + m.Village.astype(str).str.strip()
    v = m.groupby(["bridge", "unit"]).cross.mean().reset_index()
    v["rank"] = v.groupby("bridge").cross.rank(ascending=False, method="first")
    v["side"] = np.where(v["rank"] <= 2, 1.0, -1.0)
    return m[["uid", "unit"]].merge(v[["unit", "side", "cross"]], on="unit")
