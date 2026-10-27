"""B2P Rwanda household panel: baseline -> follow-up round 3 change in assets vs consumption/income proxies.

  python scripts/b2p_change.py

Input : b2psurvey data/B2P Scale-up Household Survey(3).csv (mWater export, 1 row per household x round)
Output: data/hh_rounds.csv (household x round), data/hh_change.csv (baseline -> round 3 per household),
        data/site_change.csv (site means), data/component_transitions.csv (asset-index components)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "b2psurvey data" / "B2P Scale-up Household Survey(3).csv"
OUT = ROOT / "data"
ROUNDS = ["Baseline", "Follow-up round 1", "Follow-up round 2", "Follow-up round 3"]
B, F = "Baseline", "Follow-up round 3"

Q = {
    "wall": "What is the main construction material of the exterior walls?",
    "roof": "What is the main material used for roofing the main dwelling?",
    "floor": "Which type of flooring material does the home have?",
    "iwall": "Which type of wall material does the interior of the home have?",
    "beds": "How many beds are in your home?",
    "light": "What is the main source of lighting in the residence?",
    "phones": "How many mobile phones do members of your household own?",
    "tv": "Do you own any of the following?  includes all of television",
    "dish": "Do you own any of the following?  includes all of satellite dish",
    "gas": "Do you own any of the following?  includes all of gas stove",
    "animals": "Do you own farm animals?",
    "n_animals": "Farm animals > Total How many do you own? ",
    "own_land": "Do you own or rent any land besides what your house is built upon? includes all of yes, I own more land that my family uses (farming, grazing, fallow)",
    "rent_out": "Do you own or rent any land besides what your house is built upon? includes all of yes, I rent more land to someone",
    "rent_in": "Do you own or rent any land besides what your house is built upon? includes all of yes, I rent more land from someone",
    "land_own_m": "How much land\xa0owned\xa0by your household are you farming, grazing, and fallow land? (magnitude) ",
    "land_own_u": "How much land\xa0owned\xa0by your household are you farming, grazing, and fallow land? (units)",
    "land_cult_m": "How much land do you use to cultivate crops? (magnitude)",
    "land_cult_u": "How much land do you use to cultivate crops? (units)",
    "debt": "Does anyone in your household currently have any debts? ",
    "food": "In total, how much did your household spend on food purchased in markets in the past week?",
    "fuel": "In total, how much did your household spend on fuel for cooking in the past week?",
    "items": "In total, how much did your household spend on household items in markets in the past week, including soap, detergent, and toilet paper?",
    "airtime": "In total, how much did your household spend on airtime in the past week?",
    "fert": "In the last season, how much (francs) in total did you spend on fertilizer?",
    "pest": "In the last season, how much (francs) in total did you spend on pesticide?",
    "seed": "In the last season, how much (francs) in total did you spend on seed?",
    "sold_animal": "Farm animals > Latest Have you sold any in the last 6 months?",
    "sale_price": "Farm animals > Total How many francs did you receive per animal?",
    "nonfarm": "In the last 12 months, how many household members ran non-farm business for cash or profit for themselves?",
    "agwage": "In the last 12 months, how many household members carried out any agricultural activity for salary, wages, or in-kind compensation?",
    "sell_comm": "How do you sell your crops? includes all of Sell them within your community (including to middlemen)",
    "sell_out": "How do you sell your crops? includes all of Take them to outside markets",
    "n_crops": "Which of the following crops do you grow?",
    "hhsize": "How many people live in your household, including yourself?",
    "fin_sat": "How satisfied are you with the financial situation of your household?",
}
DEBT_AMT = [c for c in pd.read_csv(SRC, nrows=0).columns if c.startswith("In total how many (francs) do members of your household owe")]
FIES = [c for c in pd.read_csv(SRC, nrows=0).columns if c.startswith("In the past four weeks,")]

# asset components (binary/ordinal) used for the wealth index.
# light_modern is left out (it is grid|solar|generator|biogas, so it would double-count grid and solar);
# own_land is left out (76%->93% jump at R1 is a questionnaire artefact). Both are still reported as transitions.
ASSETS = ["floor_impr", "roof_metal", "wall_cement", "iwall_cement", "beds", "light_grid", "light_solar",
          "phones", "tv", "dish", "gas", "animals", "rent_out"]
EXTRA = ["light_modern", "own_land", "rent_in", "debt_any"]  # transitions reported, not in index
LABEL = {
    "floor_impr": "Improved floor (cement/tile/stone/wood)", "roof_metal": "Metal/concrete roof (vs clay tile)",
    "wall_cement": "Exterior wall cement-plastered / fired brick / block", "iwall_cement": "Interior wall cement-plastered / brick",
    "beds": "Beds (0-3+)", "light_grid": "Lighting: grid electricity", "light_solar": "Lighting: solar panel",
    "light_modern": "Lighting: grid/solar/generator/biogas", "phones": "Mobile phones (0/1/2+)", "tv": "Television",
    "dish": "Satellite dish", "gas": "Gas stove", "animals": "Owns any farm animals", "own_land": "Owns farm land",
    "rent_out": "Rents land out",
}


def to_m2(mag, unit):
    m = pd.to_numeric(mag, errors="coerce")
    # 'hectares' entries > 20 are almost always square metres typed under the wrong unit
    ha = (unit == "hectares") & (m <= 20)
    return np.where(ha, m * 1e4, m)


def wins(s, p=0.99):
    return s.clip(upper=s.quantile(p))


def clean(df):
    o = pd.DataFrame({"uid": df["Unique ID"], "round": df["Deployment"], "site": df["Site Name"],
                      "district": df["Site District"], "date": pd.to_datetime(df["Submitted On"], errors="coerce")})
    w, iw, fl, rf, lt = (df[Q[k]] for k in ["wall", "iwall", "floor", "roof", "light"])
    good_wall = lambda s: s.str.contains("covered with cement|Cement blocks|Oven-fired|Stones", na=False)
    o["answered"] = rf.notna()
    o["floor_impr"] = fl.isin(["Cement", "Ceramic tiles", "Stone/brick", "Wood planks"])
    o["roof_metal"] = rf.isin(["Metal", "Concrete"])
    o["wall_cement"] = good_wall(w)
    o["iwall_cement"] = good_wall(iw)
    o["beds"] = df[Q["beds"]].map({"0": 0, "1": 1, "2": 2, "3 or more": 3})
    o["light_grid"] = lt.eq("Electric grid (Rwanda Energy Group)")
    o["light_solar"] = lt.eq("Solar Panel")
    o["light_modern"] = lt.isin(["Electric grid (Rwanda Energy Group)", "Solar Panel", "Generator", "Biogas"])
    o["phones"] = df[Q["phones"]].map({"0": 0, "1": 1, "2 or more": 2})
    for k in ["tv", "dish", "gas", "own_land", "rent_out", "rent_in"]:
        o[k] = df[Q[k]].astype("boolean").fillna(False)
    o["animals"] = df[Q["animals"]].eq("Yes")
    o["n_animals"] = np.where(o["animals"], pd.to_numeric(df[Q["n_animals"]], errors="coerce"), 0)
    o["land_own_m2"] = np.where(o["own_land"], to_m2(df[Q["land_own_m"]], df[Q["land_own_u"]]), 0)
    o["land_cult_m2"] = to_m2(df[Q["land_cult_m"]], df[Q["land_cult_u"]])
    o["debt_any"] = df[Q["debt"]].eq("yes")
    o["debt_amt"] = df[DEBT_AMT[:-1]].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)  # last = owed TO hh
    for k in ["food", "fuel", "items", "airtime"]:
        o[k] = pd.to_numeric(df[Q[k]], errors="coerce")
    o["cons_week"] = o[["food", "fuel", "items", "airtime"]].sum(axis=1, min_count=1)
    o["inputs_season"] = df[[Q["fert"], Q["pest"], Q["seed"]]].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    o["sold_animal"] = df[Q["sold_animal"]].eq("Yes")
    o["animal_sale_px"] = pd.to_numeric(df[Q["sale_price"]], errors="coerce").fillna(0)
    o["nonfarm_biz"] = pd.to_numeric(df[Q["nonfarm"]], errors="coerce").fillna(0) > 0
    o["ag_wage"] = pd.to_numeric(df[Q["agwage"]], errors="coerce").fillna(0) > 0
    o["sells_crops"] = df[Q["sell_comm"]].astype("boolean").fillna(False) | df[Q["sell_out"]].astype("boolean").fillna(False)
    o["sells_outside"] = df[Q["sell_out"]].astype("boolean").fillna(False)
    o["n_crops"] = pd.to_numeric(df[Q["n_crops"]], errors="coerce")
    o["hhsize"] = pd.to_numeric(df[Q["hhsize"]], errors="coerce")
    o["fin_sat"] = pd.to_numeric(df[Q["fin_sat"]], errors="coerce")
    o["fies"] = df[FIES].eq("Yes").sum(axis=1).where(df[FIES[0]].notna())
    for c in ["n_animals", "land_own_m2", "land_cult_m2", "debt_amt", "cons_week", "inputs_season", "animal_sale_px"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    return o


def pca_index(base, data):
    X = base[ASSETS].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).fillna(0).values  # missing item -> baseline mean
    vals, vecs = np.linalg.eigh(np.cov(Z.T))
    w = vecs[:, -1]
    w = w if w.sum() > 0 else -w
    share = vals[-1] / vals.sum()
    idx = (((data[ASSETS].astype(float) - mu) / sd).fillna(0).values @ w)
    s0 = (Z @ w).std()
    return idx / s0, pd.Series(w, ASSETS), share  # SD units of baseline distribution


def main():
    raw = pd.read_csv(SRC, low_memory=False)
    raw = raw[raw["Deployment"].isin(ROUNDS)]
    d = clean(raw)
    d = d[d["answered"]].sort_values("date").drop_duplicates(["uid", "round"], keep="last")
    for c in ["n_animals", "land_own_m2", "land_cult_m2", "debt_amt", "cons_week", "inputs_season", "food"]:
        d[c] = wins(d[c])
    base = d[d["round"] == B]
    d["asset_idx"], weights, share = pca_index(base, d)
    d["asset_count"] = d[["floor_impr", "roof_metal", "wall_cement", "light_modern", "tv", "dish", "gas",
                          "animals"]].astype(int).sum(axis=1) + (d["phones"] > 0) + (d["beds"] > 0)

    # ---- household panel: baseline vs round 3
    p = d[d["round"].isin([B, F])].pivot(index="uid", columns="round")
    p = p[p[("answered", B)].notna() & p[("answered", F)].notna()]
    hh = pd.DataFrame(index=p.index)
    hh["site"], hh["district"] = p[("site", B)], p[("district", B)]
    num = ["asset_idx", "asset_count", "beds", "phones", "n_animals", "land_own_m2", "land_cult_m2", "debt_amt",
           "cons_week", "food", "fuel", "items", "airtime", "inputs_season", "animal_sale_px", "n_crops", "hhsize",
           "fin_sat", "fies"]
    binv = ["floor_impr", "roof_metal", "wall_cement", "iwall_cement", "light_grid", "light_solar", "light_modern",
            "tv", "dish", "gas", "animals", "own_land", "rent_out", "rent_in", "debt_any", "sold_animal",
            "nonfarm_biz", "ag_wage", "sells_crops", "sells_outside"]
    for c in num + binv:
        b0 = pd.to_numeric(p[(c, B)], errors="coerce").astype(float)
        f0 = pd.to_numeric(p[(c, F)], errors="coerce").astype(float)
        hh[c + "_b"], hh[c + "_f"], hh["d_" + c] = b0, f0, f0 - b0
    for c in ["cons_week", "food", "inputs_season", "land_own_m2", "n_animals", "debt_amt"]:
        hh["dlog_" + c] = np.log1p(hh[c + "_f"]) - np.log1p(hh[c + "_b"])
    hs_b, hs_f = hh["hhsize_b"].where(hh["hhsize_b"] > 0), hh["hhsize_f"].where(hh["hhsize_f"] > 0)
    hh["dlog_cons_pc"] = hh["dlog_cons_week"] - (np.log(hs_f) - np.log(hs_b))
    OUT.mkdir(parents=True, exist_ok=True)
    hh.to_csv(OUT / "hh_change.csv")

    # ---- component transitions
    rows = []
    for c in ASSETS + EXTRA:
        b0, f0 = hh[c + "_b"], hh[c + "_f"]
        if c in ("beds", "phones"):
            rows.append([c, b0.mean(), f0.mean(), (f0 > b0).mean(), (f0 < b0).mean(), (f0 == b0).mean(),
                         (f0 - b0).mean()])
        else:
            rows.append([c, b0.mean(), f0.mean(), ((b0 == 0) & (f0 == 1)).mean(), ((b0 == 1) & (f0 == 0)).mean(),
                         (b0 == f0).mean(), (f0 - b0).mean()])
    tr = pd.DataFrame(rows, columns=["component", "base_mean", "r3_mean", "gain", "loss", "same", "net"])
    tr["gain_loss_ratio"] = tr["gain"] / tr["loss"].replace(0, np.nan)
    tr["pca_weight"] = tr["component"].map(weights)
    # contribution of each component to mean change in the PCA index
    sd = base[ASSETS].astype(float).std()
    s0 = ((base[ASSETS].astype(float) - base[ASSETS].astype(float).mean()) / sd).fillna(0).values @ weights.values
    tr["idx_contrib"] = tr.apply(lambda r: r.net / sd[r.component] * weights[r.component] / s0.std()
                                 if r.component in ASSETS else np.nan, axis=1)
    tr["label"] = tr["component"].map(LABEL).fillna(tr["component"])
    tr.to_csv(OUT / "component_transitions.csv", index=False)
    d.drop(columns="answered").to_csv(OUT / "hh_rounds.csv", index=False)

    # ---- site means of the changes (sites with >= 30 panel households)
    site = hh.groupby("site").agg(n=("d_asset_idx", "size"), d_asset_idx=("d_asset_idx", "mean"),
                                  dlog_cons_week=("dlog_cons_week", "mean"))
    site[site["n"] >= 30].to_csv(OUT / "site_change.csv")


if __name__ == "__main__":
    main()
