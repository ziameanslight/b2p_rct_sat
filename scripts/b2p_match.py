"""Match B2P survey households to RCT bridges and fix one GPS point per household.

Input : b2psurvey data/rct_bridges.csv, b2psurvey data/B2P Scale-up Household Survey(3).csv
Output: data/site_bridge_match.csv  (survey site -> bridge, match type, distance site median GPS -> bridge)
        data/hh_points.parquet      (uid, site, bridge, arm, close date, lat/lon = median of the household's
                                     round GPS points after dropping fixes > 3 km from the site median)

Rules: a survey site is (Site Name, Site District) - several names cover two places in different districts (Gasharu,
Kabuga, Kagarama, Munanira, Nyarusange). Names are compared after lower-casing and dropping digits, spaces and
punctuation; a name match needs similarity >= 0.85. Among name matches the bridge in the same district (Level 2
Government) wins, else the nearest. Name matches whose bridge GPS is > 10 km from the households are kept but flagged
(the bridge coordinates are wrong there; imagery uses household GPS). One proximity match is added by hand (Remera,
Huye -> Bukomeye - Taba, same district, 1.8 km, no other site for that bridge); other sites with no name match are
left unmatched.
Arms: built = Intervention arm and Stage Complete (close date = completion); unbuilt = Intervention arm, not complete
(identified / cancelled / prospecting / under construction); ltc = Long Term Control.
"""
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "b2psurvey data"
OUT = ROOT / "data"
ROUNDS = ["Baseline", "Follow-up round 1", "Follow-up round 2", "Follow-up round 3"]
LAT, LON = "Latitude of Record GPS location", "Longitude of Record GPS location"


def norm(s):
    return re.sub(r"[^a-z]", "", re.sub(r"\d+", "", s.lower()))


def km(lat0, lon0, lat1, lon1):
    lat0, lon0, lat1, lon1 = map(np.radians, (lat0, lon0, lat1, lon1))
    return 6371 * np.hypot(lat1 - lat0, (lon1 - lon0) * np.cos((lat0 + lat1) / 2))


def bridges():
    b = pd.read_csv(RAW / "rct_bridges.csv")
    arm = np.where(b["Research Initiative"].str.contains("Long Term Control"), "ltc",
                   np.where(b["Stage"].eq("Complete"), "built", "unbuilt"))
    return pd.DataFrame({"bridge": b["Bridge Name"].str.strip(), "project": b["Project Code"], "arm": arm,
                         "stage": b["Stage"], "b_lat": b["GPS (Latitude)"], "b_lon": b["GPS (Longitude)"],
                         "close": pd.to_datetime(b["Close Date"], errors="coerce").where(arm == "built"),
                         "district": b["Level 2 Government"].str.strip(),
                         "ground": pd.to_datetime(b["Ground Breaking Date"], errors="coerce")})


def main():
    B = bridges()
    d = pd.read_csv(RAW / "B2P Scale-up Household Survey(3).csv", low_memory=False,
                    usecols=["Unique ID", "Deployment", "Site Name", "Site District", LAT, LON])
    d = d[d.Deployment.isin(ROUNDS)].rename(columns={"Unique ID": "uid", "Site Name": "site", LAT: "lat", LON: "lon"})
    d["lat"], d["lon"] = pd.to_numeric(d.lat, errors="coerce"), pd.to_numeric(d.lon, errors="coerce")
    d.loc[~d.lat.between(-3, -1) | ~d.lon.between(28.8, 31), ["lat", "lon"]] = np.nan   # outside Rwanda
    # site district: a name's district only splits it if that district has >= 20 baseline households
    d["site"] = d.site.str.strip()
    nd = d[d.Deployment == "Baseline"].groupby(["site", "Site District"]).size()
    big = nd[nd >= 20].reset_index().groupby("site")["Site District"].apply(set)
    modal = nd.reset_index().sort_values(0).groupby("site")["Site District"].last()
    dist_key = [di if di in big.get(si, set()) else modal.get(si, di) for si, di in zip(d.site, d["Site District"])]
    d["site"] = d.site + " | " + pd.Series(dist_key, index=d.index).astype(str)
    S = d[d.Deployment == "Baseline"].groupby("site").agg(s_lat=("lat", "median"), s_lon=("lon", "median"),
                                                          district=("Site District", "first"))
    # site -> bridge by name, then district, then distance
    cand = {}
    for site, r in S.iterrows():
        name = site.split(" | ")[0]
        alt = lambda n: [norm(n), norm(re.sub(r"\(.*?\)", "", n)), norm(re.sub(r".*\((.*?)\).*", r"\1", n))]
        sim = np.array([max(difflib.SequenceMatcher(None, norm(name), a).ratio() for a in alt(n)) for n in B.bridge])
        dist = km(r.s_lat, r.s_lon, B.b_lat.to_numpy(), B.b_lon.to_numpy())
        same_d = B.district.to_numpy() == r.district
        ok = np.flatnonzero((sim >= 0.85) & (same_d | (dist <= 10)))    # same district, or close if not
        ok = ok[sim[ok] == sim[ok].max()] if len(ok) else ok
        same = ok[same_d[ok]]
        pick = same if len(same) else ok
        cand[site] = [int(pick[np.argmin(dist[pick])])] if len(pick) else []
    cand["Remera | Huye"] = [int(np.flatnonzero(B.bridge == "Bukomeye - Taba")[0])]

    # household point: median of its round fixes within 3 km of the site median
    d = d.join(S[["s_lat", "s_lon"]], on="site")
    far = km(d.s_lat, d.s_lon, d.lat, d.lon) > 3
    d.loc[far, ["lat", "lon"]] = np.nan
    base_site = d[d.Deployment == "Baseline"].drop_duplicates("uid").set_index("uid").site
    H = d.groupby("uid").agg(lat=("lat", "median"), lon=("lon", "median"), n_fix=("lat", "count"))
    H["site"] = base_site.reindex(H.index).fillna(d.groupby("uid").site.first())
    H = H[H.lat.notna()]
    bi = np.full(len(H), -1)
    for site, idx in H.groupby("site").indices.items():
        c = cand.get(site, [])
        if not c:
            continue
        g = H.iloc[idx]
        dd = np.stack([km(g.lat.to_numpy(), g.lon.to_numpy(), B.b_lat[k], B.b_lon[k]) for k in c], 1)
        bi[idx] = np.array(c)[dd.argmin(1)]
    H["bi"] = bi
    Bm = B.reset_index().rename(columns={"index": "bi"})
    H = H.reset_index().merge(Bm, on="bi", how="left")
    H.loc[H.bi < 0, ["bridge", "arm"]] = [None, "unmatched"]
    H["d_bridge_km"] = km(H.lat, H.lon, H.b_lat, H.b_lon)

    # site table
    rows = []
    for site, r in S.iterrows():
        c = cand[site]
        names = "; ".join(B.bridge[k] for k in c)
        name = site.split(" | ")[0]
        typ = "none" if not c else ("proximity" if site == "Remera | Huye" else
                                    "exact" if norm(B.bridge[c[0]]) == norm(name) else "fuzzy")
        if c and km(r.s_lat, r.s_lon, B.b_lat[c[0]], B.b_lon[c[0]]) > 10:
            typ += ", bridge GPS > 10 km off"
        n = (H.site == site).sum()
        rows.append(dict(site=site, district=r.district, bridges=names, match=typ, n_hh=n,
                         arm="; ".join(sorted({B.arm[k] for k in c})) or "unmatched",
                         km_site_to_bridge=np.round(min(km(r.s_lat, r.s_lon, B.b_lat[k], B.b_lon[k]) for k in c), 2)
                         if c else np.nan))
    M = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    M.to_csv(OUT / "site_bridge_match.csv", index=False)
    H.drop(columns=["bi"]).to_parquet(OUT / "hh_points.parquet")
    used = set(H.bridge.dropna())
    print(M.match.value_counts().to_string())
    print("unmatched sites:", M[M.match == "none"].site.tolist())
    print("bridges with no survey site:", sorted(set(B.bridge) - used))
    print(H.groupby("arm").agg(hh=("uid", "size"), bridges=("bridge", "nunique")))
    print("household distance to its bridge, km: median %.2f, p95 %.2f" % (H.d_bridge_km.median(),
                                                                            H.d_bridge_km.quantile(.95)))


if __name__ == "__main__":
    main()
