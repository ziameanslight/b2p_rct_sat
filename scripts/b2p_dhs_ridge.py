"""Rwanda-DHS-only change ridge (village-wealth-change hybrid ridge inputs), predicting change at every B2P village.

  python scripts/b2p_dhs_ridge.py          (seconds; needs data/rw_pair_feats.npz, data/hh_year_msfeats.npz)

Training: the 52 Rwanda DHS matched village pairs of village-wealth-change (2019->2023, 2019->2025, 2023->2025), one
row per place of a pair with features in both years: x = [Δemb (64), Δsat (192), gap in years] -> observed change of
IWI and of the DHS items that the B2P survey also asks: electricity, TV, phone, finished floor. Standardised inputs;
ridge penalty by 5-fold cross-validation grouped by pair (both places of a pair in the same fold). No B2P row is
used in training, so every B2P village is out of sample.
B2P: each household's own Δ features (baseline imagery year -> wave imagery year, b2p_common.imagery_year) -> its
predicted change; observed change of the matched B2P items (grid lighting, TV, mobile phones as count and as any
phone, improved floor), the asset index and log consumption. Village x wave means (>= 20 households), all arms, with
the river side of b2p_common.village_sides.
Output: data/dhs_ridge_preds.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402
from b2p_common import WAVES, load, village_sides  # noqa: E402
from common import item_shares, locations  # noqa: E402
from pairs import pair_tables  # noqa: E402

TGT = ["iwi", "electricity", "tv", "phone", "floor_finished"]                   # DHS pair targets
OBS = ["light_grid", "tv", "phones", "phone_any", "floor_impr", "asset_idx", "log_cons"]   # B2P observed change


def ridge_grouped(X, y, groups):
    g = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-1, 5, 13)},
                     cv=GroupKFold(5), scoring="neg_mean_squared_error")
    g.fit(X, y, groups=groups)
    return g.best_estimator_, g.best_params_["ridge__alpha"]


def training_rows():
    L = locations()
    T, _ = pair_tables(L, item_shares(), targets=TGT)
    T = T[(T.cc == "RW") & (T.src == "DHS")].reset_index(drop=True)
    z = np.load(DATA / "rw_pair_feats.npz")
    F = {(int(c), int(y)): (e, s) for c, y, e, s in zip(z["cid"], z["year"], z["emb"], z["sat"])}
    g, X, Y = [], [], []
    for n, r in enumerate(T.itertuples()):
        for p in (int(r.i), int(r.j)):
            a, b = F.get((p, int(r.y1))), F.get((p, int(r.y2)))
            if a is None or b is None:
                continue
            g.append(n)
            X.append(np.r_[b[0] - a[0], b[1] - a[1], r.gap])
            Y.append([getattr(r, f"obs_{t}") for t in TGT])
    print(f"{len(T)} Rwanda DHS pairs, {len(set(g))} with features, {len(g)} place rows")
    return np.array(g), np.array(X), np.array(Y)


def b2p_rows():
    d = load()
    d["phone_any"] = (d.phones > 0).astype(float)
    V = village_sides().set_index("uid")
    z = np.load(DATA / "hh_year_msfeats.npz")
    F = {(u, int(y)): (e, s) for u, y, e, s in zip(z["uid"], z["year"], z["emb"], z["sat"])}
    base = d[d.r == "B"].set_index("uid")
    rows = []
    for k in WAVES:
        w = d[d.r == k].set_index("uid")
        for uid in w.index.intersection(base.index).intersection(V.index):
            b, f = base.loc[uid], w.loc[uid]
            fb, ff = F.get((uid, int(b.year))), F.get((uid, int(f.year)))
            if fb is None or ff is None or any(np.isnan(v).any() for v in (*fb, *ff)):
                continue
            rows.append(dict(uid=uid, site=V.unit[uid], side=V.side[uid], bridge=b.bridge, arm=b.arm, r=k,
                             x=np.r_[ff[0] - fb[0], ff[1] - fb[1], f.year - b.year],
                             **{o: float(f[o]) - float(b[o]) for o in OBS}))
    return pd.DataFrame(rows)


def main():
    g, X, Y = training_rows()
    H = b2p_rows()
    XB = np.stack(H.x.to_numpy())
    for n, t in enumerate(TGT):
        m, alpha = ridge_grouped(X, Y[:, n], g)
        H[f"pred_{t}"] = m.predict(XB)
        print(f"{t}: alpha {alpha:g}")
    V = H.groupby(["site", "r"]).agg(side=("side", "first"), bridge=("bridge", "first"), arm=("arm", "first"),
                                     n=("uid", "size"), **{c: (c, "mean") for c in OBS},
                                     **{f"pred_{t}": (f"pred_{t}", "mean") for t in TGT}).reset_index()
    V = V[V.n >= 20].reset_index(drop=True)
    V.to_parquet(DATA / "dhs_ridge_preds.parquet")
    print(f"B2P: {len(H)} household-waves -> {len(V)} village-waves ({V.site.nunique()} villages, "
          f"{V.bridge.nunique()} bridges)")


if __name__ == "__main__":
    main()
