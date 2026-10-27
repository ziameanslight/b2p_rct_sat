"""Satellite-predicted vs observed change of the B2P items that match the DHS items used for training, at every B2P
village, pooled over waves and per wave (figure 21).

  python scripts/b2p_item_scatter.py          (~2 min; needs data/dhs_ridge_preds.parquet from b2p_dhs_ridge.py)

Rows: grid electricity, mobile phones, finished floor (B2P: improved = cement / tile / stone / wood), TV, each against
its own Rwanda-DHS item ridge, and a DHS-matched index of the four (IWI weights applied to observed and to predicted
item changes; phones as any phone). Points: village x wave means (>= 20 households), all arms, both river sides.
Statistics, 95% CI and p from a bridge-cluster bootstrap (2000 draws; whole bridges resampled):
  Pearson r; Spearman rho (ties = midranks); AUC for "any gain" (observed > 0 vs <= 0; 0.5 = chance); Spearman among
  villages that gained; within district: ranks / predictions centred on district x wave means.
Output: plots/21_elec_scatter.png, data/elec_scatter_stats.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, PLOTS  # noqa: E402

WAVE = {"R1": "#2a78d6", "R2": "#eb6834", "R3": "#1baf7a"}
NAME = {"R1": "Round 1", "R2": "Round 2", "R3": "Round 3"}
rng = np.random.default_rng(0)


def boot(q, f, n=2000):
    """Statistic f(frame) with 95% CI and the bootstrap draws, resampling bridges."""
    q = q.reset_index(drop=True)
    g = dict(q.groupby("bridge").indices)
    keys = np.array(list(g))
    est = f(q)
    bs = np.array([f(q.iloc[np.concatenate([g[k] for k in rng.choice(keys, len(keys))])]) for _ in range(n)])
    return est, *np.nanpercentile(bs, [2.5, 97.5]), bs


def pval(bs, null):
    return max(min(1.0, 2 * min(np.nanmean(bs <= null), np.nanmean(bs >= null))), 1 / len(bs))


def spearman(q, x="x", y="y"):
    return q[x].rank().corr(q[y].rank())


def auc(q, x="x", y="y"):
    g = q[y] > 0
    rk = q[x].rank()
    n1, n0 = g.sum(), (~g).sum()
    return (rk[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def centre(q):
    """Within district x wave: predictions and ranks centred on their district x wave means."""
    k = ["district", "r"]
    return q.assign(x=q.x - q.groupby(k).x.transform("mean"),
                    yr=q.groupby(k).y.rank(pct=True) - 0.5, xr=q.groupby(k).x.rank(pct=True) - 0.5)


def stats(q):
    c = centre(q)
    rows = {"Pearson r": (lambda d: d.x.corr(d.y), 0),
            "Spearman rho": (spearman, 0),
            "AUC any gain": (auc, 0.5),
            "Spearman among gainers": (lambda d: spearman(d[d.y > 0]), 0),
            "within district: Spearman": (lambda d: d.xr.corr(d.yr), 0),
            "within district: AUC any gain": (auc, 0.5)}
    out = {}
    for k, (f, null) in rows.items():
        e, lo, hi, bs = boot(c if k.startswith("within") else q, f)
        out[k] = dict(est=e, lo=lo, hi=hi, p=pval(bs, null))
    out["n"] = len(q)
    out["share zero"] = (q.y == 0).mean()
    out["share gained"] = (q.y > 0).mean()
    return out


P = pd.read_parquet(DATA / "dhs_ridge_preds.parquet")
P["district"] = P.site.str.split(" | ", regex=False).str[1]
# DHS-matched index: the four matched items with their IWI weights (Smits & Steendijk), in IWI points. Improved floor
# takes the natural-floor weight (improved = not natural); phones use the DHS any-phone definition.
IW = {"light_grid": 8.056664, "tv": 8.612657, "phone_any": 7.127699, "floor_impr": 7.558471}
PW = {"pred_electricity": "light_grid", "pred_tv": "tv", "pred_phone": "phone_any", "pred_floor_finished": "floor_impr"}
P["obs_index"] = sum(IW[c] * P[c] for c in IW)
P["pred_index"] = sum(IW[o] * P[c] for c, o in PW.items())
# rows: (label, predicted column, observed column, x scale, y scale, y label). DHS item ridges are trained on the share of
# households owning the item; B2P phones are a 0/1/2+ count, so its observed change is in phones per household.
ITEMS = [("Grid electricity", "pred_electricity", "light_grid", 100, 100, "Observed Δ households lit by grid (pp)"),
         ("Mobile phones", "pred_phone", "phones", 100, 1, "Observed Δ mean phones per household (0/1/2+)"),
         ("Finished floor", "pred_floor_finished", "floor_impr", 100, 100,
          "Observed Δ households with improved floor (pp)"),
         ("Television", "pred_tv", "tv", 100, 100, "Observed Δ households with a TV (pp)"),
         ("DHS-matched index", "pred_index", "obs_index", 1, 1,
          "Observed Δ DHS-matched index (IWI points)")]

fig, axes = plt.subplots(len(ITEMS), 4, figsize=(20, 6.4 * len(ITEMS)), sharex="row", sharey="row")
ps = lambda v: "p < 0.001" if v < 0.001 else f"p = {v:.3f}"  # noqa: E731
rows = []
for axr, (item, xc, yc, xs_, ys, ylab) in zip(axes, ITEMS):
    Q = P.dropna(subset=[xc, yc]).assign(x=lambda v: v[xc] * xs_, y=lambda v: v[yc] * ys)
    lim_x = np.percentile(Q.x, [0.5, 99.5]) + np.array([-1, 1])
    for ax, (title, q) in zip(axr, [("Pooled, all waves", Q)] + [(NAME[w], Q[Q.r == w]) for w in WAVE]):
        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
        for w, c in WAVE.items():
            s_ = q[q.r == w]
            ax.scatter(s_.x, s_.y, s=np.clip(s_.n, 20, 60) * 0.6, color=c, alpha=0.55, edgecolor="white",
                       linewidth=0.5, label=NAME[w] if title.startswith("Pooled") else None, zorder=2)
        b = np.polyfit(q.x, q.y, 1)
        xs = np.linspace(*lim_x, 50)
        ax.plot(xs, np.polyval(b, xs), color="#333333", lw=2, zorder=3)
        st = stats(q)
        rows += [dict(item=item, sample=title, statistic=k, **v) for k, v in st.items() if isinstance(v, dict)]
        txt = "\n".join(f"{k} = {st[k]['est']:.2f} [{st[k]['lo']:.2f}, {st[k]['hi']:.2f}], {ps(st[k]['p'])}"
                         for k in ["Pearson r", "Spearman rho", "AUC any gain", "Spearman among gainers",
                                   "within district: Spearman", "within district: AUC any gain"])
        txt += f"\nn = {st['n']} villages; {st['share zero']:.0%} no change, {st['share gained']:.0%} gained"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="#dddddd", alpha=0.9))
        ax.set_title(f"{item}: {title}", fontsize=12)
        ax.set_xlabel((f"Predicted Δ {item.lower()} share since baseline (pp)" if xs_ == 100 else
                       "Predicted Δ DHS-matched index (IWI points)") + "\nRwanda DHS-trained ridge, no B2P training")
        ax.grid(color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axr[0].set_ylabel(ylab + " since baseline")
    axr[0].legend(loc="lower right", fontsize=9, frameon=False)
    axr[0].set_xlim(*lim_x)
fig.suptitle("B2P asset items with a direct or partial match to the Rwanda DHS items used for training: "
             "satellite-predicted vs observed change\n"
             "Direct: grid electricity (DHS electricity), TV.  Partial: mobile phones (B2P count 0/1/2+ vs DHS any phone), "
             "floor (B2P improved incl. wood/stone vs DHS finished).\n"
             "Bottom row: index of the four, IWI weights (grid 8.06, TV 8.61, any phone 7.13, floor 7.56) "
             "applied to observed and to predicted item changes.\n"
             "B2P villages, all arms, both river sides; 95% CI and p by bridge-cluster bootstrap",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
T = pd.DataFrame(rows)
T.to_csv(DATA / "elec_scatter_stats.csv", index=False)
print(T.assign(ci=T.apply(lambda r: f"{r.est:.2f} [{r.lo:.2f},{r.hi:.2f}] p={r.p:.3f}", axis=1))
      .pivot(index=["item", "statistic"], columns="sample", values="ci").to_string())
out = PLOTS / "21_elec_scatter.png"
fig.savefig(out, dpi=130)
print(out)
