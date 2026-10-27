"""Figures for the B2P baseline -> round 3 change analysis. Run after b2p_change.py.

Reads data/{hh_rounds,hh_change,site_change,component_transitions}.csv, writes plots/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, PLOTS = ROOT / "data", ROOT / "plots"
ROUNDS = ["Baseline", "Follow-up round 1", "Follow-up round 2", "Follow-up round 3"]
RLAB = ["Baseline\n2020/21", "R1\n2021/22", "R2\n2022/23", "R3\n2023/24"]

# reference palette (dataviz skill, light mode)
SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#c3c2b7"
CPI_LO, CPI_HI = 0.30, 0.45  # approx. cumulative Rwanda CPI 2020/21 -> 2023/24

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def trajectory(d):
    """Share of households with each asset, by round (small multiples)."""
    items = [("light_modern", "Modern lighting (any)"), ("light_solar", "Solar lighting"),
             ("light_grid", "Grid electricity"), ("floor_impr", "Improved floor"),
             ("roof_metal", "Metal/concrete roof"), ("wall_cement", "Cement-plastered ext. wall"),
             ("tv", "Television"), ("animals", "Owns livestock"), ("own_land", "Owns farm land*")]
    g = d.groupby("round")[[k for k, _ in items]].mean().reindex(ROUNDS) * 100
    fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
    for ax, (k, lab) in zip(axs.flat, items):
        y = g[k].values
        ax.plot(range(4), y, color=BLUE, marker="o", ms=5)
        ax.set_title(lab, fontsize=10)
        ax.set_ylim(0, max(100 if y.max() > 50 else y.max() * 1.6, 5))
        ax.annotate(f"{y[0]:.0f}%", (0, y[0]), textcoords="offset points", xytext=(0, 8), ha="center", color=INK2, fontsize=9)
        ax.annotate(f"{y[-1]:.0f}%", (3, y[-1]), textcoords="offset points", xytext=(0, 8), ha="center", color=INK, fontsize=9)
        ax.set_xticks(range(4), RLAB, fontsize=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.suptitle("Share of households owning each asset, by survey round (all respondents in each round)", x=0.01,
                 ha="left", fontweight="bold")
    fig.text(0.01, -0.01, "*Land jump at R1 coincides with a fall in 'no land' answers (13%→3%) — likely a questionnaire "
             "artefact; excluded from the asset index.", color=INK2, fontsize=8)
    fig.tight_layout()
    save(fig, "01_asset_shares_by_round.png")


def gains_losses(tr):
    """Per-component share of panel households gaining vs losing (butterfly)."""
    # assets only: no debt/renting, no land (questionnaire artefact), no 'any modern lighting' (= grid + solar + ...)
    tr = tr[~tr["component"].isin(["rent_in", "rent_out", "debt_any", "own_land", "light_modern"])].copy()
    tr["label"] = tr["label"].str.replace(" (vs clay tile)", "").str.replace(" (cement/tile/stone/wood)", "")
    tr = tr.sort_values("gain")
    y = np.arange(len(tr))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y, tr["gain"] * 100, color=BLUE, height=0.7, label="Gained (0→1 or count up)")
    ax.barh(y, -tr["loss"] * 100, color=RED, height=0.7, label="Lost (1→0 or count down)")
    for yi, (g, l, r) in enumerate(zip(tr["gain"], tr["loss"], tr["gain_loss_ratio"])):
        ax.text(g * 100 + 0.6, yi, f"{g*100:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(-l * 100 - 0.6, yi, f"{l*100:.0f}%", va="center", ha="right", fontsize=8, color=INK2)
        ax.text(44, yi, f"{r:.1f}×", va="center", ha="right", fontsize=8, color=INK2)
    ax.text(44, len(tr) - 0.3, "gain:loss", ha="right", fontsize=8, color=INK2, fontweight="bold")
    ax.set_yticks(y, tr["label"])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(-24, 45)
    ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of panel households (n = 14,452), baseline → round 3")
    ax.set_title("Which assets changed: gains vs losses per household")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "02_component_gains_losses.png")


def contribution(tr):
    """Contribution of each component to the mean asset-index change."""
    t = tr.dropna(subset=["idx_contrib"]).sort_values("idx_contrib")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t["label"], t["idx_contrib"], color=BLUE, height=0.7)
    for i, v in enumerate(t["idx_contrib"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Contribution to mean Δ asset index (baseline SD units; total = {t['idx_contrib'].sum():.2f})")
    ax.set_title("What drove the asset-index rise")
    save(fig, "03_index_contribution.png")


def hist_delta(hh):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = hh["d_asset_idx"].clip(-2.5, 3)
    bins = np.round(np.arange(-2.5, 3.05, 0.1), 2)
    # stacked so the exactly-unchanged households (all land in the [0, 0.1) bin) stay visible
    ax.hist([x[x < 0], x[x == 0], x[x > 0]], bins=bins, color=[RED, NEUTRAL, BLUE], edgecolor=SURF, lw=0.5,
            stacked=True)
    med = hh["d_asset_idx"].median()
    ax.axvline(med, color=INK, lw=1, ls="--")
    ax.text(med + 0.05, ax.get_ylim()[1] * 0.92, f"median {med:+.2f}", fontsize=9)
    ax.text(-2.4, ax.get_ylim()[1] * 0.8, f"{(hh['d_asset_idx'] < 0).mean():.0%} down", color=RED, fontsize=9)
    ax.text(-2.4, ax.get_ylim()[1] * 0.7, f"{(hh['d_asset_idx'] == 0).mean():.0%} unchanged", color=INK2, fontsize=9)
    ax.text(1.6, ax.get_ylim()[1] * 0.8, f"{(hh['d_asset_idx'] > 0).mean():.0%} up", color=BLUE, fontsize=9)
    ax.set_xlabel("Δ asset index per household, baseline → round 3 (baseline SD units)")
    ax.set_ylabel("Households")
    ax.set_title("Distribution of household asset change")
    save(fig, "04_hh_asset_change_hist.png")


def money(hh):
    """Nominal median growth of money variables vs CPI band."""
    items = [("cons_week", "Weekly cash consumption"), ("food", "Weekly food purchases"),
             ("items", "Weekly household items"), ("inputs_season", "Farm inputs / season"),
             ("animal_sale_px", "Livestock sale price (sellers)")]
    rows = []
    for k, lab in items:
        b, f = hh[k + "_b"], hh[k + "_f"]
        if k == "animal_sale_px":
            b, f = b.where(b > 0), f.where(f > 0)
        rows.append((lab, f.median() / b.median() - 1))
    r = pd.DataFrame(rows, columns=["item", "g"]).sort_values("g")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(CPI_LO * 100, CPI_HI * 100, color=NEUTRAL, alpha=0.35, lw=0)
    ax.barh(r["item"], r["g"] * 100, color=ORANGE, height=0.6)
    for i, v in enumerate(r["g"]):
        ax.text(v * 100 + 2, i, f"+{v*100:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 175)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Nominal change in household median, baseline → round 3\n(grey band = approx. cumulative CPI 2020/21→2023/24, 30–45%)")
    ax.set_title("Money measures: nominal growth vs inflation")
    save(fig, "05_money_growth_vs_cpi.png")


def site_rho_checks(hh, site, rho, reps=2000, seed=0, x="d_asset_idx"):
    """Robustness of the site-level Spearman rho: p, bootstrap CI, within-district permutation, split-half."""
    rng = np.random.default_rng(seed)
    s = site[["site", x, "dlog_cons_week"]].rename(columns={x: "d_asset_idx"})
    s["district"] = s["site"].map(hh.groupby("site")["district"].first())
    rx, ry = s["d_asset_idx"].rank(), s["dlog_cons_week"].rank()
    p = stats.spearmanr(rx, ry).pvalue
    idx = rng.integers(0, len(s), (reps, len(s)))
    ci = np.percentile([np.corrcoef(rx.values[i], ry.values[i])[0, 1] for i in idx], [2.5, 97.5])
    groups = [np.flatnonzero(s["district"].values == g) for g in s["district"].unique()]
    perm = np.empty(reps)
    for k in range(reps):
        yy = ry.values.copy()
        for g in groups:
            yy[g] = yy[rng.permutation(g)]
        perm[k] = np.corrcoef(rx, yy)[0, 1]
    p_perm = max((np.abs(perm) >= abs(rho)).mean(), 1 / reps)
    dm = lambda r: r - r.groupby(s["district"]).transform("mean")
    rho_w = np.corrcoef(dm(rx), dm(ry))[0, 1]
    # asset change from a random half of each site's households, consumption change from the other half;
    # repeated over 200 random splits since a single split is noisy
    sh = []
    for _ in range(200):
        half = rng.integers(0, 2, len(hh)).astype(bool)
        a = hh[half].groupby("site")[x].mean().reindex(s["site"])
        c = hh[~half].groupby("site")["dlog_cons_week"].mean().reindex(s["site"])
        sh.append(stats.spearmanr(a, c, nan_policy="omit").statistic)
    sh_lo, sh_med, sh_hi = np.percentile(sh, [5, 50, 95])
    return (f"Site ρ robustness:  p = {p:.1g};  bootstrap 95% CI {ci[0]:.2f}–{ci[1]:.2f};  "
            f"within districts ({len(groups)}) ρ = {rho_w:.2f}, permutation p {'<' if p_perm <= 1 / reps else '='} "
            f"{p_perm:.3f}\n"
            f"split-half (asset change from half the hh, consumption change from the other half): "
            f"median ρ = {sh_med:.2f} (5–95% over 200 splits: {sh_lo:.2f}–{sh_hi:.2f})")


# DHS-matched asset index: the four B2P items with a direct or partial DHS match, at their IWI weights (IWI points);
# phones as any phone (DHS definition), improved floor at the natural-floor weight
IWI_MATCHED = {"light_grid": 8.056664, "tv": 8.612657, "any_phone": 7.127699, "floor_impr": 7.558471}


def matched_index(hh):
    hh = hh.assign(d_any_phone=(hh.phones_f > 0).astype(float) - (hh.phones_b > 0).astype(float))
    hh["d_matched_idx"] = sum(w * hh[f"d_{k}"] for k, w in IWI_MATCHED.items())
    hh["d_elec_phone_idx"] = sum(IWI_MATCHED[k] * hh[f"d_{k}"] for k in ("light_grid", "any_phone"))
    return hh


def assets_vs_cons(hh, site, x="d_asset_idx", xlab="Δ asset index (SD", clip=(-2.5, 3), unit="SD",
                   name="06_assets_vs_consumption.png", title="Asset change vs consumption change: weak per household, "
                   "clearer per site"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    m = hh[[x, "dlog_cons_week"]].dropna()
    hb = a1.hexbin(m[x].clip(*clip), m["dlog_cons_week"].clip(-3, 3.5), gridsize=40, cmap="Blues",
                   mincnt=1, bins="log", linewidths=0)
    rho = m.corr(method="spearman").iloc[0, 1]
    a1.set_title(f"Households (n={len(m):,}), Spearman ρ = {rho:.2f}")
    a1.set_xlabel(f"{xlab}; clipped at {clip[0]:g}, {clip[1]:g})")
    a1.set_ylabel("Δ log weekly consumption (nominal; clipped -3, 3.5)")
    a1.axhline(np.log(1 + (CPI_LO + CPI_HI) / 2), color=INK2, lw=1, ls="--")
    a1.text(0.98, np.log(1.375) + 0.1, "≈ CPI", ha="right", fontsize=8, color=INK2, transform=a1.get_yaxis_transform())
    fig.colorbar(hb, ax=a1, label="households (log)")
    rho_s = site[[x, "dlog_cons_week"]].corr(method="spearman").iloc[0, 1]
    a2.scatter(site[x], site["dlog_cons_week"], s=np.sqrt(site["n"]) * 2.5, color=BLUE, alpha=0.8,
               edgecolor=SURF, lw=1)
    b = np.polyfit(site[x], site["dlog_cons_week"], 1)
    xs = np.linspace(site[x].min(), site[x].max(), 2)
    a2.plot(xs, np.polyval(b, xs), color=INK2, lw=1)
    a2.axhline(np.log(1.375), color=INK2, lw=1, ls="--")
    a2.set_title(f"Sites (n={len(site)}), Spearman ρ = {rho_s:.2f}")
    a2.set_xlabel(f"Site mean {xlab.split(' (')[0]} ({unit})")
    a2.set_ylabel("Site mean Δ log consumption")
    note = site_rho_checks(hh, site, rho_s, x=x)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8, color=INK2, linespacing=1.6)
    save(fig, name)


def by_transition(hh):
    """Median consumption growth for households that gained / kept / lost each item."""
    items = [("phones", "Mobile phones"), ("beds", "Beds"), ("tv", "Television"), ("floor_impr", "Improved floor"),
             ("wall_cement", "Cement ext. wall"), ("animals", "Livestock"), ("light_solar", "Solar lighting"),
             ("light_grid", "Grid electricity"), ("roof_metal", "Metal roof")]
    rows = []
    for k, lab in items:
        dd = hh["d_" + k]
        rows.append([lab] + [hh.loc[s, "dlog_cons_week"].median() for s in (dd < 0, dd == 0, dd > 0)])
    r = pd.DataFrame(rows, columns=["item", "lost", "same", "gained"])
    y = np.arange(len(r))[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for yi, row in zip(y, r.itertuples()):
        ax.plot([row.lost, row.gained], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter(r["lost"], y, color=RED, s=60, zorder=3, label="Lost item", edgecolor=SURF, lw=1.5)
    ax.scatter(r["same"], y, color=NEUTRAL, s=60, zorder=3, label="No change", edgecolor=SURF, lw=1.5)
    ax.scatter(r["gained"], y, color=BLUE, s=60, zorder=3, label="Gained item", edgecolor=SURF, lw=1.5)
    ax.set_yticks(y, r["item"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median Δ log weekly consumption, baseline → round 3")
    ax.set_title("Consumption growth by asset transition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    save(fig, "07_consumption_by_asset_transition.png")


def site_rank(site, hh):
    s = site.sort_values("d_asset_idx").reset_index()
    g = hh.groupby("site")["d_asset_idx"].agg(["mean", "var", "size"])
    g = g[g["size"] >= 30]
    rel = 1 - (g["var"] / g["size"]).mean() / g["mean"].var()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(s)), s["d_asset_idx"], color=BLUE, width=0.8)
    ax.axhline(s["d_asset_idx"].mean(), color=INK2, lw=1, ls="--")
    ax.text(2, s["d_asset_idx"].mean() + 0.03, f"mean {s['d_asset_idx'].mean():.2f}", fontsize=8, color=INK2)
    ax.set_xticks([])
    ax.set_xlim(-1, len(s))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(f"Sites, ranked (n = {len(s)})")
    ax.set_ylabel("Mean Δ asset index (SD)")
    ax.set_title(f"Site-level asset change (reliability {rel:.2f})")
    save(fig, "08_site_asset_change_ranked.png")


def main():
    PLOTS.mkdir(exist_ok=True)
    d = pd.read_csv(DATA / "hh_rounds.csv")
    hh = pd.read_csv(DATA / "hh_change.csv")
    site = pd.read_csv(DATA / "site_change.csv")
    tr = pd.read_csv(DATA / "component_transitions.csv")
    trajectory(d)
    gains_losses(tr)
    contribution(tr)
    hist_delta(hh)
    money(hh)
    assets_vs_cons(hh, site)
    hh = matched_index(hh)
    site = site.merge(hh.groupby("site")[["d_matched_idx", "d_elec_phone_idx"]].mean().reset_index(), on="site")
    assets_vs_cons(hh, site, x="d_matched_idx", clip=(-10, 20), unit="IWI points",
                   xlab="Δ DHS-matched asset index (IWI points", name="06b_matched_assets_vs_consumption.png",
                   title="DHS-matched asset index change vs consumption change\n"
                   "Index = IWI weights on the four B2P items matching DHS: grid 8.06, TV 8.61, any phone 7.13, "
                   "improved floor 7.56")
    assets_vs_cons(hh, site, x="d_elec_phone_idx", clip=(-10, 20), unit="IWI points",
                   xlab="Δ grid + phone index (IWI points", name="06c_elec_phone_vs_consumption.png",
                   title="Grid electricity + phone index change vs consumption change\n"
                   "Index = IWI weights: grid electricity 8.06, any phone 7.13")
    by_transition(hh)
    site_rank(site, hh)
    print(sorted(p.name for p in PLOTS.glob("*.png")))


if __name__ == "__main__":
    main()
