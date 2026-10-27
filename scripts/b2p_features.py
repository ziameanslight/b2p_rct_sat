"""Satellite features for every B2P household and imagery year 2020-2024, and for the Rwanda DHS training places, from
the village-wealth-change model.

  python scripts/b2p_features.py          (~10 min on an Apple M-series GPU; needs data/aef from b2p_fetch_aef.py)

Per place-year, at the household's GPS point (B2P) or the relocated village point (DHS):
  emb  64-d embedding of the multi-scale CNN level net for test year 2025 (village-wealth-change
       data/models/ms/t2025.pt; trained on wealth levels of every rural DHS + LSMS cluster surveyed before 2025, never
       on change). Inputs: 1.28 km at 10 m, 25 km at 160 m, 100 km at 640 m, as src/cnn_ms.py.
  sat  mean AlphaEarth embedding in 1.28 / 2.56 / 10.24 km squares at 160 m (192; NaN below 60% coverage), as
       src/common.box_feats.
B2P inputs are cut from the tiles of b2p_fetch_aef.py: the 10 m patch from the raw tiles, the 160 m and 640 m windows
from a per-(CRS, year) mosaic of the 16x overviews (640 m = 4 x 4 mean of the 160 m mosaic). DHS inputs are read from
the village-wealth-change workspace ($VC_WORK/data/aef_fine, aef_ms) for the Rwanda DHS matched pairs.
Outputs: data/hh_year_msfeats.npz (uid, year, emb, sat), data/rw_pair_feats.npz (cid, year, emb, sat)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, VC_DATA, VWC  # noqa: E402
from b2p_common import imagery_year  # noqa: E402
from b2p_fetch_aef import YEARS, points  # noqa: E402
from build_multiscale import quant  # noqa: E402
from cnn_ms import DEV, Data, MultiScale, log  # noqa: E402
from common import P, SHIFT, crop, deq, item_shares, locations, pairs_10m  # noqa: E402
from pairs import Feats, pair_tables  # noqa: E402

AEF = DATA / "aef"
TS = 1024
HP = 67                                      # 134 px fine patch = 128 px + 3 px shift margin each side
BOX = {"1.28": 4, "2.56": 8, "10.24": 32}    # half-width in 160 m px
S = 156                                      # mid / wide window, px


def load_net():
    ck = torch.load(VWC / "data/models/ms/t2025.pt", map_location=DEV)
    net = MultiScale().to(DEV)
    net.load_state_dict(ck["net"])
    return net.eval()


@torch.no_grad()
def embed(net, fine, mid, wide, bs=128):
    return np.concatenate([net(*Data._tensors(fine[k:k + bs], mid[k:k + bs], wide[k:k + bs], False)).cpu().numpy()
                           for k in range(0, len(fine), bs)])


# ------------------------------------------------------------------------------------------------ B2P inputs
class Tiles:
    """Raw 10 m tiles of b2p_fetch_aef.py."""

    def __init__(self):
        self.F = pd.read_parquet(AEF / "files.parquet")
        self.cache = {}

    def tile(self, fid, ty, tx):
        k = (fid, ty, tx)
        if k not in self.cache:
            f = AEF / "t10" / f"{fid}_{ty}_{tx}.npy"
            self.cache[k] = np.load(f, mmap_mode="r") if f.exists() else None
        return self.cache[k]

    def patch(self, year, crs, x, y):
        """(64, 134, 134) int8 north-up centred on (x, y); -128 where missing."""
        out = np.full((64, 2 * HP, 2 * HP), -128, np.int8)       # south-up while filling
        f = self.F[(self.F.year == year) & (self.F.crs == crs) & (self.F.utm_west <= x) & (self.F.utm_east > x) &
                   (self.F.utm_south <= y) & (self.F.utm_north > y)]
        if not len(f):
            return out
        r = f.iloc[0]
        c0 = int(np.floor((x - r.utm_west) / 10)) - HP
        r0 = int(np.floor((y - r.utm_south) / 10)) - HP
        for ty in range(max(r0, 0) // TS, min(r0 + 2 * HP - 1, 8 * TS - 1) // TS + 1):
            for tx in range(max(c0, 0) // TS, min(c0 + 2 * HP - 1, 8 * TS - 1) // TS + 1):
                t = self.tile(r.fid, ty, tx)
                if t is None:
                    continue
                fr0, fr1 = max(r0, ty * TS), min(r0 + 2 * HP, (ty + 1) * TS)
                fc0, fc1 = max(c0, tx * TS), min(c0 + 2 * HP, (tx + 1) * TS)
                out[:, fr0 - r0:fr1 - r0, fc0 - c0:fc1 - c0] = t[:, fr0 - ty * TS:fr1 - ty * TS,
                                                                 fc0 - tx * TS:fc1 - tx * TS]
        return out[:, ::-1].copy()


class Mosaic160:
    """One (CRS, year): 160 m overviews pasted on the global 160 m grid, row 0 = south; plus its 640 m 4 x 4 mean."""

    def __init__(self, F):
        self.gx0 = int(F.utm_west.min() // 160)
        self.gy0 = int(F.utm_south.min() // 160)
        nx = int(F.utm_east.max() // 160) - self.gx0
        ny = int(F.utm_north.max() // 160) - self.gy0
        self.A = np.full((64, ny, nx), -128, np.int8)
        for r in F.itertuples():
            ov = np.load(AEF / "t160" / f"{r.fid}.npy")
            ox, oy = int(r.utm_west // 160) - self.gx0, int(r.utm_south // 160) - self.gy0
            sub = self.A[:, oy:oy + ov.shape[1], ox:ox + ov.shape[2]]
            ok = ov[0, :sub.shape[1], :sub.shape[2]] != -128
            sub[:, ok] = ov[:, :sub.shape[1], :sub.shape[2]][:, ok]
        ny4, nx4 = ny // 4 * 4, nx // 4 * 4
        self.W = np.empty((64, ny4 // 4, nx4 // 4), np.int8)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for b in range(64):
                v = deq(self.A[b, :ny4, :nx4]).reshape(ny4 // 4, 4, nx4 // 4, 4)
                self.W[b] = quant(np.nanmean(v, axis=(1, 3)))

    def boxes(self, x, y):
        cx = int(np.round(x / 160 - 0.5)) - self.gx0
        cy = int(np.round(y / 160 - 0.5)) - self.gy0
        out = []
        for h in BOX.values():
            v = deq(self.A[:, max(cy - h, 0):cy + h, max(cx - h, 0):cx + h].reshape(64, -1))
            cov = (~np.isnan(v[0])).sum() / (2 * h) ** 2
            out.append(np.nanmean(v, 1) if cov >= 0.6 else np.full(64, np.nan, np.float32))
        return np.concatenate(out)

    @staticmethod
    def _cut(A, r0, c0, n):
        out = np.full((64, n, n), -128, np.int8)
        R0, C0 = max(r0, 0), max(c0, 0)
        R1, C1 = min(r0 + n, A.shape[1]), min(c0 + n, A.shape[2])
        if R1 > R0 and C1 > C0:
            out[:, R0 - r0:R1 - r0, C0 - c0:C1 - c0] = A[:, R0:R1, C0:C1]
        return out[:, ::-1]                                    # north-up

    def windows(self, x, y):
        gx, gy = x / 160 - self.gx0, y / 160 - self.gy0
        mid = self._cut(self.A, int(round(gy)) - S // 2, int(round(gx)) - S // 2, S)
        wide = self._cut(self.W, int(round(gy / 4)) - S // 2, int(round(gx / 4)) - S // 2, S)
        return mid, wide


def b2p_feats(net):
    t0 = time.time()
    H = points()
    base = pd.read_csv(DATA / "hh_rounds.csv", parse_dates=["date"])
    H = H[H.uid.isin(base.uid)]
    T = Tiles()
    feats = {}
    for year in YEARS:
        for crs, g in H.groupby("crs"):
            M = Mosaic160(T.F[(T.F.year == year) & (T.F.crs == crs)])
            for k in range(0, len(g), 1000):
                gg = g.iloc[k:k + 1000]
                fine = np.stack([T.patch(year, crs, r.x, r.y) for r in gg.itertuples()])
                mw = [M.windows(r.x, r.y) for r in gg.itertuples()]
                e = embed(net, fine, np.stack([a for a, _ in mw]), np.stack([b for _, b in mw]))
                e[(fine[:, 0, SHIFT:SHIFT + P, SHIFT:SHIFT + P] != -128).mean((1, 2)) <= 0.9] = np.nan
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = np.stack([M.boxes(r.x, r.y) for r in gg.itertuples()])
                for u, ee, ss in zip(gg.uid, e, s):
                    feats[(u, year)] = (ee, ss)
            log(f"B2P {year} {crs}: {len(g)} households, {time.time() - t0:.0f}s")
    ks = list(feats)
    np.savez(DATA / "hh_year_msfeats.npz", uid=np.array([k[0] for k in ks]), year=np.array([k[1] for k in ks]),
             emb=np.stack([feats[k][0] for k in ks]), sat=np.stack([feats[k][1] for k in ks]))


# ------------------------------------------------------------------------------------------------ DHS inputs
def rw_pair_feats(net):
    """Both survey years of every place of the Rwanda DHS matched pairs: the fine patch at the relocated point (a
    matched place's other-year patch cut from the other cluster's chip), mid / wide windows of the workspace."""
    L = locations()
    T, _ = pair_tables(L, item_shares(), targets=[])
    T = T[(T.cc == "RW") & (T.src == "DHS")]
    keys = sorted({(int(p), int(y)) for r in T.itertuples() for p in (r.i, r.j) for y in (r.y1, r.y2)})
    chips = np.load(VC_DATA / "aef_fine/chips_10m.npy", mmap_mode="r")
    lim = 88 - P // 2 - SHIFT
    own = {(int(c), int(r.year)): (int(r.chip_row), int(np.clip(round(r.fine_dx_px), -lim, lim)),
                                   int(np.clip(round(r.fine_dy_px), -lim, lim))) for c, r in L.iterrows()}
    for r in pairs_10m(L).itertuples():
        own.setdefault((int(r.place), int(r.chip_year)), (int(L.chip_row[r.chip]), r.ox, r.oy))
    Mm = pd.read_parquet(VC_DATA / "aef_ms/meta.parquet")
    ms_key = {(int(c), int(y)): int(r) for c, y, r in zip(Mm.cid, Mm.year, Mm.row)}
    Fs = Feats()
    keys = [k for k in keys if k in own and k in ms_key and Fs.get(*k) is not None]
    Xm = np.load(VC_DATA / "aef_ms/mid_160m.npy", mmap_mode="r")
    Xw = np.load(VC_DATA / "aef_ms/wide_640m.npy", mmap_mode="r")
    rows = np.array([ms_key[k] for k in keys])
    fine = np.stack([crop(chips, *own[k]) for k in keys])
    e = embed(net, fine, np.stack([Xm[i] for i in rows]), np.stack([Xw[i] for i in rows]))
    np.savez(DATA / "rw_pair_feats.npz", cid=np.array([k[0] for k in keys]), year=np.array([k[1] for k in keys]),
             emb=e, sat=np.stack([Fs.get(*k) for k in keys]))
    log(f"Rwanda DHS: {len(T)} pairs, {len(keys)} place-years embedded")


def main():
    net = load_net()
    rw_pair_feats(net)
    b2p_feats(net)


if __name__ == "__main__":
    main()
