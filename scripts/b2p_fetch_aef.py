"""Fetch AlphaEarth (AEF) annual embeddings around every B2P household point, 2020-2024.

  python scripts/b2p_fetch_aef.py [--count]      (restartable)

10 m: every 1024 px tile (10.24 km) of the full-resolution COG that a household's 1.28 km box touches is read once
(HTTP range requests, village-wealth-change pipeline/fetch_aef_fine.py) and stored raw -> data/aef/t10/<fid>_<ty>_<tx>.npy
(int8 (64, 1024, 1024), row 0 = south, -128 = nodata; ~67 MB each, ~37 GB total). Raw pixels are kept so that the
CNN can read full 1.28 km patches as well as the ridge its box means.
160 m: the whole 16x overview of every file within 50 km of a household (512 x 512 px, int8; the multi-scale
CNN reads 100 km windows) -> data/aef/t160/<fid>.npy, reusing
$VC_WORK/data/aef_tiles16 when the file is already cached there.
CRS per point: the UTM zone of its longitude (35S west of 30 E, 36S east), matching the AEF file grid.
"""
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, VC_DATA  # noqa: E402  (puts village-wealth-change/pipeline on the path)
from fetch_aef_fine import fetch_tile, header  # noqa: E402
from fetch_aef_coarse import read_overview  # noqa: E402

OUT = DATA / "aef"
YEARS = [2020, 2021, 2022, 2023, 2024]
TS, NT, BLK = 1024, 8, 8
HALF10 = 640.0                  # 1.28 km box half-width, m (+ 1 block margin below)
CTX = 313 * 160.0               # 50.08 km: 160 m overviews out to the multi-scale CNN's 100 km window


def points():
    H = pd.read_parquet(DATA / "hh_points.parquet")
    H["crs"] = np.where(H.lon < 30, "EPSG:32735", "EPSG:32736")
    for crs, g in H.groupby("crs"):
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(g.lon.values, g.lat.values)
        H.loc[g.index, "x"], H.loc[g.index, "y"] = x, y
    return H


def file_index(H):
    ix = gpd.read_parquet(VC_DATA / "aef/aef_index.parquet")
    ix = ix[ix.year.isin(YEARS) & ix["crs"].isin(H.crs.unique())]
    keep = []
    for r in ix.itertuples():
        g = H[H.crs == r.crs]
        if ((g.x + CTX > r.utm_west) & (g.x - CTX < r.utm_east) & (g.y + CTX > r.utm_south) &
                (g.y - CTX < r.utm_north)).any():
            keep.append(r.Index)
    return pd.DataFrame(ix).loc[keep, ["fid", "path", "crs", "year", "utm_west", "utm_south", "utm_east", "utm_north"]]


def tiles10(H, F):
    """(fid, path, ty, tx) of every 10 m tile touched by a household's 1.28 km box (+80 m margin)."""
    need = set()
    m = HALF10 + 80
    for r in F.itertuples():
        g = H[(H.crs == r.crs) & (H.x + m > r.utm_west) & (H.x - m < r.utm_east) & (H.y + m > r.utm_south) &
              (H.y - m < r.utm_north)]
        if not len(g):
            continue
        c0 = np.clip(((g.x - m - r.utm_west) / 10 // TS).astype(int), 0, NT - 1)
        c1 = np.clip(((g.x + m - r.utm_west) / 10 // TS).astype(int), 0, NT - 1)
        r0 = np.clip(((g.y - m - r.utm_south) / 10 // TS).astype(int), 0, NT - 1)
        r1 = np.clip(((g.y + m - r.utm_south) / 10 // TS).astype(int), 0, NT - 1)
        for a, b, c, d in zip(c0, c1, r0, r1):
            for tx in range(a, b + 1):
                for ty in range(c, d + 1):
                    need.add((r.fid, r.path, ty, tx))
    return sorted(need)


def main():
    H = points()
    F = file_index(H)
    T = tiles10(H, F)
    print(f"{len(H)} households, {len(F)} files, {len(T)} 10 m tiles over {len(YEARS)} years", flush=True)
    (OUT / "t10").mkdir(parents=True, exist_ok=True)
    (OUT / "t160").mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT / "files.parquet")
    pd.DataFrame(T, columns=["fid", "path", "ty", "tx"]).to_parquet(OUT / "tiles10.parquet")

    # 160 m overviews
    cache = VC_DATA / "aef_tiles16"
    todo = []
    for r in F.itertuples():
        dst = OUT / "t160" / f"{r.fid}.npy"
        if dst.exists():
            continue
        if (cache / f"{r.fid}.npy").exists():
            shutil.copyfile(cache / f"{r.fid}.npy", dst)
        else:
            todo.append(r)
    print(f"160 m: {len(F) - len(todo)} files on disk, {len(todo)} to read", flush=True)
    if "--count" not in sys.argv:
        with ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(read_overview, r.path): r for r in todo}
            for fu in as_completed(futs):
                r = futs[fu]
                np.save(OUT / "t160" / f"{r.fid}.tmp.npy", fu.result())
                (OUT / "t160" / f"{r.fid}.tmp.npy").rename(OUT / "t160" / f"{r.fid}.npy")
        print("160 m done", flush=True)

    # 10 m tiles
    paths = sorted({p for _, p, _, _ in T})
    with ThreadPoolExecutor(16) as ex:
        hdr = {p: (o, c) for p, o, c in ex.map(header, paths)}
    todo = [t for t in T if not (OUT / "t10" / f"{t[0]}_{t[2]}_{t[3]}.npy").exists()]
    est = sum(sum(hdr[p][1][b * 64 + ty * NT + tx] for b in range(64)) for _, p, ty, tx in todo) / 1e9
    print(f"10 m: {len(T) - len(todo)} tiles on disk, {len(todo)} to fetch, {est:.1f} GB", flush=True)
    if "--count" in sys.argv:
        return
    t0, got = time.time(), 0
    with ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(fetch_tile, p, ty, tx, *hdr[p]): (fid, ty, tx) for fid, p, ty, tx in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            fid, ty, tx = futs.pop(fu)
            _, _, _, tile, nb = fu.result()
            got += nb
            dst = OUT / "t10" / f"{fid}_{ty}_{tx}"
            np.save(str(dst) + ".tmp.npy", tile)       # raw int8, row 0 = south (the CNN needs full 10 m patches)
            Path(str(dst) + ".tmp.npy").rename(str(dst) + ".npy")
            del tile
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} tiles, {got / 1e9:.1f} GB, {got / 1e6 / el:.1f} MB/s, "
                      f"ETA {(len(todo) - i) * el / i / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
