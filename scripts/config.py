"""Paths shared by the scripts, and the link to the village-wealth-change model repo.

  VWC_REPO  clone of https://github.com/ziameanslight/wealthchange (default: ../village-wealth-change); provides the
            multi-scale CNN (src/cnn_ms.py, data/models/ms/t2025.pt), the matched DHS pairs and their 160 m features
            (src/pairs.py, data/feats_160m.npz) and the AlphaEarth fetch code (pipeline/)
  VC_WORK   that repo's workspace (default <VWC_REPO>/work; $VC_WORK/data: relocated clusters, households, AEF chips
            and windows, AEF index)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "b2psurvey data"              # raw B2P survey export + rct_bridges.csv (never committed)
DATA, PLOTS = ROOT / "data", ROOT / "plots"
VWC = Path(os.environ.get("VWC_REPO", ROOT.parent / "village-wealth-change")).resolve()
os.environ.setdefault("VC_WORK", str(VWC / "work"))    # village-wealth-change's own default
VC_DATA = Path(os.environ["VC_WORK"]) / "data"
for p in (VWC / "src", VWC / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
