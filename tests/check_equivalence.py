"""Equivalence test: the packaged node must give the same flags as the research code on the images the research results were computed from.
    python tests/check_equivalence.py <path to the keel research repo>      (needs its data/ and results/a20_results.json; not part of this repo)
Goes through the real node class with ComfyUI-style tensors (IMAGE [B,H,W,3] float 0-1, MASK [B,H,W]) but does not need ComfyUI installed."""
import importlib.util, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keelpkg", root / "__init__.py", submodule_search_locations=[str(root)])
pkg = importlib.util.module_from_spec(spec); sys.modules["keelpkg"] = pkg; spec.loader.exec_module(pkg)
node = pkg.NODE_CLASS_MAPPINGS["KeelSpillCheck"]()
keel = Path(sys.argv[1])
ALLOWED = {"A": (0, 0, .4, .4), "B": (.6, .6, 1, 1), "C": (0, 0, .5, .5), "D": (.5, .5, 1, 1)}
rows = {r["id"]: r for r in json.load(open(keel / "results/a20_results.json"))["rows"] if "arms" in r}
key = json.load(open(keel / "data/a20/blind_key.json"))


def img(p):
    return torch.from_numpy(cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255.0)[None]


def mask_for(al, h, w):
    m = torch.zeros(1, h, w); x0, y0, x1, y1 = al
    m[0, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
    return m


bad, n = 0, 0
for iid in sorted(rows)[:48]:
    it = key[iid]; a = rows[iid]["arms"]
    o, e = img(keel / "data/real_cc0b" / (it["photo"] + ".jpg")), img(keel / it["out"])
    m = mask_for(ALLOWED[it["prompt"]], e.shape[1], e.shape[2])
    got = {}
    for mode in ("keel_v2", "mean_shift_only", "mask_gate"):
        got[mode] = node.run(o, e, mode, "cpu", m)[0]
    want = {"keel_v2": a["3 Keel v2"], "mean_shift_only": a["1 mean-shift rule"], "mask_gate": a["5 mask gate + Keel v2"]}
    n += 1
    if got != want:
        bad += 1; print("MISMATCH", iid, got, want)
print("compared %d images x 3 modes: %d mismatches" % (n, bad))
r = node.run(o, e, "keel_v2", "cpu", None)
print("no mask (nothing may change):", r[0], "| overlay", tuple(r[3].shape), "| report:", r[2][:160])
r = node.run(torch.cat([o, o]), torch.cat([e, e]), "keel_v2", "cpu", torch.cat([m, m]))
print("batch of 2 ok:", tuple(r[3].shape))
sys.exit(1 if bad else 0)
