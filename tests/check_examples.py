"""Checks the shipped example images behave as the README says: the clean edit is not flagged, the edit with a colour cast is.
    python tests/check_examples.py        (needs torch, numpy, opencv; does not need ComfyUI)"""
import importlib.util, sys
from pathlib import Path

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keelpkg", root / "__init__.py", submodule_search_locations=[str(root)])
pkg = importlib.util.module_from_spec(spec); sys.modules["keelpkg"] = pkg; spec.loader.exec_module(pkg)
node = pkg.NODE_CLASS_MAPPINGS["KeelSpillCheck"]()


def img(name):
    return torch.from_numpy(cv2.imread(str(root / "examples" / name))[:, :, ::-1].astype(np.float32) / 255.0)[None]


mask = torch.zeros(1, 1000, 1000); mask[0, :400, :400] = 1.0      # the same mask the example workflow builds: top-left 40 percent
o = img("keel_example_original.jpg")
bad = 0
for name, want in (("keel_example_edit_ok.png", False), ("keel_example_edit_spill.png", True)):
    for mode in ("keel_v2", "mean_shift_only", "mask_gate"):
        got = node.run(o, img(name), mode, "cpu", mask)
        status = "ok" if got[0] == want else "WRONG"
        bad += got[0] != want
        print("%-30s %-16s flagged=%-5s expected=%-5s %s | %s" % (name, mode, got[0], want, status, got[2].split("|")[3].strip()))
sys.exit(1 if bad else 0)
