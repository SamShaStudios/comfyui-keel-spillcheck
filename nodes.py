"""ComfyUI node: Keel Spill Check. Flags an edit that changed things outside the region it was allowed to change."""
import os

import cv2
import numpy as np
import torch

from .keel_core import check as K

_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "keel_m1.pt")
_NETS = {}


def _net(device):
    if device not in _NETS:
        _NETS[device] = K.load_net(_WEIGHTS, device)
    return _NETS[device]


def _to_u8(t):
    return (t.detach().cpu().float().clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)[..., :3]


def _region(mask, i):
    """MASK -> allowed box as fractions (x0, y0, x1, y1) of the bounding box of the white area, or None if the mask is empty or absent."""
    if mask is None:
        return None
    m = mask[i % mask.shape[0]] if mask.dim() == 3 else mask
    m = (m.detach().cpu().numpy() > 0.5)
    if not m.any():
        return None
    ys, xs = np.where(m)
    h, w = m.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _overlay(edit_u8, diff, allowed):
    h, w = edit_u8.shape[:2]
    heat = np.clip(cv2.resize(diff, (w, h), interpolation=cv2.INTER_LINEAR) / K.DIFF_T, 0, 1)[..., None]
    red = np.zeros_like(edit_u8, dtype=np.float32); red[..., 0] = 255
    out = edit_u8.astype(np.float32) * (1 - 0.65 * heat) + red * 0.65 * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    if allowed:
        cv2.rectangle(out, (int(allowed[0] * w), int(allowed[1] * h)), (int(allowed[2] * w) - 1, int(allowed[3] * h) - 1), (255, 212, 0), max(2, w // 400))
    return out.astype(np.float32) / 255.0


class KeelSpillCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "mode": (list(K.MODES), {"default": "keel_v2"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
            "optional": {
                "allowed_region": ("MASK",),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("spill_detected", "global_shift", "report", "overlay")
    FUNCTION = "run"
    CATEGORY = "image/quality"
    DESCRIPTION = ("Checks whether an image edit changed things OUTSIDE the region it was allowed to change. "
                   "Validated on Qwen-Image edits only; see the README before trusting it on other generators.")

    def run(self, original, edited, mode, device, allowed_region=None):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        net, dev = _net(device)
        n = max(original.shape[0], edited.shape[0])
        lines, overlays, flagged_any, worst = [], [], False, 0.0
        for i in range(n):
            ref, cand = _to_u8(original[i % original.shape[0]]), _to_u8(edited[i % edited.shape[0]])
            allowed = _region(allowed_region, i)
            r = K.spill_check(net, dev, ref, cand, allowed, mode)
            flagged_any |= r["flagged"]; worst = max(worst, r["shift"])
            kinds = sorted({f["kind"] for f in r["findings"]})
            lines.append("image %d: %s | mode %s | global shift %.1f (rule fires at %.0f) | pixel change outside %.2f | findings %s%s%s" % (
                i, "SPILL" if r["flagged"] else "ok", mode, r["shift"], K.GLOBAL_T, r["pix_mean"], ", ".join(kinds) or "none",
                " | region: " + ("none (nothing may change)" if allowed is None else "bounding box of the mask"),
                " | WARNING: aspect ratio differs from the original" if r["aspect_changed"] else ""))
            overlays.append(_overlay(cand, r["diff"], allowed))
        h, w = overlays[0].shape[:2]
        ov = torch.from_numpy(np.stack([o if o.shape[:2] == (h, w) else cv2.resize(o, (w, h)) for o in overlays]))
        return (bool(flagged_any), float(worst), "\n".join(lines), ov)


NODE_CLASS_MAPPINGS = {"KeelSpillCheck": KeelSpillCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"KeelSpillCheck": "Keel Spill Check"}
