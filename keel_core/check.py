"""Spill check on numpy images: did an edit change things outside the region it was allowed to change?

Logic is the Keel v2 checker from the research repo (src/loop/keel_check.py, Amendment 16), written for arrays instead of file paths, plus the
pixel mask rule (src/loop/truth.py) used by the mask-gate mode. Constants are the frozen ones; nothing here was retuned.
"""
import time

import cv2
import numpy as np
import torch

from . import model as M
from .registration import register

THR = 0.85              # network foreground threshold (Amendment 13)
PHOTO_INTENT = 3        # "no hint about where the change should be"
GLOBAL_T = 8.0          # mean-shift rule: mean |channel mean shift| outside the region, in 0-255 levels (Amendment 16)
MARGIN = 0.05           # the allowed region is grown by this fraction before measuring outside it
DIFF_T, BLUR = 30.0, 3.0
SPILL_FRAC_T, GLOBAL_SHIFT_T = 0.005, 6.0     # pixel mask rule (Amendment 15)
MODES = ("keel_v2", "mean_shift_only", "mask_gate")


def load_net(weights, device="cpu"):
    dev = torch.device(device)
    net = M.UNet().to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev))
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, M.NIN, M.S, M.S, device=dev))
    return net, dev


def _boxes(pr, min_area=36):
    fg = 1 - pr[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats((fg > THR).astype(np.uint8), connectivity=8)
    out = []
    for c in range(1, n):
        x, y, w, h, area = st[c]
        if area < min_area:
            continue
        reg = lab == c
        kind = int(np.argmax(pr[1:, reg].sum(1))) + 1
        out.append(dict(kind=M.KIND[kind], bbox=[int(x * 2), int(y * 2), int((x + w) * 2), int((y + h) * 2)], magnitude=float(fg[reg].mean())))
    return out


def spill_check(net, dev, ref, cand, allowed=None, mode="keel_v2"):
    """ref, cand: uint8 RGB arrays (any sizes). allowed: (x0, y0, x1, y1) as fractions of the image where the edit is allowed, or None for 'nothing may change'.
    Returns dict(flagged, mode, shift, findings, pix_mean, pix_frac, aspect_changed, diff, ms) where diff is the 512x512 blurred difference map."""
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    t0 = time.perf_counter()
    ch, cw = cand.shape[:2]
    aspect_changed = abs(ref.shape[1] / ref.shape[0] - cw / ch) > 0.02
    big = [cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) for x in (ref, cand)]
    fr, _ = register(big)
    keep = np.ones((512, 512), bool)
    if allowed:
        x0, y0, x1, y1 = allowed
        keep[int(max(0, y0 - MARGIN) * 512):int(min(1, y1 + MARGIN) * 512), int(max(0, x0 - MARGIN) * 512):int(min(1, x1 + MARGIN) * 512)] = False
    shift = float(np.mean([abs(fr[1][..., c][keep].mean() - fr[0][..., c][keep].mean()) for c in range(3)]))
    # pixel measure (blurred absolute difference after alignment)
    a, b = [cv2.GaussianBlur(np.clip(x, 0, 255), (0, 0), BLUR) for x in fr]
    d = np.abs(a - b).mean(2)
    pix_mean, pix_frac = float(d[keep].mean()), float((d[keep] > DIFF_T).mean())
    pix = bool(pix_frac >= SPILL_FRAC_T or pix_mean >= GLOBAL_SHIFT_T)
    findings = []
    if mode != "mean_shift_only":
        g = fr[1].copy()                       # tone matching: the network sees the candidate with the reference's channel mean/std
        for c in range(3):
            a_, b_ = fr[0][..., c], fr[1][..., c]
            g[..., c] = (b_ - b_.mean()) / (b_.std() + 1e-6) * a_.std() + a_.mean()
        small = np.stack([cv2.resize(np.clip(x, 0, 255).astype(np.uint8), (M.S, M.S), interpolation=cv2.INTER_AREA) for x in (fr[0], g)])
        x = M.build_input(small[0], small[0], small[1], PHOTO_INTENT).unsqueeze(0).to(dev)
        with torch.no_grad():
            pr = torch.softmax(net(x).float(), 1).cpu().numpy()
        for c in _boxes(pr[0]):
            f = dict(kind=c["kind"], score=c["magnitude"], box=[c["bbox"][0] * cw // 512, c["bbox"][1] * ch // 512, c["bbox"][2] * cw // 512, c["bbox"][3] * ch // 512])
            if allowed and allowed[0] <= (f["box"][0] + f["box"][2]) / 2 / cw <= allowed[2] and allowed[1] <= (f["box"][1] + f["box"][3]) / 2 / ch <= allowed[3]:
                continue
            findings.append(f)
    if shift >= GLOBAL_T:
        findings.append(dict(kind="global_tone", score=1.0, box=[0, 0, cw, ch]))
    v2 = bool(findings)
    flagged = {"keel_v2": v2, "mean_shift_only": shift >= GLOBAL_T, "mask_gate": pix and v2}[mode]
    d_out = d.copy()
    d_out[~keep] = 0
    return dict(flagged=flagged, mode=mode, shift=shift, findings=findings, pix_mean=pix_mean, pix_frac=pix_frac, aspect_changed=bool(aspect_changed),
                diff=d_out, ms=round((time.perf_counter() - t0) * 1000, 1))
