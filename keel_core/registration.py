"""Global registration (ECC affine, brightness invariant). Copied unchanged from the Keel research repo (src/seqbench/register.py: estimate, register)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2

H = W = 512
MIN_DISP, MAX_DISP = float(os.environ.get("TK_REG_MIN", 0.8)), 14.0   # mean displacement (px): below MIN_DISP the frame is left as it is, above MAX_DISP the estimate is rejected


def _gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f.astype(np.float32), cv2.COLOR_RGB2GRAY), (0, 0), 1.5)


def _ecc(r, c, W0, mask):
    Wm = W0.copy()
    for lvl in (2, 1, 0):
        s = 1.0 / (2 ** lvl)
        rr, cc = cv2.resize(r, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mm = None if mask is None else cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        Wl = Wm.copy()
        Wl[:, 2] *= s
        try:
            _, Wl = cv2.findTransformECC(rr, cc, Wl, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), mm, 5)
        except cv2.error:
            return Wm
        Wm = Wl.copy()
        Wm[:, 2] /= s
    return Wm


def estimate(ref, cur):
    """Affine 2x3 matrix M with cur(M x) = ref(x) i.e. M maps frame-0 coordinates to the current frame's coordinates (cv2.WARP_INVERSE_MAP convention)."""
    r, c = _gray(ref), _gray(cur)
    mask = np.zeros((H, W), np.uint8)
    mask[12:-12, 12:-12] = 1
    Wm = np.eye(2, 3, dtype=np.float32)
    for _ in range(2):
        Wm = _ecc(r, c, Wm, mask)
        warped = cv2.warpAffine(c, Wm, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        a, b = (r - r.mean()) / (r.std() + 1e-6), (warped - warped.mean()) / (warped.std() + 1e-6)
        res = np.abs(a - b)
        keep = (res < max(0.35, 2.5 * float(np.median(res)))).astype(np.uint8)
        mask = keep * np.pad(np.ones((H - 24, W - 24), np.uint8), 12)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    return Wm


def _disp(M):
    """mean displacement (px) the transform causes over a grid of image points"""
    g = np.array([[x, y, 1] for x in range(40, 480, 110) for y in range(40, 480, 110)], float)
    return float(np.linalg.norm(g @ M.T - g @ np.eye(2, 3).T, axis=1).mean())


def negligible(M):
    return _disp(M) < MIN_DISP


def implausible(M):
    sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    return _disp(M) > MAX_DISP or abs(sc - 1) > 0.04


def register(frames):
    """frames: list of float32 RGB arrays (frame 0 first). Returns frames expressed in frame-0 coordinates (and the transforms)."""
    out, Ms = [frames[0]], [np.eye(2, 3, dtype=np.float32)]
    for f in frames[1:]:
        M = estimate(frames[0], f)
        if implausible(M):
            M = np.eye(2, 3, dtype=np.float32)
        Ms.append(M)
        if negligible(M):
            out.append(f)
        else:
            out.append(cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE))
    return out, Ms


