# Keel Spill Check (ComfyUI custom node) — research prototype

Checks whether an image edit changed things **outside the region it was supposed to change**.

You give it the original image, the edited image, and a mask of the region the edit was allowed to touch. It tells you
whether anything else visibly changed (a colour or brightness shift, an added or moved object) and draws where.

> **Read this first.** This is a research prototype, not a product. It was evaluated on Qwen-Image edits with one human labeller.
> It is **not reliable on ChatGPT-image edits** (it flags most of them). Numbers and limits are below; please check them before relying on it.

## What it does

Node: **Keel Spill Check** (category `image/quality`).

| Input | |
|---|---|
| `original` | the image before the edit |
| `edited` | the edited result (any size; both are compared at 512×512 after alignment) |
| `allowed_region` (optional MASK) | white = where the edit is allowed. The **bounding box** of the white area is used. No mask = nothing may change |
| `mode` | `keel_v2` (default), `mean_shift_only`, or `mask_gate` — see below |
| `device` | `cpu` (default; about 0.1–0.4 s per image) or `cuda` |

| Output | |
|---|---|
| `spill_detected` | BOOLEAN, true if any image in the batch was flagged |
| `global_shift` | FLOAT, the largest "mean colour shift outside the region" in the batch (0–255 levels) |
| `report` | STRING, one line per image |
| `overlay` | IMAGE, the edited image with red where it changed outside the region and a yellow box around the allowed region |

Modes:
- `mean_shift_only` — a plain rule: flags when the average colour of everything outside the region moved by 8 or more levels. Fast, no network, almost no false alarms on Qwen.
- `keel_v2` — the rule above plus a small learned change detector (1.9M parameters, included in `weights/`) that finds local changes outside the region.
- `mask_gate` — flags only when the `keel_v2` result **and** a pixel-difference rule both fire (fewer false alarms, fewer catches).

## Install

Copy or clone this folder into `ComfyUI/custom_nodes/comfyui-keel-spillcheck/`, install `requirements.txt` into ComfyUI's Python
environment (it needs `opencv-python-headless`; `torch` and `numpy` come with ComfyUI), and restart ComfyUI.
Wire: `Load Image` (original) and your edited output → **Keel Spill Check**; optionally a mask for the allowed region.

**Not yet tested inside a running ComfyUI.** The node follows ComfyUI's conventions and was tested through its class with ComfyUI-style tensors
(`tests/check_equivalence.py`: identical results to the research code on 48 images in all three modes), but I have not loaded it in the ComfyUI UI.
If it fails to load, please open an issue with the console log.

## How well it works (honest numbers)

Human-labelled by one person ("is there a change outside the box you would notice at normal size?"). Catch = share of problem outputs flagged;
false alarm = share of clean outputs flagged.

**Qwen-Image edits, 96 outputs (52 problems / 44 clean), four edit types on 24 CC0 photos:**

| mode | catch | false alarm |
|---|---|---|
| `mean_shift_only` | 0.75 | 0.05 |
| `keel_v2` | 0.85 | 0.07 |
| `mask_gate` | 0.83 | 0.07 |

The simple rule does most of the work; the learned network adds about ten points of catch.

**ChatGPT-image edits, 96 outputs (42 problems / 54 clean): do not rely on it.**

| mode | catch | false alarm |
|---|---|---|
| `mean_shift_only` | 0.02 | 0.00 |
| `keel_v2` | 0.83 | 0.63 |
| `mask_gate` | 0.40 | 0.09 |

ChatGPT regenerates the whole image, and the network flags the re-rendering. A few ways of combining the checks did not fix this (catch about 0.50 at a 0.25 false-alarm budget).

Other known limits:
- **Brightness/tint edits** (e.g. "brighten the top-left quarter"): nearly every output gets flagged, including ones a person calls clean. The soft edge of such edits is ambiguous.
- **Object edits** (add a balloon or sticker): in the first-round data (Qwen plus a few ChatGPT outputs, 60 object-edit outputs) only 6 were labelled as problems and the checker caught none of them; it is not shown to work for those.
- Non-box masks use their bounding box. Only box regions were evaluated.
- 24 photos, one labeller, one run per cell. Intervals are wide. The labeller changed the label on 25% of the control pairs on a later review.

The full pre-registered write-up (including the failures) is in the Keel research notes; ask if you want a copy.

## What it does not do

It only checks for unwanted change outside the region. It does **not** judge whether the edit itself is good, and it does not repair anything.

## Provenance

The network was trained by the author on synthetic cartoon scenes plus edits made to CC0 photographs, with no third-party model weights.
The code in `keel_core/` is copied from the author's research repository. **Licence: not yet chosen.** Do not redistribute until a licence file is added.
