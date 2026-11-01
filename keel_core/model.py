"""Keel change-detector network (inference only). Copied unchanged from the Keel research repo (src/seqbench/a13_model.py: UNet, build_input)."""
import argparse, glob, json, math, random, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

S = 256
NCLS = 7
NIN = 15 + 4


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = conv(NIN, w), conv(w, 2 * w), conv(2 * w, 4 * w), conv(4 * w, 8 * w)
        self.b = conv(8 * w, 8 * w)
        self.d4, self.d3, self.d2, self.d1 = conv(16 * w, 4 * w), conv(8 * w, 2 * w), conv(4 * w, w), conv(2 * w, w)
        self.out = nn.Conv2d(w, NCLS, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat([up(b), e4], 1))
        d3 = self.d3(torch.cat([up(d4), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)


def build_input(f0, fp, ft, intent):
    """uint8 arrays (S,S,3) -> float tensor (NIN,S,S)"""
    a, b, c = (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255 for x in (ft, fp, f0))
    inp = torch.cat([a, b, c, (a - b).abs() * 4, (a - c).abs() * 4], 0)
    oh = torch.zeros(4, S, S)
    oh[intent] = 1
    return torch.cat([inp, oh], 0)


KIND = {1: "colour", 2: "size", 3: "position", 4: "presence_any", 5: "lighting", 6: "state"}
