"""Development fixture: synthetic Blender-like media for ingestion tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 640, 360


@dataclass(frozen=True)
class UIState:
    mode: str = "Object Mode"
    obj_w: float = 0.12       # object width as a fraction of the viewport width
    obj_h: float = 0.2
    bottom: float = 0.75      # object bottom edge (fraction of image height)
    camera: float = 0.0       # orbit angle proxy: shifts the viewport gradient and object


def draw(state: UIState) -> Image.Image:
    img = Image.new("RGB", (W, H), (43, 43, 43))
    viewport = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0, 1, W)[None, :]
    ys = np.linspace(0, 1, H)[:, None]
    shade = 55 + 40 * np.sin(3.0 * (xs + ys * 0.5) + state.camera * 2.0)
    viewport[..., 0] = shade
    viewport[..., 1] = shade + 3
    viewport[..., 2] = shade + 8
    img.paste(Image.fromarray(viewport).crop((0, 0, int(W * 0.78), H - 22)), (0, 22))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 21], fill=(35, 35, 35))
    d.text((8, 5), state.mode, fill=(220, 220, 220))
    d.rectangle([int(W * 0.78), 22, W, H], fill=(60, 60, 60))
    shift = state.camera * 60
    cx = W * 0.39 + shift
    x0, x1 = cx - state.obj_w * W / 2, cx + state.obj_w * W / 2
    y1 = state.bottom * H
    y0 = y1 - state.obj_h * H
    d.rectangle([x0, y0, x1, y1], fill=(190, 190, 195))
    return img


def tutorial_script() -> list[tuple[float, UIState]]:
    s0 = UIState()
    s1 = replace(s0, mode="Edit Mode")
    s2 = replace(s1, obj_w=0.04)                  # scale x (narrower)
    s3 = replace(s2, obj_h=0.5)                   # scale z (taller)
    s4 = replace(s3, camera=0.8)                  # orbit
    s5 = replace(s4, obj_h=0.6)                   # grows upwards only: extrusion-like
    return [(2.0, s0), (2.0, s1), (2.0, s2), (2.0, s3), (2.0, s4), (2.0, s5)]


def write_video(path: Path, script: list[tuple[float, UIState]], fps: int = 10) -> float:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    total = 0.0
    for duration, state in script:
        frame = np.asarray(draw(state))[:, :, ::-1].copy()
        for _ in range(int(duration * fps)):
            writer.write(frame)
        total += duration
    writer.release()
    return total
