#!/usr/bin/env python3
"""
generate_sample_media.py
------------------------
Creates demo inputs so every VisionSuite command can be tried right after
cloning -- no downloads and no photos of real people needed.

Creates (inside the sample_data/ folder):
    sample_scene.png                         synthetic street-like picture
                                             (a smoke test: it deliberately
                                             contains NO real face or person)
    sample_video.mp4                         short synthetic video clip
    sample_digits/sample_digit_<i>_label_<d>.png
                                             five handwritten digits taken
                                             from scikit-learn's digits data

Run from the project root:
    python sample_data/generate_sample_media.py
"""

import os

import cv2
import numpy as np
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))


def make_scene(path: str) -> None:
    img = np.full((480, 640, 3), (235, 206, 135), np.uint8)       # sky (BGR)
    cv2.rectangle(img, (0, 330), (640, 480), (90, 90, 90), -1)    # road
    cv2.rectangle(img, (0, 300), (640, 335), (150, 150, 150), -1)  # pavement
    for i, x in enumerate(range(20, 600, 150)):                   # buildings
        h = 120 + (i * 37) % 90
        cv2.rectangle(img, (x, 300 - h), (x + 110, 300), (60 + 25 * i, 80, 120), -1)
        for wy in range(300 - h + 15, 285, 35):
            for wx in range(x + 12, x + 100, 30):
                cv2.rectangle(img, (wx, wy), (wx + 16, wy + 20), (200, 230, 250), -1)
    cv2.circle(img, (560, 60), 35, (60, 200, 250), -1)            # sun
    for x in range(0, 640, 80):                                   # lane markings
        cv2.rectangle(img, (x, 400), (x + 40, 408), (240, 240, 240), -1)
    cv2.putText(img, "VisionSuite sample scene", (130, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


def make_video(path: str, seconds: int = 2, fps: int = 15) -> bool:
    size = (320, 240)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        return False
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), (200, 180, 150), np.uint8)
        cv2.rectangle(frame, (0, 170), (320, 240), (90, 90, 90), -1)
        x = 10 + i * 8
        cv2.rectangle(frame, (x, 130), (x + 60, 175), (30, 30, 200), -1)   # "car"
        cv2.circle(frame, (x + 14, 177), 9, (20, 20, 20), -1)
        cv2.circle(frame, (x + 46, 177), 9, (20, 20, 20), -1)
        writer.write(frame)
    writer.release()
    return True


def make_digit_images(folder: str, count: int = 5, upscale: int = 16) -> list:
    """Save the first ``count`` dataset digits as dark-ink-on-white PNGs."""
    os.makedirs(folder, exist_ok=True)
    digits = load_digits()
    paths = []
    for i in range(count):
        small = digits.images[i].astype(np.float32)                      # 0..16, 16 = ink
        big = cv2.resize(small, (8 * upscale, 8 * upscale), interpolation=cv2.INTER_CUBIC)
        big = np.clip(big, 0, 16)
        gray = (255 - np.round(big / 16.0 * 255)).astype(np.uint8)       # dark ink, white paper
        out = os.path.join(folder, f"sample_digit_{i}_label_{int(digits.target[i])}.png")
        cv2.imwrite(out, gray)
        paths.append(out)
    return paths


def main() -> None:
    scene = os.path.join(HERE, "sample_scene.png")
    make_scene(scene)
    print(f"Created {os.path.relpath(scene)}")

    video = os.path.join(HERE, "sample_video.mp4")
    if make_video(video):
        print(f"Created {os.path.relpath(video)}")
    else:
        print("Skipped sample_video.mp4 (this OpenCV build cannot write MP4 files).")

    for p in make_digit_images(os.path.join(HERE, "sample_digits")):
        print(f"Created {os.path.relpath(p)}")


if __name__ == "__main__":
    main()
