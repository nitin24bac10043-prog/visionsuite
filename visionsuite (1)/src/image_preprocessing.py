"""
image_preprocessing.py
----------------------
Shared computer-vision utilities used by more than one module, so the
face and people modules do not duplicate code.

Boxes are always ``(x, y, w, h)`` tuples of ints.
"""

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Read an image from disk as a BGR array, or raise a clear error."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file does not exist: '{path}'")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read '{path}' as an image (unsupported or corrupt file).")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    """Write an image, creating parent folders; raise IOError on failure."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise IOError(f"Could not write image to '{path}' (check the file extension and permissions).")


# --------------------------------------------------------------------------
# Resizing (performance)
# --------------------------------------------------------------------------
def resize_max_dimension(image: np.ndarray, max_dim: int = 1280) -> Tuple[np.ndarray, float]:
    """Shrink ``image`` so its longest side is at most ``max_dim`` pixels.

    Returns ``(resized_image, scale)`` where ``scale <= 1.0``. Images that
    are already small enough are returned unchanged (scale 1.0) -- we never
    upscale. Detecting on a smaller copy is much faster; use ``scale_boxes``
    to map the detections back to the original resolution.
    """
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def scale_boxes(boxes: Sequence[Sequence[int]], scale: float) -> List[Box]:
    """Map boxes found on a resized image back to the original image."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    factor = 1.0 / scale
    return [tuple(int(round(v * factor)) for v in b) for b in boxes]  # type: ignore[misc]


def clamp_box(box: Sequence[int], img_w: int, img_h: int) -> Box:
    """Clip a box so it lies fully inside an ``img_w`` x ``img_h`` image."""
    x, y, w, h = [int(v) for v in box]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


# --------------------------------------------------------------------------
# Non-maximum suppression (used by the people counter)
# --------------------------------------------------------------------------
def non_max_suppression(boxes: Sequence[Sequence[int]],
                        overlap_thresh: float = 0.65,
                        scores: Optional[Sequence[float]] = None) -> List[Box]:
    """Remove boxes that overlap too much with a better box.

    HOG detectors fire several times around one person. For each cluster of
    heavily overlapping boxes we keep just one (highest score if ``scores``
    are given, otherwise the lowest box -- the usual convention).
    ``overlap_thresh`` is the fraction of overlap above which a box is
    considered a duplicate.
    """
    if len(boxes) == 0:
        return []
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(np.asarray(scores, dtype=np.float64).reshape(-1) if scores is not None else y2)[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / areas[rest]
        order = rest[overlap <= overlap_thresh]

    return [tuple(int(v) for v in arr[i]) for i in keep]  # type: ignore[misc]


# --------------------------------------------------------------------------
# Blur helper (used by the face module)
# --------------------------------------------------------------------------
def blur_region(image: np.ndarray, box: Sequence[int], kernel: int = 35) -> np.ndarray:
    """Return a copy of ``image`` with ``box`` Gaussian-blurred."""
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Gaussian blur kernel must be a positive odd integer.")
    out = image.copy()
    h, w = out.shape[:2]
    x, y, bw, bh = clamp_box(box, w, h)
    if bw == 0 or bh == 0:
        return out
    roi = out[y:y + bh, x:x + bw]
    out[y:y + bh, x:x + bw] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    return out


# --------------------------------------------------------------------------
# Video helpers (shared by the face and people modules)
# --------------------------------------------------------------------------
def open_video(path: str):
    """Open a video file; returns ``(capture, fps, (width, height))``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file does not exist: '{path}'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open '{path}' as a video (unsupported or corrupt file).")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:      # 0 / NaN -> fall back to 25
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), (width, height)


def create_video_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 writer (``mp4v`` codec ships with OpenCV)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise IOError(f"Could not create output video '{path}'.")
    return writer
