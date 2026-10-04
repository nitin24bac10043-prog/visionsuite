"""
people_counter_module.py
------------------------
Module 2: detect and count pedestrians.

Technique: HOG (Histogram of Oriented Gradients) features + OpenCV's
built-in, pre-trained linear SVM people detector. Overlapping detections of
the same person are merged with non-maximum suppression so nobody is counted
twice.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.image_preprocessing import (
    Box, create_video_writer, load_image, non_max_suppression, open_video,
    resize_max_dimension, save_image, scale_boxes,
)
from src.logger_setup import get_logger

logger = get_logger(__name__)

DETECTION_MAX_DIM = 800      # HOG is slow on big images; 800 px is a good trade-off
BOX_COLOUR = (0, 200, 0)     # BGR green


class PeopleCounter:
    """HOG + SVM pedestrian detector.

    Parameters (configurable in ``config.yaml``):
      hit_threshold -- SVM decision threshold; lower = more detections
      win_stride    -- sliding-window step in pixels; smaller = slower, finer
    """

    def __init__(self, hit_threshold: float = 0.0, win_stride: int = 8,
                 scale: float = 1.05, nms_overlap: float = 0.65):
        if win_stride < 1:
            raise ValueError("win_stride must be >= 1")
        if scale <= 1.0:
            raise ValueError("scale must be greater than 1.0")
        if not (0.0 < nms_overlap <= 1.0):
            raise ValueError("nms_overlap must be in (0, 1]")

        self.hit_threshold = float(hit_threshold)
        self.win_stride = int(win_stride)
        self.scale = scale
        self.nms_overlap = nms_overlap

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Box]:
        """Return de-duplicated person boxes ``(x, y, w, h)`` for ``image``."""
        small, factor = resize_max_dimension(image, DETECTION_MAX_DIM)
        rects, weights = self._hog.detectMultiScale(
            small,
            hitThreshold=self.hit_threshold,
            winStride=(self.win_stride, self.win_stride),
            padding=(8, 8),
            scale=self.scale,
        )
        if len(rects) == 0:
            return []
        scores = np.asarray(weights, dtype=np.float64).reshape(-1)
        kept = non_max_suppression([tuple(int(v) for v in r) for r in rects],
                                   overlap_thresh=self.nms_overlap, scores=scores)
        return scale_boxes(kept, factor) if factor != 1.0 else kept

    def annotate(self, image: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
        """Return a copy of ``image`` with boxes and a running count drawn on."""
        out = image.copy()
        thickness = max(2, int(round(max(out.shape[:2]) / 400)))
        font_scale = max(0.6, max(out.shape[:2]) / 1000.0)
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOUR, thickness)
        label = f"People: {len(boxes)}"
        cv2.putText(out, label, (10, int(30 * font_scale) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)   # outline
        cv2.putText(out, label, (10, int(30 * font_scale) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out


# --------------------------------------------------------------------------
# File-level helpers used by the CLI
# --------------------------------------------------------------------------
def process_image_file(input_path: str, output_path: str, counter: PeopleCounter) -> int:
    """Count people in an image file and save an annotated copy."""
    image = load_image(input_path)
    boxes = counter.detect(image)
    save_image(output_path, counter.annotate(image, boxes))
    logger.info("Image '%s': %d person(s) detected -> %s", input_path, len(boxes), output_path)
    return len(boxes)


def process_video_file(input_path: str, output_path: str, counter: PeopleCounter,
                       frame_skip: int = 2) -> Tuple[int, float]:
    """Count people in a video file. Returns ``(max_count, average_count)``.

    People are *detected* on every Nth frame; the last boxes are re-used for
    drawing on the frames in between. Statistics use analysed frames only.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    cap, fps, size = open_video(input_path)
    writer = None
    frames_read = frames_processed = total = max_count = 0
    last_boxes: List[Box] = []
    try:
        writer = create_video_writer(output_path, fps, size)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames_read % frame_skip == 0:
                last_boxes = counter.detect(frame)
                total += len(last_boxes)
                max_count = max(max_count, len(last_boxes))
                frames_processed += 1
            writer.write(counter.annotate(frame, last_boxes))
            frames_read += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_read == 0:
        raise ValueError(f"No frames could be read from '{input_path}'.")
    avg = total / frames_processed if frames_processed else 0.0
    logger.info("Video '%s': %d frames (%d analysed), max %d, avg %.2f people -> %s",
                input_path, frames_read, frames_processed, max_count, avg, output_path)
    return max_count, avg
