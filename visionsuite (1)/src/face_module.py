"""
face_module.py
--------------
Module 1: detect human faces and anonymise (blur) them.

Technique: OpenCV's Haar Cascade classifier (``haarcascade_frontalface_default``),
which ships inside the ``opencv-python`` package -- no download needed.

Privacy by design: this module only *detects and blurs*. It never
recognises or stores who a face belongs to.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.image_preprocessing import (
    Box, blur_region, clamp_box, create_video_writer, load_image, open_video,
    resize_max_dimension, save_image, scale_boxes,
)
from src.logger_setup import get_logger

logger = get_logger(__name__)

CASCADE_FILE = "haarcascade_frontalface_default.xml"
DETECTION_MAX_DIM = 1280     # detect on a downscaled copy for speed
BOX_PADDING = 0.10           # blur 10% beyond each side (forehead / chin)


class FaceDetector:
    """Detects frontal faces and blurs them.

    Parameters (all configurable in ``config.yaml``):
      scale_factor  -- image-pyramid step; smaller = slower but finds more faces
      min_neighbors -- overlapping hits needed to accept a face; higher = fewer
                       false positives but may miss real faces
      blur_kernel   -- Gaussian kernel size (odd); larger = stronger blur
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 blur_kernel: int = 35, min_face_size: Tuple[int, int] = (30, 30)):
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 1:
            raise ValueError("min_neighbors must be >= 1")
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.blur_kernel = blur_kernel
        self.min_face_size = min_face_size

        cascade_path = cv2.data.haarcascades + CASCADE_FILE
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load the Haar cascade file: {cascade_path}")

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Box]:
        """Return face boxes ``(x, y, w, h)`` in the coordinates of ``image``."""
        small, scale = resize_max_dimension(image, DETECTION_MAX_DIM)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.equalizeHist(gray)
        found = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )
        boxes = [tuple(int(v) for v in b) for b in found]
        return scale_boxes(boxes, scale) if scale != 1.0 else boxes

    def blur_faces(self, image: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
        """Return a copy of ``image`` with every box blurred."""
        h, w = image.shape[:2]
        out = image.copy()
        for (x, y, bw, bh) in boxes:
            pad_x, pad_y = int(bw * BOX_PADDING), int(bh * BOX_PADDING)
            padded = clamp_box((x - pad_x, y - pad_y, bw + 2 * pad_x, bh + 2 * pad_y), w, h)
            out = blur_region(out, padded, self.blur_kernel)
        return out

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List[Box]]:
        """Detect + blur in one call. Returns ``(blurred_image, boxes)``."""
        boxes = self.detect(image)
        return self.blur_faces(image, boxes), boxes


# --------------------------------------------------------------------------
# File-level helpers used by the CLI
# --------------------------------------------------------------------------
def process_image_file(input_path: str, output_path: str, detector: FaceDetector) -> int:
    """Blur all faces in an image file. Returns the number of faces found."""
    image = load_image(input_path)
    blurred, boxes = detector.process_frame(image)
    save_image(output_path, blurred)
    logger.info("Image '%s': %d face(s) blurred -> %s", input_path, len(boxes), output_path)
    return len(boxes)


def process_video_file(input_path: str, output_path: str, detector: FaceDetector,
                       frame_skip: int = 1) -> int:
    """Blur all faces in a video file. Returns total detections.

    With ``frame_skip = N`` faces are *detected* on every Nth frame only. The
    boxes from the last detection are re-used (and blurred) on the frames in
    between, so the output keeps its full length and no frame is left with an
    un-blurred face just because it was skipped.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    cap, fps, size = open_video(input_path)
    writer = None
    total_faces = frames_read = frames_processed = 0
    last_boxes: List[Box] = []
    try:
        writer = create_video_writer(output_path, fps, size)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames_read % frame_skip == 0:
                last_boxes = detector.detect(frame)
                total_faces += len(last_boxes)
                frames_processed += 1
            writer.write(detector.blur_faces(frame, last_boxes))
            frames_read += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_read == 0:
        raise ValueError(f"No frames could be read from '{input_path}'.")
    logger.info("Video '%s': %d frames (%d analysed), %d face detections -> %s",
                input_path, frames_read, frames_processed, total_faces, output_path)
    return total_faces
