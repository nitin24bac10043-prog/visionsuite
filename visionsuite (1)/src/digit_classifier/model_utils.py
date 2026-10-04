"""
model_utils.py
--------------
Helpers shared by training and prediction for the handwritten-digit
classifier.

Model choice -- RBF-kernel Support Vector Machine (SVM)
    * The scikit-learn ``digits`` dataset is small (1,797 images of 8x8
      pixels = 64 features), where an SVM is both very accurate (~99%) and
      trains in under a second on any laptop -- no GPU needed.
    * ``C=10`` and ``gamma=0.001`` are standard, well-tested values for this
      dataset (pixel values range 0..16).

Preprocessing convention (must match the training data!)
    * The dataset images are 8x8, values 0..16, where **16 = ink** and
      0 = empty paper.
    * A normal photo / scan is usually *dark ink on a light background*, so
      we detect that case and invert it.
"""

import os
import pickle
from typing import Any, Dict, Optional

import cv2
import numpy as np
from sklearn.svm import SVC

DIGIT_SIZE = 8          # images are 8 x 8
MAX_PIXEL_VALUE = 16.0  # sklearn digits pixel range is 0..16


def build_model(random_state: int = 42) -> SVC:
    """Create an (untrained) RBF-kernel SVM with probability estimates."""
    return SVC(kernel="rbf", C=10, gamma=0.001, probability=True, random_state=random_state)


# --------------------------------------------------------------------------
# Image -> feature vector
# --------------------------------------------------------------------------
def load_digit_image(path: str) -> np.ndarray:
    """Read an image file as a grayscale array, or raise a clear error."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Digit image does not exist: '{path}'")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read '{path}' as an image (unsupported or corrupt file).")
    return image


def preprocess_digit_image(image: np.ndarray) -> np.ndarray:
    """Convert any grayscale/BGR image into a ``(1, 64)`` feature vector.

    Steps: grayscale -> make ink bright (invert dark-on-light images) ->
    stretch contrast -> shrink to 8x8 -> scale to the 0..16 range.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image.astype(np.float32)

    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        raise ValueError("The image has no contrast (it looks blank), so there is no digit to classify.")

    # Dark ink on light paper is the common case: the *median* pixel is the
    # background, so a bright median means we must invert.
    if np.median(gray) > (lo + hi) / 2.0:
        gray = hi - (gray - lo)
    gray = (gray - gray.min()) / max(float(gray.max() - gray.min()), 1e-6)   # 0..1, ink = 1

    small = cv2.resize(gray, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA)
    peak = float(small.max())
    if peak <= 0:
        raise ValueError("No digit strokes were found in the image.")
    small = small / peak * MAX_PIXEL_VALUE
    return small.reshape(1, -1).astype(np.float64)


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------
def save_model(model: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Pickle ``model`` (plus optional metadata) to ``path``; returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "metadata": metadata or {}}, fh)
    return path


def load_model(path: str) -> Any:
    """Load a model saved by :func:`save_model`.

    Raises ``FileNotFoundError`` with a helpful hint if it has not been
    trained yet, and ``ValueError`` if the file is not a valid model file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. "
            f"Train one first with: python main.py train-digits"
        )
    try:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        return bundle["model"]
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is a bad file
        raise ValueError(f"'{path}' is not a valid VisionSuite model file: {exc}") from exc
