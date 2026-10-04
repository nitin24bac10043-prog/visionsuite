"""
predict.py
----------
Module 3 (part 2): classify a single handwritten-digit image with the model
trained by ``python main.py train-digits``.

The image should contain ONE roughly-centred digit (like the training data).
"""

import os
from typing import Any, Dict

import numpy as np

from src.digit_classifier.model_utils import (
    load_digit_image, load_model, preprocess_digit_image,
)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def predict_digit(image_path: str,
                  model_path: str = os.path.join("models", "digit_classifier.pkl")) -> Dict[str, Any]:
    """Return ``{"predicted_digit": int, "confidence": float, "probabilities": [...]}``.

    Raises ``FileNotFoundError`` if the image or the trained model is missing
    and ``ValueError`` if the image cannot be read or contains no digit.
    """
    image = load_digit_image(image_path)
    features = preprocess_digit_image(image)
    model = load_model(model_path)

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    best = int(np.argmax(probabilities))

    result = {
        "predicted_digit": int(classes[best]),
        "confidence": float(probabilities[best]),
        "probabilities": [round(float(p), 4) for p in probabilities],
    }
    logger.info("Predicted digit %d (confidence %.2f%%) for '%s'",
                result["predicted_digit"], result["confidence"] * 100, image_path)
    return result
