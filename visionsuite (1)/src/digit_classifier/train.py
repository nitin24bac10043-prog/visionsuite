"""
train.py
--------
Module 3 (part 1): train and evaluate the handwritten-digit classifier.

Evaluation methodology
    * 80/20 stratified train/test split with a fixed random seed, so every
      run is reproducible and every digit class appears in both sets in the
      same proportion.
    * The model never sees the test set during training, so the reported
      accuracy is an honest estimate of performance on new digits.
    * Reported metrics: overall accuracy, per-class precision / recall / F1,
      and a confusion matrix (text + PNG image).
"""

import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")   # no display needed (works on servers / Codespaces)
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.digit_classifier.model_utils import build_model, save_model  # noqa: E402
from src.logger_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

REPORT_FILE = "digit_classifier_evaluation.txt"
CM_FILE = "digit_classifier_confusion_matrix.png"


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42,
                       model_out_path: str = os.path.join("models", "digit_classifier.pkl"),
                       report_dir: str = "outputs") -> Dict[str, Any]:
    """Train the SVM, evaluate it on held-out data, and save everything.

    Returns a dict with: accuracy, model_path, report_path, cm_path,
    n_train, n_test.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test_size,
        random_state=random_state, stratify=digits.target,
    )
    logger.info("Training digit classifier on %d samples (testing on %d).", len(x_train), len(x_test))

    model = build_model(random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report_text = classification_report(y_test, predictions, digits=4)
    cm = confusion_matrix(y_test, predictions)

    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("VisionSuite - Digit classifier evaluation\n")
        fh.write("=" * 44 + "\n")
        fh.write(f"Model        : RBF-kernel SVM (C=10, gamma=0.001)\n")
        fh.write(f"Dataset      : sklearn load_digits ({len(digits.data)} samples, 8x8 pixels)\n")
        fh.write(f"Train / test : {len(x_train)} / {len(x_test)} (test_size={test_size}, seed={random_state})\n")
        fh.write(f"Accuracy     : {accuracy:.4f}\n\n")
        fh.write("Per-class precision / recall / F1\n")
        fh.write(report_text + "\n")
        fh.write("Confusion matrix (rows = true digit, columns = predicted digit)\n")
        fh.write(str(cm) + "\n")

    cm_path = os.path.join(report_dir, CM_FILE)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Digit classifier confusion matrix (accuracy {accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    model_path = save_model(model, model_out_path, metadata={
        "accuracy": accuracy, "test_size": test_size, "random_state": random_state,
        "n_train": int(len(x_train)), "n_test": int(len(x_test)),
    })
    logger.info("Training finished. Accuracy=%.4f, model saved to %s", accuracy, model_path)

    return {
        "accuracy": accuracy,
        "model_path": model_path,
        "report_path": report_path,
        "cm_path": cm_path,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
