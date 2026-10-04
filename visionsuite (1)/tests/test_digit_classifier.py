import os

import cv2
import numpy as np
import pytest
from sklearn.datasets import load_digits

from src.digit_classifier.model_utils import (
    build_model, load_model, preprocess_digit_image, save_model,
)
from src.digit_classifier.predict import predict_digit
from src.digit_classifier.train import train_and_evaluate


def _digit_as_photo(index=0, dark_on_light=True, up=16):
    small = load_digits().images[index].astype(np.float32)
    big = np.clip(cv2.resize(small, (8 * up, 8 * up), interpolation=cv2.INTER_CUBIC), 0, 16)
    gray = np.round(big / 16.0 * 255).astype(np.uint8)       # bright ink on dark
    return 255 - gray if dark_on_light else gray


def test_preprocess_returns_64_features_in_range():
    features = preprocess_digit_image(_digit_as_photo())
    assert features.shape == (1, 64)
    assert features.min() >= 0 and abs(features.max() - 16.0) < 1e-6


def test_preprocess_handles_both_ink_polarities():
    dark_ink = preprocess_digit_image(_digit_as_photo(dark_on_light=True))
    light_ink = preprocess_digit_image(_digit_as_photo(dark_on_light=False))
    assert np.allclose(dark_ink, light_ink, atol=1e-3)


def test_preprocess_accepts_colour_images():
    gray = _digit_as_photo()
    assert preprocess_digit_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)).shape == (1, 64)


def test_preprocess_rejects_blank_image():
    with pytest.raises(ValueError):
        preprocess_digit_image(np.full((64, 64), 255, np.uint8))


def test_model_save_and_load_roundtrip(tmp_path):
    digits = load_digits()
    model = build_model().fit(digits.data[:300], digits.target[:300])
    path = save_model(model, str(tmp_path / "models" / "m.pkl"), metadata={"accuracy": 1.0})
    loaded = load_model(path)
    assert (loaded.predict(digits.data[300:320]) == model.predict(digits.data[300:320])).all()


def test_load_model_missing_file_gives_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_model(str(tmp_path / "nope.pkl"))
    assert "train-digits" in str(info.value)


def test_load_model_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ValueError):
        load_model(str(bad))


def test_predict_without_trained_model_raises(tmp_path):
    img = str(tmp_path / "d.png")
    cv2.imwrite(img, _digit_as_photo())
    with pytest.raises(FileNotFoundError):
        predict_digit(img, model_path=str(tmp_path / "missing.pkl"))


def test_train_rejects_invalid_test_size(tmp_path):
    with pytest.raises(ValueError):
        train_and_evaluate(test_size=1.2, model_out_path=str(tmp_path / "m.pkl"), report_dir=str(tmp_path))


@pytest.mark.slow
def test_train_and_predict_end_to_end(tmp_path):
    model_path = str(tmp_path / "models" / "digit.pkl")
    result = train_and_evaluate(model_out_path=model_path, report_dir=str(tmp_path / "out"))
    assert result["accuracy"] > 0.95
    for key in ("model_path", "report_path", "cm_path"):
        assert os.path.isfile(result[key]), f"{key} was not created"

    correct = 0
    for index in range(5):
        img = str(tmp_path / f"d{index}.png")
        cv2.imwrite(img, _digit_as_photo(index))
        correct += predict_digit(img, model_path=model_path)["predicted_digit"] == int(load_digits().target[index])
    assert correct >= 4
