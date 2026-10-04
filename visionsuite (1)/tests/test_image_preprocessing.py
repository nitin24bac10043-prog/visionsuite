import cv2
import numpy as np
import pytest

from src.image_preprocessing import (
    blur_region, clamp_box, load_image, non_max_suppression,
    resize_max_dimension, scale_boxes,
)


def test_resize_max_dimension_downscales_large_images():
    image = np.zeros((1000, 2000, 3), np.uint8)
    small, scale = resize_max_dimension(image, 1000)
    assert small.shape[:2] == (500, 1000)
    assert scale == 0.5


def test_resize_never_upscales_small_images():
    image = np.zeros((100, 200, 3), np.uint8)
    same, scale = resize_max_dimension(image, 1000)
    assert same.shape == image.shape and scale == 1.0


def test_scale_boxes_maps_back_to_original_size():
    assert scale_boxes([(10, 20, 30, 40)], 0.5) == [(20, 40, 60, 80)]


def test_clamp_box_keeps_box_inside_image():
    assert clamp_box((-10, -10, 50, 50), 100, 100) == (0, 0, 40, 40)
    assert clamp_box((90, 90, 50, 50), 100, 100) == (90, 90, 10, 10)


def test_nms_merges_heavily_overlapping_boxes():
    boxes = [(10, 10, 100, 200), (14, 12, 100, 200), (8, 9, 100, 200)]
    assert len(non_max_suppression(boxes)) == 1


def test_nms_keeps_separate_boxes():
    boxes = [(0, 0, 50, 100), (300, 0, 50, 100)]
    assert len(non_max_suppression(boxes)) == 2


def test_nms_keeps_highest_scoring_box():
    boxes = [(10, 10, 100, 200), (12, 10, 100, 200)]
    kept = non_max_suppression(boxes, scores=[0.2, 0.9])
    assert kept == [(12, 10, 100, 200)]


def test_nms_empty_input_returns_empty_list():
    assert non_max_suppression([]) == []


def test_blur_region_changes_only_the_box():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_region(image, (20, 20, 40, 40), kernel=15)
    assert not np.array_equal(out[20:60, 20:60], image[20:60, 20:60])   # blurred
    assert np.array_equal(out[:20], image[:20])                          # untouched
    assert np.array_equal(out[60:], image[60:])
    assert out is not image                                              # input not mutated


def test_blur_region_rejects_even_kernel():
    with pytest.raises(ValueError):
        blur_region(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), kernel=4)


def test_load_image_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "missing.png"))
    fake = tmp_path / "fake.png"
    fake.write_text("this is not an image")
    with pytest.raises(ValueError):
        load_image(str(fake))


def test_load_image_reads_valid_file(tmp_path):
    path = str(tmp_path / "ok.png")
    cv2.imwrite(path, np.full((20, 30, 3), 200, np.uint8))
    assert load_image(path).shape == (20, 30, 3)
