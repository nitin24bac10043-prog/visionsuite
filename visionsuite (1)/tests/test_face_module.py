import cv2
import numpy as np
import pytest

from src.face_module import FaceDetector, process_image_file, process_video_file


def _blank_image(path, size=(240, 320)):
    cv2.imwrite(path, np.full((*size, 3), 180, np.uint8))
    return path


def test_rejects_invalid_parameters():
    for kwargs in ({"scale_factor": 1.0}, {"min_neighbors": 0}, {"blur_kernel": 10}):
        with pytest.raises(ValueError):
            FaceDetector(**kwargs)


def test_blank_image_has_no_faces():
    assert FaceDetector().detect(np.full((240, 320, 3), 180, np.uint8)) == []


def test_blur_faces_blurs_only_the_given_box():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = FaceDetector(blur_kernel=15).blur_faces(image, [(80, 80, 40, 40)])
    assert not np.array_equal(out[90:110, 90:110], image[90:110, 90:110])
    assert np.array_equal(out[:50], image[:50])           # far from the face: untouched


def test_process_image_file_writes_output(tmp_path):
    src = _blank_image(str(tmp_path / "in.png"))
    dst = str(tmp_path / "nested" / "out.png")
    assert process_image_file(src, dst, FaceDetector()) == 0
    assert cv2.imread(dst) is not None


def test_process_image_file_rejects_corrupt_input(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("nope")
    with pytest.raises(ValueError):
        process_image_file(str(bad), str(tmp_path / "out.png"), FaceDetector())


def test_video_rejects_invalid_frame_skip(tmp_path):
    with pytest.raises(ValueError):
        process_video_file(str(tmp_path / "x.mp4"), str(tmp_path / "y.mp4"), FaceDetector(), frame_skip=0)


def test_video_processing_keeps_every_frame(tmp_path):
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(12):
        writer.write(np.full((120, 160, 3), 150, np.uint8))
    writer.release()

    assert process_video_file(src, dst, FaceDetector(), frame_skip=3) == 0
    cap = cv2.VideoCapture(dst)
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames == 12          # frame_skip must not shorten the video
