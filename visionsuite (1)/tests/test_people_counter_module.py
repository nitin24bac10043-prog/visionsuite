import cv2
import numpy as np
import pytest

from src.people_counter_module import PeopleCounter, process_image_file, process_video_file


def test_rejects_invalid_parameters():
    for kwargs in ({"win_stride": 0}, {"scale": 1.0}, {"nms_overlap": 0}):
        with pytest.raises(ValueError):
            PeopleCounter(**kwargs)


def test_blank_image_has_zero_people():
    assert PeopleCounter().detect(np.full((300, 300, 3), 128, np.uint8)) == []


def test_annotate_draws_boxes_without_modifying_the_input():
    image = np.zeros((200, 200, 3), np.uint8)
    out = PeopleCounter().annotate(image, [(50, 50, 40, 90)])
    assert image.sum() == 0              # original untouched
    assert out.sum() > 0                 # something was drawn


def test_process_image_file_saves_annotated_copy(tmp_path):
    src = str(tmp_path / "in.png")
    cv2.imwrite(src, np.full((240, 320, 3), 128, np.uint8))
    dst = str(tmp_path / "out" / "counted.png")
    assert process_image_file(src, dst, PeopleCounter()) == 0
    assert cv2.imread(dst) is not None


def test_video_statistics_on_empty_scene(tmp_path):
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(6):
        writer.write(np.full((120, 160, 3), 100, np.uint8))
    writer.release()
    assert process_video_file(src, dst, PeopleCounter(), frame_skip=2) == (0, 0.0)


def test_video_rejects_invalid_frame_skip(tmp_path):
    with pytest.raises(ValueError):
        process_video_file(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"), PeopleCounter(), frame_skip=0)
