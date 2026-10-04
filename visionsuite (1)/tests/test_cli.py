"""End-to-end tests of the command-line interface (main.py)."""
import json
import sys
from unittest import mock

import cv2
import numpy as np

import main as cli


def _config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"general:\n  output_dir: {(tmp_path / 'out').as_posix()}\n"
        f"digit_classifier:\n  model_path: {(tmp_path / 'm.pkl').as_posix()}\n",
        encoding="utf-8",
    )
    return str(path)


def _run(*argv):
    with mock.patch.object(sys, "argv", ["visionsuite", *argv]):
        return cli.main()


def test_missing_input_file_returns_exit_code_1(tmp_path):
    assert _run("--config", _config(tmp_path), "face-blur", "--input", str(tmp_path / "nope.jpg")) == 1


def test_bad_config_path_returns_exit_code_2(tmp_path):
    img = tmp_path / "x.png"
    cv2.imwrite(str(img), np.zeros((20, 20, 3), np.uint8))
    assert _run("--config", str(tmp_path / "missing.yaml"), "count-people", "--input", str(img)) == 2


def test_face_blur_end_to_end_writes_output_and_report(tmp_path):
    img = tmp_path / "scene.png"
    cv2.imwrite(str(img), np.full((240, 320, 3), 170, np.uint8))
    assert _run("--config", _config(tmp_path), "face-blur", "--input", str(img)) == 0

    assert (tmp_path / "out" / "scene_face_blurred.png").is_file()
    report = json.loads((tmp_path / "out" / "last_run_report.json").read_text())
    assert report["status"] == "success" and report["results"]["faces_detected"] == 0
    assert (tmp_path / "out" / "run_history.csv").is_file()


def test_count_people_end_to_end(tmp_path):
    img = tmp_path / "street.png"
    cv2.imwrite(str(img), np.full((240, 320, 3), 100, np.uint8))
    assert _run("--config", _config(tmp_path), "count-people", "--input", str(img)) == 0
    report = json.loads((tmp_path / "out" / "last_run_report.json").read_text())
    assert report["results"]["people_detected"] == 0


def test_predict_digit_without_model_fails_cleanly(tmp_path):
    img = tmp_path / "d.png"
    cv2.imwrite(str(img), np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8))
    assert _run("--config", _config(tmp_path), "predict-digit", "--input", str(img)) == 1
    report = json.loads((tmp_path / "out" / "last_run_report.json").read_text())
    assert report["status"] == "failed"


def test_unknown_command_is_rejected_by_argparse():
    with mock.patch.object(sys, "argv", ["visionsuite", "do-magic"]):
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("argparse should have exited")
