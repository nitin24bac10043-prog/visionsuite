#!/usr/bin/env python3
"""
main.py
-------
VisionSuite command-line interface.

VisionSuite is a modular computer-vision analysis toolkit with three
functional modules, all reachable from this single CLI entry point:

    1. face-blur     -- detect and anonymise (blur) faces in an image/video
    2. count-people   -- detect and count pedestrians in an image/video
    3. train-digits   -- train + evaluate the handwritten digit classifier
    4. predict-digit   -- classify a single handwritten digit image

Run `python main.py --help` or `python main.py <command> --help` for details.
"""

import argparse
import os
import sys
import time

from src.config_loader import AppConfig, ConfigError
from src.face_module import FaceDetector
from src.face_module import process_image_file as face_process_image
from src.face_module import process_video_file as face_process_video
from src.logger_setup import get_logger
from src.people_counter_module import PeopleCounter
from src.people_counter_module import process_image_file as people_process_image
from src.people_counter_module import process_video_file as people_process_video
from src.report_generator import RunReport

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _validate_input_path(path: str) -> None:
    """Security/robustness check: reject missing files early with a clear
    error rather than letting OpenCV fail with a cryptic message."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file does not exist: '{path}'")


def cmd_face_blur(args, config: AppConfig) -> int:
    _validate_input_path(args.input)
    os.makedirs(config.output_dir, exist_ok=True)

    detector = FaceDetector(
        scale_factor=config.face_scale_factor,
        min_neighbors=config.face_min_neighbors,
        blur_kernel=config.face_blur_kernel,
    )

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    ext = ".mp4" if _is_video(args.input) else ".png"
    output_path = args.output or os.path.join(config.output_dir, f"{base_name}_face_blurred{ext}")

    report = RunReport(command="face-blur", input_path=args.input, output_dir=config.output_dir)
    start = time.time()
    try:
        if _is_video(args.input):
            total_faces = face_process_video(args.input, output_path, detector,
                                              frame_skip=args.frame_skip)
            report.add_result("total_face_detections", total_faces)
        else:
            n_faces = face_process_image(args.input, output_path, detector)
            report.add_result("faces_detected", n_faces)
    except (ValueError, IOError, RuntimeError) as exc:
        report.add_error(str(exc))
        _finalize_report(report, config, time.time() - start)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report.add_result("output_path", output_path)
    _finalize_report(report, config, time.time() - start)
    print(f"Done. Output written to: {output_path}")
    return 0


def cmd_count_people(args, config: AppConfig) -> int:
    _validate_input_path(args.input)
    os.makedirs(config.output_dir, exist_ok=True)

    counter = PeopleCounter(
        hit_threshold=config.people_hit_threshold,
        win_stride=config.people_win_stride,
    )

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    ext = ".mp4" if _is_video(args.input) else ".png"
    output_path = args.output or os.path.join(config.output_dir, f"{base_name}_people_counted{ext}")

    report = RunReport(command="count-people", input_path=args.input, output_dir=config.output_dir)
    start = time.time()
    try:
        if _is_video(args.input):
            max_count, avg_count = people_process_video(args.input, output_path, counter,
                                                          frame_skip=args.frame_skip)
            report.add_result("max_people_in_frame", max_count)
            report.add_result("avg_people_per_frame", round(avg_count, 2))
        else:
            count = people_process_image(args.input, output_path, counter)
            report.add_result("people_detected", count)
    except (ValueError, IOError) as exc:
        report.add_error(str(exc))
        _finalize_report(report, config, time.time() - start)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report.add_result("output_path", output_path)
    _finalize_report(report, config, time.time() - start)
    print(f"Done. Output written to: {output_path}")
    return 0


def cmd_train_digits(args, config: AppConfig) -> int:
    from src.digit_classifier.train import train_and_evaluate

    report = RunReport(command="train-digits", input_path="sklearn.datasets.load_digits",
                        output_dir=config.output_dir)
    start = time.time()
    try:
        result = train_and_evaluate(
            test_size=config.digit_test_size,
            random_state=config.digit_random_state,
            model_out_path=config.digit_model_path,
            report_dir=config.output_dir,
        )
        report.add_result("accuracy", round(result["accuracy"], 4))
        report.add_result("model_path", result["model_path"])
        report.add_result("evaluation_report", result["report_path"])
        report.add_result("confusion_matrix_image", result["cm_path"])
    except Exception as exc:  # noqa: BLE001 - top level CLI boundary
        report.add_error(str(exc))
        _finalize_report(report, config, time.time() - start)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _finalize_report(report, config, time.time() - start)
    print(f"Training complete. Test accuracy: {result['accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    return 0


def cmd_predict_digit(args, config: AppConfig) -> int:
    from src.digit_classifier.predict import predict_digit

    _validate_input_path(args.input)
    report = RunReport(command="predict-digit", input_path=args.input, output_dir=config.output_dir)
    start = time.time()
    try:
        result = predict_digit(args.input, model_path=config.digit_model_path)
    except (FileNotFoundError, ValueError) as exc:
        report.add_error(str(exc))
        _finalize_report(report, config, time.time() - start)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report.add_result("predicted_digit", result["predicted_digit"])
    report.add_result("confidence", round(result["confidence"], 4))
    _finalize_report(report, config, time.time() - start)
    print(f"Predicted digit: {result['predicted_digit']} "
          f"(confidence: {result['confidence'] * 100:.2f}%)")
    return 0


def _finalize_report(report: RunReport, config: AppConfig, elapsed_seconds: float) -> None:
    report.add_result("elapsed_seconds", round(elapsed_seconds, 3))
    json_path = os.path.join(config.output_dir, "last_run_report.json")
    csv_path = os.path.join(config.output_dir, "run_history.csv")
    report.save_json(json_path)
    report.append_csv(csv_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionsuite",
        description="VisionSuite -- a modular computer-vision analysis CLI toolkit.",
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a YAML config file (default: ./config.yaml).")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_face = subparsers.add_parser("face-blur", help="Detect and blur faces in an image/video.")
    p_face.add_argument("--input", required=True, help="Path to input image or video.")
    p_face.add_argument("--output", default=None, help="Path to write the annotated output.")
    p_face.add_argument("--frame-skip", type=int, default=1,
                         help="Process every Nth frame for video input (default: 1).")
    p_face.set_defaults(func=cmd_face_blur)

    p_people = subparsers.add_parser("count-people", help="Detect and count people in an image/video.")
    p_people.add_argument("--input", required=True, help="Path to input image or video.")
    p_people.add_argument("--output", default=None, help="Path to write the annotated output.")
    p_people.add_argument("--frame-skip", type=int, default=2,
                           help="Process every Nth frame for video input (default: 2).")
    p_people.set_defaults(func=cmd_count_people)

    p_train = subparsers.add_parser("train-digits", help="Train + evaluate the digit classifier.")
    p_train.set_defaults(func=cmd_train_digits)

    p_predict = subparsers.add_parser("predict-digit", help="Classify a handwritten digit image.")
    p_predict.add_argument("--input", required=True, help="Path to a digit image (PNG/JPG).")
    p_predict.set_defaults(func=cmd_predict_digit)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = AppConfig.load(args.config) if args.config else AppConfig.load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger.info("VisionSuite started. Command: %s", args.command)
    try:
        exit_code = args.func(args, config)
    except FileNotFoundError as exc:
        # Catches _validate_input_path() failures and any other missing-file
        # errors raised before a RunReport could be constructed, so the CLI
        # always fails gracefully with a clear message instead of a traceback.
        logger.error(str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI safety net
        logger.exception("Unexpected error while running command '%s'.", args.command)
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    logger.info("VisionSuite finished with exit code %d.", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
