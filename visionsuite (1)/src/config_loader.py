"""
config_loader.py
----------------
Loads and *validates* config.yaml into a typed, immutable ``AppConfig``.

Every value is range-checked so that a typo in the YAML file produces a
clear ``ConfigError`` ("face_detection.blur_kernel must be a positive odd
integer") instead of a cryptic OpenCV crash later on.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from src.logger_setup import VALID_LEVELS, configure_logging, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed or invalid."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class AppConfig:
    output_dir: str = "outputs"
    log_level: str = "INFO"

    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_blur_kernel: int = 35

    people_hit_threshold: float = 0.0
    people_win_stride: int = 8

    digit_model_path: str = os.path.join("models", "digit_classifier.pkl")
    digit_test_size: float = 0.2
    digit_random_state: int = 42

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None, setup_logging: bool = True) -> "AppConfig":
        """Load config from ``path`` (or the project's config.yaml).

        * An explicitly given path that does not exist -> ``ConfigError``.
        * A missing *default* config.yaml -> built-in defaults are used.
        """
        explicit = path is not None
        cfg_path = path if explicit else DEFAULT_CONFIG_PATH
        found = os.path.isfile(cfg_path)

        if not found:
            if explicit:
                raise ConfigError(f"Config file not found: '{cfg_path}'")
            data: Any = {}
        else:
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Config file '{cfg_path}' is not valid YAML: {exc}") from exc
            except OSError as exc:
                raise ConfigError(f"Could not read config file '{cfg_path}': {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigError("Config file must contain a YAML mapping (key: value pairs).")

        config = cls._from_dict(data)

        if setup_logging:
            configure_logging(os.path.join(config.output_dir, "logs"), config.log_level)
        logger.info("Configuration loaded from %s", cfg_path if found else "built-in defaults")
        return config

    # ------------------------------------------------------------------
    @classmethod
    def _from_dict(cls, data: dict) -> "AppConfig":
        d = cls()  # defaults

        def section(name: str) -> dict:
            value = data.get(name) or {}
            if not isinstance(value, dict):
                raise ConfigError(f"'{name}' must be a section of key: value pairs.")
            return value

        general = section("general")
        face = section("face_detection")
        people = section("people_detection")
        digit = section("digit_classifier")

        output_dir = general.get("output_dir", d.output_dir)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigError("general.output_dir must be a non-empty string.")

        log_level = str(general.get("log_level", d.log_level)).upper()
        if log_level not in VALID_LEVELS:
            raise ConfigError(f"general.log_level must be one of {', '.join(VALID_LEVELS)}.")

        scale_factor = face.get("scale_factor", d.face_scale_factor)
        if not _is_number(scale_factor) or scale_factor <= 1.0:
            raise ConfigError("face_detection.scale_factor must be a number greater than 1.0.")

        min_neighbors = face.get("min_neighbors", d.face_min_neighbors)
        if not _is_int(min_neighbors) or min_neighbors < 1:
            raise ConfigError("face_detection.min_neighbors must be an integer >= 1.")

        blur_kernel = face.get("blur_kernel", d.face_blur_kernel)
        if not _is_int(blur_kernel) or blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ConfigError("face_detection.blur_kernel must be a positive odd integer (e.g. 35).")

        hit_threshold = people.get("hit_threshold", d.people_hit_threshold)
        if not _is_number(hit_threshold):
            raise ConfigError("people_detection.hit_threshold must be a number.")

        win_stride = people.get("win_stride", d.people_win_stride)
        if not _is_int(win_stride) or win_stride < 1:
            raise ConfigError("people_detection.win_stride must be an integer >= 1.")

        model_path = digit.get("model_path", d.digit_model_path)
        if not isinstance(model_path, str) or not model_path.strip():
            raise ConfigError("digit_classifier.model_path must be a non-empty string.")

        test_size = digit.get("test_size", d.digit_test_size)
        if not _is_number(test_size) or not (0.0 < test_size < 1.0):
            raise ConfigError("digit_classifier.test_size must be a number between 0 and 1 (exclusive).")

        random_state = digit.get("random_state", d.digit_random_state)
        if not _is_int(random_state):
            raise ConfigError("digit_classifier.random_state must be an integer.")

        return cls(
            output_dir=output_dir,
            log_level=log_level,
            face_scale_factor=float(scale_factor),
            face_min_neighbors=min_neighbors,
            face_blur_kernel=blur_kernel,
            people_hit_threshold=float(hit_threshold),
            people_win_stride=win_stride,
            digit_model_path=model_path,
            digit_test_size=float(test_size),
            digit_random_state=random_state,
        )
