import pytest

from src.config_loader import DEFAULT_CONFIG_PATH, AppConfig, ConfigError


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_loads_valid_config(tmp_path):
    path = _write(tmp_path, "general:\n  output_dir: out\n  log_level: debug\n"
                            "face_detection:\n  blur_kernel: 51\n")
    cfg = AppConfig.load(path, setup_logging=False)
    assert cfg.output_dir == "out"
    assert cfg.log_level == "DEBUG"          # normalised to upper case
    assert cfg.face_blur_kernel == 51


def test_missing_sections_fall_back_to_defaults(tmp_path):
    cfg = AppConfig.load(_write(tmp_path, "general:\n  output_dir: out\n"), setup_logging=False)
    assert cfg.face_min_neighbors == 5
    assert cfg.digit_test_size == 0.2


def test_explicit_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        AppConfig.load(str(tmp_path / "does_not_exist.yaml"), setup_logging=False)


def test_malformed_yaml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        AppConfig.load(_write(tmp_path, "general: [unclosed"), setup_logging=False)


def test_non_mapping_yaml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        AppConfig.load(_write(tmp_path, "- just\n- a list\n"), setup_logging=False)


def test_invalid_values_are_rejected(tmp_path):
    bad_snippets = [
        "face_detection:\n  blur_kernel: 34\n",       # must be odd
        "face_detection:\n  scale_factor: 1.0\n",      # must be > 1
        "face_detection:\n  min_neighbors: 0\n",
        "people_detection:\n  win_stride: 0\n",
        "digit_classifier:\n  test_size: 1.5\n",
        "general:\n  log_level: LOUD\n",
    ]
    for snippet in bad_snippets:
        with pytest.raises(ConfigError):
            AppConfig.load(_write(tmp_path, snippet), setup_logging=False)


def test_project_config_yaml_is_valid():
    cfg = AppConfig.load(DEFAULT_CONFIG_PATH, setup_logging=False)
    assert cfg.face_blur_kernel % 2 == 1
    assert 0 < cfg.digit_test_size < 1
