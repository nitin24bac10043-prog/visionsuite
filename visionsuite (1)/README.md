# VisionSuite

**A modular, offline, command-line computer-vision analysis toolkit** --
face anonymisation, pedestrian counting, and handwritten-digit
classification, all from a single CLI, with structured logging and
audit-friendly run reports.

> Built as a Computer Vision course project. See [`statement.md`](statement.md)
> for the full problem statement, scope, and target users.

---

## Overview

VisionSuite provides three independent but consistently-designed computer
vision capabilities, reachable through one command-line entry point
(`main.py`):

| Module | Command | What it does | Technique used |
|---|---|---|---|
| **Face Blur** | `face-blur` | Detects faces in an image/video and blurs them for privacy | Haar Cascade classifier (OpenCV) |
| **People Counter** | `count-people` | Detects and counts pedestrians in an image/video | HOG descriptor + linear SVM (OpenCV) |
| **Digit Classifier** | `train-digits`, `predict-digit` | Trains, evaluates, and runs inference for handwritten digit recognition | RBF-kernel SVM (scikit-learn) |

All three models are bundled with their respective libraries (OpenCV /
scikit-learn) -- **no external dataset or model download is required to run
this project**, which keeps it fully reproducible for evaluation. (Internet is
only needed once, to `pip install` the libraries.)

## Features

- Detect & blur faces in images **and** videos, with adjustable blur
  strength and frame-skip for speed on long videos. Frames skipped by
  `--frame-skip` re-use the last detected boxes, so no frame is left with an
  un-blurred face.
- Detect & count pedestrians in images **and** videos, with non-maximum
  suppression to avoid double-counting overlapping detections.
- Train a handwritten-digit classifier from scratch and see its accuracy,
  per-class precision/recall/F1 report, and a confusion-matrix plot.
- Classify a new handwritten digit image using the trained model.
- Centralised, validated YAML configuration (`config.yaml`) -- tune detection
  sensitivity without touching any code; typos give a clear error message.
- Structured, timestamped run reports (`outputs/last_run_report.json`,
  `outputs/run_history.csv`) and rotating application logs
  (`outputs/logs/visionsuite.log`) after every command.
- Graceful error handling everywhere: missing files, bad config, or a
  missing trained model all produce a clear message and a non-zero exit
  code instead of a stack trace.
- 51 automated tests (`pytest`) covering every module and the CLI.
- Demo data included (and re-creatable with one script) so you can try every
  command immediately, with no external images/videos required.

## Technologies / Tools Used

- **Python 3.10+**
- **OpenCV** (`opencv-python-headless`) -- image/video I/O, Haar Cascade face
  detection, HOG pedestrian detection. The *headless* build is used because
  the tool only reads and writes files (no pop-up windows), which also lets it
  run on servers and cloud machines that have no display.
- **scikit-learn** -- SVM classifier, the `digits` dataset, train/test
  split, evaluation metrics
- **NumPy** -- array/image manipulation
- **Matplotlib** -- confusion-matrix visualisation
- **PyYAML** -- configuration file parsing
- **pytest** -- automated testing
- Standard library: `argparse`, `logging` (with rotation), `dataclasses`,
  `json`, `csv`, `pickle`

## Project Structure

```
visionsuite/
├── main.py                          # CLI entry point (all subcommands)
├── config.yaml                      # tunable runtime configuration
├── requirements.txt
├── pytest.ini
├── statement.md                     # problem statement / scope / users
├── README.md                        # this file
├── src/
│   ├── logger_setup.py              # centralised, rotating logging
│   ├── config_loader.py             # loads + validates config.yaml
│   ├── image_preprocessing.py       # shared CV utilities (resize, NMS, blur, video I/O)
│   ├── face_module.py               # Module 1: face detection & blur
│   ├── people_counter_module.py     # Module 2: pedestrian detection & count
│   ├── report_generator.py          # JSON/CSV run-report writer
│   └── digit_classifier/
│       ├── model_utils.py           # preprocessing, save/load trained model
│       ├── train.py                 # Module 3: train + evaluate SVM
│       └── predict.py               # Module 3: run inference
├── sample_data/
│   ├── generate_sample_media.py     # (re)creates the demo files below
│   ├── sample_scene.png             # synthetic scene (no real people)
│   ├── sample_video.mp4             # short synthetic clip
│   └── sample_digits/               # five handwritten-digit images
├── tests/                           # pytest tests (51)
├── docs/                            # architecture & UML diagrams (Mermaid)
├── models/                          # trained model is saved here (created automatically)
└── outputs/                         # results, reports, logs (created automatically)
```

## Setup & Installation

### Option A -- run in your browser (no installation): GitHub Codespaces

1. Put this project in a GitHub repository.
2. On the repository page click the green **Code** button, open the
   **Codespaces** tab and click **Create codespace on main**.
3. Wait about a minute: a VS Code editor opens inside your browser, with a
   terminal at the bottom already inside the project folder.
4. In that terminal run `pip install -r requirements.txt`, then follow
   **How to Run** below. Drag your own photos into the file list on the left to
   use them as inputs; click any file in `outputs/` to view the result.

GitHub gives every free personal account a monthly allowance of Codespaces
time. Stop the codespace when you are done (**Codespaces** page -> `...` ->
**Stop**).

### Option B -- run on your own computer

1. **Prerequisites:** Python 3.10 or newer and `pip`. No GPU needed.
2. **Get the code**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
3. **(Recommended) create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate          # on Windows: venv\Scripts\activate
   ```
4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
5. **(Optional) re-create the demo files**

   ```bash
   python3 sample_data/generate_sample_media.py
   ```

On Windows use `python` instead of `python3`.

## How to Run

All commands are run through `main.py`. Use `--help` at any level for
details:

```bash
python3 main.py --help
python3 main.py face-blur --help
```

Results go to the `outputs/` folder.

### Module 1 -- Face Detection & Blur

```bash
# On an image
python3 main.py face-blur --input path/to/photo.jpg

# On a video, analysing every 2nd frame for speed
python3 main.py face-blur --input path/to/video.mp4 --frame-skip 2

# Custom output path
python3 main.py face-blur --input photo.jpg --output outputs/redacted.png
```

> Note: the bundled `sample_scene.png` and `sample_video.mp4` are smoke-test
> files with no real faces or people in them (by design, to avoid bundling
> photos of real people in the repository). They run through the whole
> pipeline and report 0 detections. To see real detections, point `--input`
> at a photo or video you have, e.g. one containing clear, front-facing faces.

### Module 2 -- Pedestrian Detection & Counting

```bash
# On an image
python3 main.py count-people --input path/to/street_photo.jpg

# On a video
python3 main.py count-people --input path/to/video.mp4 --frame-skip 3
```

> The HOG detector is trained on **full-body, upright pedestrians**. Use a
> street/crowd photo where whole bodies are visible. Close-up portraits or
> half-body shots can produce false detections.

### Module 3 -- Handwritten Digit Classification

```bash
# Step 1: train and evaluate the model (creates models/digit_classifier.pkl)
python3 main.py train-digits

# Step 2: classify a sample digit image
python3 main.py predict-digit --input sample_data/sample_digits/sample_digit_0_label_0.png
```

Expected output after training (numbers may differ slightly between library
versions):

```
Training complete. Test accuracy: 0.9917
Model saved to: models/digit_classifier.pkl
```

Training also writes:
- `outputs/digit_classifier_evaluation.txt` -- full precision/recall/F1
  report + confusion matrix (text)
- `outputs/digit_classifier_confusion_matrix.png` -- confusion matrix plot

If you run `predict-digit` before `train-digits`, you get a clear message
telling you to train first.

### Every run produces a report

After any command, check:

```bash
cat outputs/last_run_report.json     # structured summary of the last run
cat outputs/run_history.csv          # append-only history of all runs
tail outputs/logs/visionsuite.log    # detailed application log
```

(On Windows use `type` instead of `cat`.)

### Using a custom configuration

```bash
python3 main.py --config path/to/my_config.yaml count-people --input photo.jpg
```

Note that `--config` goes **before** the command name. See
[`config.yaml`](config.yaml) for all tunable parameters (detection
thresholds, blur strength, output directory, log level, etc.). Set
`log_level: DEBUG` to also stream log lines to the terminal.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the command failed (missing/unreadable input, no trained model, ...) |
| `2` | configuration error (or invalid command-line usage) |

## Testing

The project ships with 51 automated tests covering input validation,
error handling, and core algorithm behaviour (non-maximum suppression,
preprocessing correctness, model save/load, report writing, the CLI itself,
and an end-to-end training test).

```bash
# Run the full test suite
python3 -m pytest -v

# Run only the fast tests (skip the full model-training test)
python3 -m pytest -v -m "not slow"
```

Expected result: all tests pass (`51 passed`).

## Design & Documentation

Detailed design artefacts -- system architecture, workflow, use-case,
class, sequence, and storage-schema diagrams (in Mermaid format, rendered
natively by GitHub) -- are in the [`docs/`](docs) folder:

1. [`docs/01_architecture_diagram.md`](docs/01_architecture_diagram.md)
2. [`docs/02_workflow_diagram.md`](docs/02_workflow_diagram.md)
3. [`docs/03_use_case_diagram.md`](docs/03_use_case_diagram.md)
4. [`docs/04_class_diagram.md`](docs/04_class_diagram.md)
5. [`docs/05_sequence_diagram.md`](docs/05_sequence_diagram.md)
6. [`docs/06_storage_schema_design.md`](docs/06_storage_schema_design.md)

## Non-Functional Requirements

| Requirement | How it's addressed |
|---|---|
| **Performance** | `resize_max_dimension()` caps processing resolution; `--frame-skip` trades accuracy for speed on video |
| **Reliability** | Every CLI command validates inputs up front and is wrapped in a top-level error boundary in `main.py` so failures never produce a raw traceback |
| **Usability** | Single consistent CLI (`argparse` with per-command `--help`); YAML config instead of hard-coded values |
| **Maintainability** | Modular package layout; shared `image_preprocessing.py` avoids duplication; type-hinted dataclasses for config and results |
| **Error Handling** | Explicit `ValueError` / `FileNotFoundError` / `ConfigError` exceptions with actionable messages; all caught and logged, never silently swallowed |
| **Logging & Monitoring** | Rotating file handler + console handler via `logger_setup.py`; every run also gets a structured JSON/CSV report |
| **Resource Efficiency** | Log rotation caps disk usage; image resizing avoids unnecessary memory/CPU use on very large inputs |
| **Scalability** | Modules are independent and stateless per call, so they can be invoked repeatedly in a batch/pipeline script over many files |

## Dataset Description (Digit Classifier)

The digit classifier uses scikit-learn's bundled `load_digits` dataset:
1,797 samples of 8x8 grayscale images of handwritten digits (0-9), derived
from the UCI ML "Optical Recognition of Handwritten Digits" dataset. It
ships with `scikit-learn` (no download needed), keeping the project fully
reproducible offline. An 80/20 stratified train/test split with a fixed seed
is used. See [`src/digit_classifier/model_utils.py`](src/digit_classifier/model_utils.py)
and [`train.py`](src/digit_classifier/train.py) docstrings for the full
model-selection rationale and evaluation methodology.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (or `sklearn`, `yaml`) | Run `pip install -r requirements.txt` (and activate your virtual environment first, if you made one). |
| `python3: command not found` on Windows | Use `python` instead of `python3`. |
| `ERROR: Input file does not exist` | Check the path you gave to `--input` (paths are relative to the folder you run the command from). |
| `No trained model found` | Run `python3 main.py train-digits` first. |
| An output `.mp4` will not open in your player | Open it in VLC, or in the Codespaces file viewer / by downloading it. |

## Known Limitations

- Haar Cascade and HOG+SVM are classical detectors -- they are fast and
  dependency-light but less accurate than modern deep-learning detectors,
  especially on non-frontal faces or unusual poses/lighting. HOG expects
  full-body, upright people.
- The digit classifier expects a single, roughly-centred digit per image
  (matching the training data's convention); it does not perform
  multi-digit segmentation from a photo of a full page.
- Video processing runs on CPU; very long/high-resolution videos will take
  a proportionally long time (mitigated by `--frame-skip`).
- The trained model is stored with Python's `pickle`; only load model files
  that you created yourself.

## License

This project was created for academic coursework submission.
