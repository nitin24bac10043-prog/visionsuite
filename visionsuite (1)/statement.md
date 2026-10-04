# Problem Statement -- VisionSuite

## Problem Statement

Organisations and researchers regularly need to perform common computer
vision tasks -- protecting privacy in visual media, understanding crowd
density from camera footage, and recognising handwritten characters -- but
often reach for heavyweight, poorly-documented, or GUI-only tools that are
hard to automate, audit, or reproduce. There is a need for a lightweight,
fully offline, command-line toolkit that applies well-understood computer
vision and machine learning techniques to these three problems in a
modular, testable, and reproducible way, so it can be dropped into
automated pipelines (e.g. batch-processing a folder of images/videos on a
server with no display) and audited afterwards through structured logs and
reports.

## Scope of the Project

VisionSuite is a **modular computer-vision CLI toolkit** covering:

- **In scope:**
  - Detecting and anonymising (blurring) human faces in images and video,
    for privacy-preserving media sharing.
  - Detecting and counting pedestrians in images and video, for basic
    crowd/traffic analytics.
  - Training, evaluating, and using a machine-learning classifier for
    handwritten digit recognition.
  - A single, consistent command-line interface across all three
    capabilities, with YAML-driven configuration, structured JSON/CSV run
    reports, and rotating log files.
  - Automated unit tests covering the core logic of every module.

- **Out of scope:**
  - Real-time streaming from live camera hardware (the tool processes
    files -- images and video files -- not live camera feeds).
  - A graphical user interface (the project is explicitly a CLI tool, per
    the assignment's "must be fully executable via the command line"
    requirement).
  - Face **recognition** / identity matching (the face module only
    *detects and anonymises*, it never identifies or stores who a face
    belongs to -- this is an intentional privacy-respecting design choice).
  - Training custom deep-learning object detectors from scratch (the
    project deliberately uses classical, well-understood, CPU-only models
    that ship with OpenCV/scikit-learn so it remains fully reproducible
    without GPUs or internet access to download large model weights).

## Target Users

- **Students / evaluators** who need to run and inspect a self-contained
  CV project from the command line without any special hardware or
  internet-dependent downloads.
- **Developers / researchers** who want a starting point for batch
  privacy-anonymisation or crowd-counting pipelines that they can extend.
- **Small organisations** that want to redact faces from photos/videos
  before sharing them publicly (e.g. event photography, CCTV footage
  review) without relying on a third-party cloud service.

## High-Level Features

1. **Face Detection & Privacy Anonymisation** -- detect faces with a Haar
   Cascade classifier and blur them; works on both single images and video
   files, with a configurable blur strength and frame-skip for speed.
2. **Pedestrian Detection & Counting** -- detect people with a HOG+SVM
   detector, apply non-maximum suppression to remove duplicate boxes, and
   report per-image counts or per-video max/average counts.
3. **Handwritten Digit Classification** -- train an RBF-kernel SVM on the
   `scikit-learn` digits dataset, evaluate it (accuracy, per-class
   precision/recall/F1, confusion matrix), persist the trained model, and
   classify new digit images from the command line.
4. **Unified CLI, configuration, logging & reporting** -- one `main.py`
   entry point with subcommands, a single `config.yaml` to tune all
   modules, rotating log files, and a JSON + CSV audit trail for every run.
