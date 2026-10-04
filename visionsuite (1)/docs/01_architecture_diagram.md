# 1. System Architecture

VisionSuite is a layered command-line application. `main.py` is the only entry
point; it loads the configuration, then dispatches to one of three independent
processing modules. Shared helpers and the reporting/logging layer are used by
every module.

```mermaid
flowchart TD
    USER(["User / batch script"]) --> CLI["main.py<br/>argparse CLI + error boundary"]

    CLI --> CFG["config_loader.py<br/>validated AppConfig"]
    CFG --> YAML[("config.yaml")]

    subgraph MODULES["Processing modules"]
        FACE["face_module.py<br/>Haar Cascade face blur"]
        PEOPLE["people_counter_module.py<br/>HOG + SVM people counter"]
        DIGIT["digit_classifier/<br/>train.py, predict.py, model_utils.py"]
    end

    CLI --> FACE
    CLI --> PEOPLE
    CLI --> DIGIT

    FACE --> PRE["image_preprocessing.py<br/>load, resize, NMS, blur, video I/O"]
    PEOPLE --> PRE

    subgraph LIBS["Libraries"]
        CV["OpenCV"]
        SK["scikit-learn"]
        MPL["Matplotlib"]
    end

    PRE --> CV
    FACE --> CV
    PEOPLE --> CV
    DIGIT --> SK
    DIGIT --> MPL

    CLI --> REP["report_generator.py<br/>RunReport"]
    CLI --> LOG["logger_setup.py<br/>rotating logs"]

    subgraph STORE["Files on disk"]
        MEDIA[("outputs/*.png, *.mp4")]
        JSON[("outputs/last_run_report.json")]
        CSV[("outputs/run_history.csv")]
        LOGF[("outputs/logs/visionsuite.log")]
        MODEL[("models/digit_classifier.pkl")]
    end

    FACE --> MEDIA
    PEOPLE --> MEDIA
    REP --> JSON
    REP --> CSV
    LOG --> LOGF
    DIGIT --> MODEL
```
