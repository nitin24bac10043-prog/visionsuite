# 2. Workflow (what happens on every run)

```mermaid
flowchart TD
    A(["Start: python main.py COMMAND ..."]) --> B["Parse arguments"]
    B --> C{"Config valid?"}
    C -- "No" --> C1["Print 'Configuration error'<br/>exit code 2"]
    C -- "Yes" --> D["Dispatch to the chosen command"]
    D --> E{"Input file exists?"}
    E -- "No" --> E1["Print 'ERROR: ...'<br/>exit code 1"]
    E -- "Yes" --> F["Build the detector / model from config values"]
    F --> G{"Image or video?"}
    G -- "Image" --> H["Process the single image"]
    G -- "Video" --> I["Process frames<br/>(detect every Nth frame)"]
    H --> J["Save annotated output"]
    I --> J
    J --> K["Create RunReport"]
    K --> L["Write last_run_report.json<br/>and append run_history.csv"]
    L --> M(["Print result, exit code 0"])
    F -. "processing error" .-> N["Record error in RunReport"]
    N --> L
```

For `train-digits` the "process" step is: load the dataset, split 80/20, fit the
SVM, evaluate on the held-out 20%, save the model, the text report and the
confusion-matrix image. For `predict-digit` it is: preprocess the image to 8x8,
load the model, return the digit and its confidence.
