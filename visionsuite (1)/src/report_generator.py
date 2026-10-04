"""
report_generator.py
-------------------
Writes an audit trail for every CLI run:

* ``last_run_report.json`` -- structured summary of the most recent run
* ``run_history.csv``      -- append-only history, one row per run
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

CSV_COLUMNS = ["timestamp", "command", "status", "input_path",
               "elapsed_seconds", "results", "errors"]


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@dataclass
class RunReport:
    command: str
    input_path: str
    output_dir: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "failed" if self.errors else "success"

    def add_result(self, key: str, value: Any) -> None:
        self.results[key] = value

    def add_error(self, message: str) -> None:
        self.errors.append(str(message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "command": self.command,
            "status": self.status,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "results": self.results,
            "errors": self.errors,
        }

    def save_json(self, path: str) -> None:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    def append_csv(self, path: str) -> None:
        _ensure_parent_dir(path)
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([
                self.timestamp,
                self.command,
                self.status,
                self.input_path,
                self.results.get("elapsed_seconds", ""),
                json.dumps(self.results, default=str),
                " | ".join(self.errors),
            ])
