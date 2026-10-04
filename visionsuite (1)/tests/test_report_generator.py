import csv
import json

from src.report_generator import RunReport


def _report():
    report = RunReport(command="face-blur", input_path="photo.jpg", output_dir="outputs")
    report.add_result("faces_detected", 3)
    report.add_result("elapsed_seconds", 0.25)
    return report


def test_save_json_contains_command_status_and_results(tmp_path):
    path = str(tmp_path / "reports" / "last.json")
    _report().save_json(path)
    data = json.load(open(path, encoding="utf-8"))
    assert data["command"] == "face-blur"
    assert data["status"] == "success"
    assert data["results"]["faces_detected"] == 3


def test_append_csv_writes_header_once_and_appends_rows(tmp_path):
    path = str(tmp_path / "history.csv")
    _report().append_csv(path)
    _report().append_csv(path)
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
    assert rows[0][0] == "timestamp"
    assert len(rows) == 3                # 1 header + 2 runs


def test_errors_mark_the_report_as_failed(tmp_path):
    report = _report()
    report.add_error("boom")
    path = str(tmp_path / "last.json")
    report.save_json(path)
    data = json.load(open(path, encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["errors"] == ["boom"]
