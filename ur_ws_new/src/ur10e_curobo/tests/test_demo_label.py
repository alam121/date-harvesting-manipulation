"""Grading must round-trip into sample.json and never lose a record.

A mis-saved grade is invisible until training, so the cheap check is here: feed
demo_label a fake session and read the files back.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ur10e_curobo import demo_label  # noqa: E402


def _session(tmp, specs):
    for name, sample in specs.items():
        os.makedirs(os.path.join(tmp, name))
        with open(os.path.join(tmp, name, "sample.json"), "w") as handle:
            json.dump(sample, handle)


def _labels(tmp):
    return {name: json.load(open(path)).get("operator_label")
            for name, path in ((os.path.basename(d), p)
                               for d, p in demo_label.records(tmp))}


def test_keys_cover_the_three_grades_and_skip():
    assert demo_label.KEYS == {"g": "good", "e": "edge", "f": "fail", "s": None}


def test_grades_are_written_and_skip_leaves_the_record_alone(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, {f"2026{i}": {"auto_outcome": {"outcome": "NO_GRAB", "end": "X"}}
                       for i in range(3)})
        keys = iter(["g", "s", "f"])
        monkeypatch.setattr(demo_label, "show", lambda d: None)
        monkeypatch.setattr("builtins.input", lambda *_: next(keys))
        assert demo_label.grade(tmp, regrade=False) == 2
        assert _labels(tmp) == {"20260": "good", "20261": None, "20262": "fail"}


def test_already_graded_records_are_not_offered_again(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, {"a": {"operator_label": "good"}, "b": {}})
        monkeypatch.setattr(demo_label, "show", lambda d: None)
        monkeypatch.setattr("builtins.input", lambda *_: "e")
        assert demo_label.grade(tmp, regrade=False) == 1
        # 'a' keeps its original grade; only 'b' was asked about.
        assert _labels(tmp) == {"a": "good", "b": "edge"}


def test_regrade_offers_everything(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, {"a": {"operator_label": "good"}, "b": {"operator_label": "fail"}})
        monkeypatch.setattr(demo_label, "show", lambda d: None)
        monkeypatch.setattr("builtins.input", lambda *_: "f")
        assert demo_label.grade(tmp, regrade=True) == 2
        assert _labels(tmp) == {"a": "fail", "b": "fail"}


def test_quit_stops_without_touching_the_rest(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, {"a": {}, "b": {}})
        keys = iter(["g", "q"])
        monkeypatch.setattr(demo_label, "show", lambda d: None)
        monkeypatch.setattr("builtins.input", lambda *_: next(keys))
        assert demo_label.grade(tmp, regrade=False) == 1
        assert _labels(tmp) == {"a": "good", "b": None}


def test_describe_survives_a_record_with_no_outcome_yet():
    assert "no gripper verdict" in demo_label.describe({})
