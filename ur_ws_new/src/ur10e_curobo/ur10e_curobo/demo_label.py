"""Grade recorded approaches good / edge / fail, one keypress each.

The point of outcome capture is that grading is cheap: the harvest cycle runs at
full speed and writes a record per attempt, then a session's worth is graded here
in a couple of minutes. Only the operator's grade is recorded and trained on.

    good  the fruit was taken cleanly
    edge  it worked, but only just: one finger, a graze, nearly dropped
    fail  nothing, or it fell

Usage:
    python3 -m ur10e_curobo.demo_label                 # grade what is ungraded
    python3 -m ur10e_curobo.demo_label --all           # regrade everything
    python3 -m ur10e_curobo.demo_label --stats         # counts only, grade nothing
    python3 -m ur10e_curobo.demo_label --dir PATH

Writes operator_label back into each sample.json. Shows full_before_marked.jpg
if a viewer is available.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

from .demo_sequence import update_training_readiness

KEYS = {"g": "good", "e": "edge", "f": "fail", "s": None}
VALID = ("good", "edge", "fail")


def records(root):
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "sample.json")
        if os.path.isfile(path):
            yield os.path.join(root, name), path


def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"  skipping {path}: {exc}")
        return None


def describe(sample):
    trace = (sample.get("before") or {}).get("depth_trace") or {}
    vision = trace.get("accepted_goal_vision") or {}
    radius = vision.get("estimated_radius_m")
    return (f"mode={trace.get('mode', '?')}"
            f"  range={sample.get('range_m', '?')}m"
            + (f"  r={radius * 1000:.0f}mm" if radius else ""))


def show(directory):
    """Open the marked approach frame. Grading works without it."""
    image = os.path.join(directory, "full_before_marked.jpg")
    if not os.path.isfile(image):
        return
    for viewer in ("xdg-open", "eog", "feh"):
        try:
            subprocess.Popen([viewer, image],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def stats(root):
    counts = Counter()
    total = 0
    for _, path in records(root):
        sample = load(path)
        if sample is None:
            continue
        total += 1
        counts[sample.get("operator_label") or "ungraded"] += 1
    print(f"{total} record(s) in {root}")
    for label in VALID + ("ungraded",):
        if counts[label]:
            print(f"  {label:9} {counts[label]:4}")
    return total


def grade(root, regrade):
    pending = []
    for directory, path in records(root):
        sample = load(path)
        if sample is None:
            continue
        if regrade or not sample.get("operator_label"):
            pending.append((directory, path, sample))
    if not pending:
        print(f"Nothing to grade in {root}.")
        return 0
    print(f"{len(pending)} to grade. [g]ood [e]dge [f]ail [s]kip [q]uit\n")
    done = 0
    for n, (directory, path, sample) in enumerate(pending, 1):
        print(f"[{n}/{len(pending)}] {os.path.basename(directory)}")
        print(f"          {describe(sample)}")
        if sample.get("operator_label"):
            print(f"          currently: {sample['operator_label']}")
        show(directory)
        while True:
            try:
                key = input("          grade> ").strip().lower()[:1]
            except (EOFError, KeyboardInterrupt):
                print(f"\nStopped. {done} graded.")
                return done
            if key == "q":
                print(f"{done} graded.")
                return done
            if key in KEYS:
                break
            print("          g / e / f / s / q")
        if KEYS[key] is None:
            continue
        sample["operator_label"] = KEYS[key]
        try:
            with open(path, "w") as handle:
                json.dump(sample, handle, indent=1, default=str)
            update_training_readiness(directory, sample["operator_label"])
            done += 1
        except Exception as exc:
            print(f"          NOT saved: {exc}")
    print(f"{done} graded.")
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="~/harvest_logs/demos")
    parser.add_argument("--all", action="store_true", help="regrade already-graded records")
    parser.add_argument("--stats", action="store_true", help="counts only, grade nothing")
    args = parser.parse_args(argv)
    root = os.path.expanduser(args.dir)
    if not os.path.isdir(root):
        print(f"No such directory: {root}")
        return 1
    if args.stats:
        return 0 if stats(root) else 1
    grade(root, args.all)
    stats(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
