"""Label grasp episodes by what really happened to the date, one keypress each.

The recorded "outcome" comes from the gripper's force signal, so a cycle can
count as GRABBED while the date slipped out during the reverse. This writes
the operator's own label, which is authoritative, next to it:

    held     still in the hand after the reverse
    slipped  grasped, then lost during the reverse
    missed   never grasped
    unclear  can't tell (fingers hide it, no after-reverse image)

Usage:
    python3 -m ur10e_curobo.label_episodes                  # label unlabelled
    python3 -m ur10e_curobo.label_episodes --since 20261029
    python3 -m ur10e_curobo.label_episodes --all            # relabel everything
    python3 -m ur10e_curobo.label_episodes --stats          # outcome vs label

Shows the episode's images (crop, crop_close, after_reverse) side by side in
one window; press the label key in that window. Writes human_label into each
episode.json.
"""

import argparse
import json
import os
from collections import Counter

KEYS = {"h": "held", "s": "slipped", "m": "missed", "u": "unclear", "k": None}


def episodes(root, since):
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "episode.json")
        if name >= since and os.path.isfile(path):
            yield os.path.join(root, name), path


def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"  skipping {path}: {exc}")
        return None


def describe(ep):
    after = ep.get("after_reverse") or {}
    peak = ep.get("close_peak_current_a")
    return (f"outcome={ep.get('outcome')}/{ep.get('end')}"
            + (f"  peak={peak:.2f}A" if peak is not None else "")
            + ("  CURRENT-LIMITED" if ep.get("close_current_limited") else "")
            + ("" if after.get("image") else "  (no after-reverse image)"))


def show(directory, title):
    """One enlarged side-by-side window of the episode's images; returns the
    key pressed in it, or None when there is no display (fall back to input)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    tiles = []
    for name in ("crop.jpg", "crop_close.jpg", "after_reverse.jpg"):
        image = cv2.imread(os.path.join(directory, name))
        if image is None:
            continue
        image = cv2.resize(image, (int(image.shape[1] * 520 / image.shape[0]), 520))
        cv2.putText(image, name[:-4], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(image)
    if not tiles:
        return None
    try:
        cv2.imshow("label_episodes", np.hstack(tiles))
        cv2.setWindowTitle("label_episodes", f"{title}   [h]eld [s]lipped [m]issed [u]nclear s[k]ip [q]uit")
        return chr(cv2.waitKey(0) & 0xFF).lower()
    except cv2.error:
        return None


def stats(root, since):
    """Cross-tab the automatic outcome against the operator's label."""
    table = Counter()
    for _, path in episodes(root, since):
        ep = load(path)
        if ep is not None:
            table[(ep.get("outcome"), ep.get("human_label") or "unlabelled")] += 1
    labels = [v for v in KEYS.values() if v] + ["unlabelled"]
    print(f"{'outcome':14}" + "".join(f"{label:>11}" for label in labels))
    for outcome in sorted({o for o, _ in table}, key=str):
        print(f"{str(outcome):14}"
              + "".join(f"{table[(outcome, label)]:>11}" for label in labels))


def label(root, since, relabel):
    pending = [(d, p, ep) for d, p in episodes(root, since)
               if os.path.isfile(os.path.join(d, "after_reverse.jpg"))  # what happened is only visible there
               and (ep := load(p)) is not None and (relabel or not ep.get("human_label"))]
    if not pending:
        print(f"Nothing to label in {root}.")
        return
    print(f"{len(pending)} to label. [h]eld [s]lipped [m]issed [u]nclear "
          "s[k]ip [q]uit\n")
    done = 0
    for n, (directory, path, ep) in enumerate(pending, 1):
        print(f"[{n}/{len(pending)}] {os.path.basename(directory)}")
        print(f"          {describe(ep)}")
        if ep.get("human_label"):
            print(f"          currently: {ep['human_label']}")
        key = show(directory, os.path.basename(directory))
        while True:
            if key in KEYS or key == "q":
                print(f"          label> {key}")
            else:
                try:
                    key = input("          label> ").strip().lower()[:1]
                except (EOFError, KeyboardInterrupt):
                    key = "q"
            if key == "q":
                print(f"{done} labelled.")
                return
            if key in KEYS:
                break
            print("          h / s / m / u / k / q")
            key = None
        if KEYS[key] is None:
            continue
        ep["human_label"] = KEYS[key]
        try:
            with open(path, "w") as handle:
                json.dump(ep, handle, indent=2, sort_keys=True)
            done += 1
        except Exception as exc:
            print(f"          NOT saved: {exc}")
    print(f"{done} labelled.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="~/harvest_logs/episodes")
    parser.add_argument("--since", default="", help="episode prefix, e.g. 20261029")
    parser.add_argument("--all", action="store_true", help="relabel labelled episodes")
    parser.add_argument("--stats", action="store_true", help="outcome vs label only")
    args = parser.parse_args(argv)
    root = os.path.expanduser(args.dir)
    if args.stats:
        stats(root, args.since)
    else:
        label(root, args.since, args.all)


if __name__ == "__main__":
    main()
