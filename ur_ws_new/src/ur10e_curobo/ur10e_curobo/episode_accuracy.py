"""Summarise grasp accuracy across recorded episodes, grouped by outcome.

Reads close_geometry from each episode.json (see snapshot_close_geometry):

    python3 -m ur10e_curobo.episode_accuracy [--since 20261026] [--dir DIR]

target_rel_tool is the date's position relative to the closure centre in the
tool frame at the moment of closing; z is the pointing axis, + = date beyond
the closure point (arm stopped short), - = arm went past it.
"""
import argparse
import json
import math
import os
import statistics


def _med(values):
    return f"{statistics.median(values):+6.1f}" if values else "     -"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="~/harvest_logs/episodes")
    ap.add_argument("--since", default="", help="episode dir prefix, e.g. 20261026")
    args = ap.parse_args()
    root = os.path.expanduser(args.dir)

    groups = {}
    for name in sorted(os.listdir(root)):
        if name < args.since:
            continue
        try:
            with open(os.path.join(root, name, "episode.json")) as fh:
                episode = json.load(fh)
        except (OSError, ValueError):
            continue
        groups.setdefault(episode.get("outcome", "?"), []).append(
            episode.get("close_geometry") or {})

    print(f"{'outcome':<9}{'n':>4}{'seen':>5}   target_rel_tool median [x y z] mm"
          f"   tracking mm   perception mm")
    for outcome, geoms in sorted(groups.items()):
        rel = [g["target_rel_tool_mm"] for g in geoms if g.get("target_rel_tool_mm")]
        tracking = [g["tracking_mm"] for g in geoms if "tracking_mm" in g]
        perception = [math.hypot(*g["perception_mm"]) for g in geoms
                      if g.get("perception_mm")]
        print(f"{outcome:<9}{len(geoms):>4}{len(rel):>5}   "
              f"[{_med([r[0] for r in rel])} {_med([r[1] for r in rel])} "
              f"{_med([r[2] for r in rel])}]      {_med(tracking)}       "
              f"{_med(perception)}")
    print("\nseen = episodes with a fresh detection of the date at close; "
          "the rest only have tracking.")


if __name__ == "__main__":
    main()
