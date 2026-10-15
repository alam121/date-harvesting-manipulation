"""Split a trained NuRec scene into the tree (dates removed) and one small NuRec asset per date.

Uses dates.json / date_labels.npy from localize_dates.py. PPISP colour correction is baked into
the Gaussians once for the whole model (as reconstruct.sh's export does), then the baked model
is sliced, so tree and dates share exactly the same appearance.

  cd ~/3dgrut && source .venv/bin/activate
  python ~/manipulatorsdatepalm/isaac/nurec/split_dates.py ~/nurec_data/scenes/palm3 --exclude 14 16

Writes SCENE/split/tree_nurec.usdz, SCENE/split/date_XX.usdz (each date re-centred on its own
centre) and SCENE/split/dates_physics.json (centre, collider ellipsoid, asset per date, in the
scene frame) for the Isaac scene.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from threedgrut.export.scripts.export_usd import load_model_from_checkpoint
from threedgrut.export.usd.nurec.exporter import NuRecExporter
from threedgrut.export.usd.post_processing.sh_bake import (
    MODE_PPISP_BAKE_VIGNETTING_NONE,
    PPISPPostProcessingBakeAdapter,
    bake_post_processing_into_sh,
)
import threedgrut.datasets as datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_frame import rot_to_quat  # noqa: E402

PER_GAUSSIAN = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="Date ids to leave in the tree.")
    parser.add_argument(
        "--bunch_radius",
        type=float,
        default=0.4,
        help="Drop dates farther than this [m] from the bunch centre (table clutter); 0 keeps all.",
    )
    parser.add_argument(
        "--carve_radius",
        type=float,
        default=1.15,
        help="Also move Gaussians inside this multiple of each date's ellipsoid into the date asset.",
    )
    return parser.parse_args()


def export_subset(exporter, model, full: dict, keep: torch.Tensor, path: Path, conf, offset=None):
    """Export only the kept Gaussians. The model owns non-copyable renderer state, so its tensors
    are swapped in place and restored afterwards."""
    for name in PER_GAUSSIAN:
        tensor = full[name][keep].clone()
        if name == "positions" and offset is not None:
            tensor = tensor - offset
        setattr(model, name, torch.nn.Parameter(tensor, requires_grad=False))
    try:
        exporter.export(model, path, conf=conf)
    finally:
        for name in PER_GAUSSIAN:
            setattr(model, name, torch.nn.Parameter(full[name], requires_grad=False))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = json.loads((args.scene / "dates.json").read_text())
    labels = np.load(args.scene / "date_labels.npy")
    checkpoint = Path(manifest["checkpoint"])

    model, conf, _, post_processing = load_model_from_checkpoint(str(checkpoint))
    out = args.scene / "split"
    out.mkdir(exist_ok=True)
    baked_path = out / "baked_gaussians.pt"
    if baked_path.exists():
        print(f"[INFO] reusing {baked_path}")
    else:
        if post_processing is not None:
            conf.path = str(args.scene)
            dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
            model = bake_post_processing_into_sh(
                model=model,
                post_processing=post_processing,
                train_dataset=dataset,
                conf=conf,
                adapter=PPISPPostProcessingBakeAdapter(
                    camera_id=0, frame_id=0, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_NONE
                ),
                num_iterations=3000,
            )
        torch.save({name: getattr(model, name).detach().cpu() for name in PER_GAUSSIAN}, baked_path)
    device = model.positions.device
    full = {name: tensor.to(device) for name, tensor in torch.load(baked_path).items()}

    positions = full["positions"].cpu().numpy()
    dates = [d for d in manifest["dates"] if d["id"] not in set(args.exclude)]
    if args.bunch_radius > 0 and dates:
        # Clutter elsewhere (table, cables) can pass the date filters; keep what hangs in the bunch.
        # Bunch centre: median of the better-supported half of the dates.
        votes = np.array([d["mean_votes"] for d in dates])
        strong = np.array([d["centre"] for d in dates])[votes >= np.median(votes)]
        bunch = np.median(strong, 0)
        far = [d["id"] for d in dates if np.linalg.norm(np.array(d["centre"]) - bunch) > args.bunch_radius]
        if far:
            print(f"[INFO] dropping {len(far)} dates farther than {args.bunch_radius} m from the bunch: {far}")
        dates = [d for d in dates if d["id"] not in far]
    owner = np.full(len(positions), -1, np.int64)
    for date in dates:
        owner[labels == date["id"]] = date["id"]
        # Voting only labels surface Gaussians seen as "date"; carve the rest of the fruit too so no
        # ghost stays on the tree when the date is removed.
        axes = np.array(date["axes"])
        radii = np.maximum(np.array(date["extents"]) / 2, 0.004) * args.carve_radius
        local = (positions - np.array(date["centre"])) @ axes.T
        inside = ((local / radii) ** 2).sum(1) <= 1.0
        owner[inside & (owner == -1)] = date["id"]

    exporter = NuRecExporter(export_cameras=False, export_post_processing=False)
    export_subset(exporter, model, full, torch.from_numpy(owner == -1).to(device), out / "tree_nurec.usdz", conf)
    np.save(out / "tree_gaussians.npy", owner == -1)  # for later cuts (bunch_skeleton.py --export)

    physics = []
    for date in dates:
        keep = torch.from_numpy(owner == date["id"]).to(device)
        centre = torch.tensor(date["centre"], dtype=full["positions"].dtype, device=device)
        asset = out / f"date_{date['id']:02d}.usdz"
        export_subset(exporter, model, full, keep, asset, conf, offset=centre)
        axes = np.array(date["axes"])
        if np.linalg.det(axes) < 0:
            axes[2] *= -1  # right-handed collider frame
        physics.append({
            "id": date["id"],
            "asset": asset.name,
            "centre": date["centre"],
            "collider_quat_wxyz": list(rot_to_quat(axes.T)),
            "collider_radii": (np.maximum(np.array(date["extents"]) / 2, 0.004)).round(4).tolist(),
            "n_gaussians": int(keep.sum()),
        })
        print(f"[INFO] date {date['id']:2d}: {int(keep.sum())} Gaussians -> {asset.name}")

    (out / "dates_physics.json").write_text(json.dumps({
        "frame": manifest["frame"],
        "tree_asset": "tree_nurec.usdz",
        "tree_gaussians": int((owner == -1).sum()),
        "dates": physics,
    }, indent=2))
    print(f"[DONE] tree ({(owner == -1).sum()} Gaussians) + {len(physics)} dates -> {out}")


if __name__ == "__main__":
    main()
