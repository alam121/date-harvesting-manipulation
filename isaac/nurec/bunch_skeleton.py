"""Extract the date bunch's stalk and strands as a skeleton tree from a split NuRec scene.

Strand Gaussians are selected by colour (tan/orange) around the dates, the 11 extracted dates
are removed, and a level-set skeleton is built: geodesic distance from the root (the highest
point, where the stalk leaves the trunk) is cut into bands; each connected cluster in a band is
one node, linked to the node of the band above it. Short twigs are pruned.

  cd ~/3dgrut && source .venv/bin/activate
  python ~/manipulatorsdatepalm/isaac/nurec/bunch_skeleton.py ~/nurec_data/scenes/palm3

Writes SCENE/split/bunch_skeleton.json (nodes in the scene frame with parent, radius, colour,
and the Gaussian indices each node carries) and bunch_skeleton_*.jpg previews.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_frame import read_cameras, read_images  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--min_opacity", type=float, default=0.1)
    parser.add_argument("--hue", type=float, nargs=2, default=(20.0, 45.0), help="Strand hue range [deg].")
    parser.add_argument("--min_saturation", type=float, default=0.35)
    parser.add_argument("--min_value", type=float, default=0.35)
    parser.add_argument("--voxel", type=float, default=0.004, help="Downsampling voxel [m].")
    parser.add_argument("--link", type=float, default=0.012, help="Neighbour graph radius [m].")
    parser.add_argument("--band", type=float, default=0.025, help="Geodesic band = skeleton segment length [m].")
    parser.add_argument("--min_branch", type=float, default=0.05, help="Prune twigs shorter than this [m].")
    parser.add_argument("--margin", type=float, nargs=3, default=(0.25, 0.25, 0.6),
                        help="Region around the dates (x, y, z-up) to search [m].")
    parser.add_argument("--export", action="store_true",
                        help="Also export split/tree_nobunch_nurec.usdz: the tree without dates and strands.")
    parser.add_argument("--carve", type=float, default=1.6, help="Carve radius around segments, x node radius.")
    parser.add_argument("--min_carve", type=float, default=0.006, help="Minimum carve radius [m].")
    return parser.parse_args()


def carve_strands(nodes: list[dict], positions: np.ndarray, candidates: np.ndarray, args) -> np.ndarray:
    """Tree Gaussians inside a cylinder around any skeleton segment (all colours: shaded sides too)."""
    carved = np.zeros(len(positions), bool)
    points = positions[candidates]
    tree = cKDTree(points)
    for node in nodes:
        if node["parent"] < 0:
            continue
        a, b = np.array(nodes[node["parent"]]["position"]), np.array(node["position"])
        radius = max(args.carve * node["radius"], args.min_carve)
        near = tree.query_ball_point((a + b) / 2, np.linalg.norm(b - a) / 2 + radius)
        if not near:
            continue
        p = points[near]
        axis = b - a
        t = np.clip(((p - a) @ axis) / max(axis @ axis, 1e-9), 0.0, 1.0)
        inside = np.linalg.norm(p - (a + t[:, None] * axis), axis=1) <= radius
        carved[candidates[np.array(near)[inside]]] = True
    return carved


def export_tree_without_bunch(split: Path, nodes: list[dict], positions: np.ndarray, args) -> None:
    from split_dates import export_subset
    from threedgrut.export.scripts.export_usd import load_model_from_checkpoint
    from threedgrut.export.usd.nurec.exporter import NuRecExporter

    tree = np.load(split / "tree_gaussians.npy")
    carved = carve_strands(nodes, positions, np.flatnonzero(tree), args)
    np.save(split / "bunch_gaussians.npy", carved)
    manifest = json.loads((split.parent / "dates.json").read_text())
    model, conf, _, _ = load_model_from_checkpoint(manifest["checkpoint"])
    device = model.positions.device
    full = {k: v.to(device) for k, v in torch.load(split / "baked_gaussians.pt").items()}
    keep = torch.from_numpy(tree & ~carved).to(device)
    export_subset(NuRecExporter(export_cameras=False, export_post_processing=False), model, full, keep,
                  split / "tree_nobunch_nurec.usdz", conf)
    print(f"[SKEL] carved {carved.sum()} strand Gaussians; tree without bunch: {int(keep.sum())} -> "
          f"{split / 'tree_nobunch_nurec.usdz'}")


def strand_points(args, split: Path):
    gaussians = torch.load(split / "baked_gaussians.pt")
    positions = gaussians["positions"].numpy()
    opacity = torch.sigmoid(gaussians["density"]).reshape(-1).numpy()
    rgb = np.clip(0.5 + 0.28209479 * gaussians["features_albedo"].numpy()[:, :3], 0, 1)
    physics = json.loads((split / "dates_physics.json").read_text())
    centres = np.array([d["centre"] for d in physics["dates"]])
    margin = np.array(args.margin)
    lo = centres.min(0) - np.array([margin[0], margin[1], margin[1]])
    hi = centres.max(0) + margin
    region = np.flatnonzero(np.all((positions > lo) & (positions < hi), 1) & (opacity >= args.min_opacity))
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb[region]])
    tan = ((hsv[:, 0] * 360 > args.hue[0]) & (hsv[:, 0] * 360 < args.hue[1])
           & (hsv[:, 1] > args.min_saturation) & (hsv[:, 2] > args.min_value))
    index = region[tan]
    # The extracted dates are their own rigid bodies; keep their Gaussians out of the strands.
    located = {d["id"]: d for d in json.loads((split.parent / "dates.json").read_text())["dates"]}
    keep = np.ones(len(index), bool)
    for date in physics["dates"]:
        axes = np.array(located[date["id"]]["axes"])
        radii = np.array(date["collider_radii"]) * 1.2
        local = (positions[index] - np.array(date["centre"])) @ axes.T
        keep &= ((local / radii) ** 2).sum(1) > 1.0
    return index[keep], positions, rgb


def main():
    args = parse_args()
    split = args.scene / "split"
    index, positions, rgb = strand_points(args, split)
    points = positions[index]

    # Voxel downsample, remembering which Gaussians each voxel carries.
    keys = np.floor(points / args.voxel).astype(np.int64)
    _, voxel_of, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel_of = voxel_of.reshape(-1)
    voxels = np.zeros((counts.size, 3))
    np.add.at(voxels, voxel_of, points)
    voxels /= counts[:, None]

    pairs = cKDTree(voxels).query_pairs(args.link, output_type="ndarray")
    lengths = np.linalg.norm(voxels[pairs[:, 0]] - voxels[pairs[:, 1]], axis=1)
    graph = coo_matrix((lengths, (pairs[:, 0], pairs[:, 1])), shape=(len(voxels),) * 2).tocsr()
    _, component = connected_components(graph, directed=False)

    # Root: the highest part of the largest component (the stalk where it leaves the trunk).
    main_component = np.bincount(component).argmax()
    members = np.flatnonzero(component == main_component)
    top = members[voxels[members, 2] >= voxels[members, 2].max() - 0.01]
    distance = dijkstra(graph, directed=False, indices=top, min_only=True)
    reachable = np.isfinite(distance)
    print(f"[SKEL] {len(index)} strand Gaussians -> {len(voxels)} voxels; "
          f"{reachable.sum()} connected to the root; root z {voxels[top, 2].mean():.3f} m")

    # Level-set skeleton.
    band = np.full(len(voxels), -1)
    band[reachable] = (distance[reachable] / args.band).astype(int)
    nodes, parents, node_voxels = [], [], []
    node_of_voxel = np.full(len(voxels), -1)
    for level in range(band.max() + 1):
        in_band = np.flatnonzero(band == level)
        if len(in_band) == 0:
            continue
        sub = graph[in_band][:, in_band]
        _, cluster = connected_components(sub, directed=False)
        for c in np.unique(cluster):
            member = in_band[cluster == c]
            parent = -1
            if level > 0:
                # Parent: the node of the band above that this cluster touches most.
                neighbours = graph[member].nonzero()[1]
                above = node_of_voxel[neighbours]
                above = above[(above >= 0) & (band[neighbours] == level - 1)]
                if len(above):
                    parent = int(np.bincount(above).argmax())
                elif len(member) < 3:
                    continue
            node_of_voxel[member] = len(nodes)
            nodes.append(voxels[member].mean(0))
            parents.append(parent)
            node_voxels.append(member)
    nodes = np.array(nodes)
    parents = np.array(parents)

    # Prune twigs: remove leaf chains shorter than min_branch (repeat until stable).
    alive = np.ones(len(nodes), bool)
    changed = True
    while changed:
        changed = False
        children = np.zeros(len(nodes), int)
        for n, p in enumerate(parents):
            if alive[n] and p >= 0:
                children[p] += 1
        for leaf in np.flatnonzero(alive & (children == 0)):
            chain, n = [leaf], leaf
            while parents[n] >= 0 and children[parents[n]] == 1 and alive[parents[n]]:
                n = parents[n]
                chain.append(n)
            length = sum(np.linalg.norm(nodes[a] - nodes[parents[a]]) for a in chain if parents[a] >= 0)
            root_attached = parents[chain[-1]] >= 0
            if root_attached and length < args.min_branch:
                alive[chain] = False
                changed = True
        # Orphans (no parent path to the root) are dropped too.
        for n in range(len(nodes)):
            if alive[n] and parents[n] >= 0 and not alive[parents[n]]:
                alive[n] = False
                changed = True

    # Keep only the tree hanging from the root (band 0); clusters that never found a parent would
    # otherwise become extra roots pinned to the world.
    roots = {n for n in range(len(nodes)) if alive[n] and parents[n] < 0 and band[node_voxels[n][0]] == 0}
    for n in range(len(nodes)):  # nodes are in band order, so parents come first
        if alive[n] and n not in roots and (parents[n] < 0 or not alive[parents[n]]):
            alive[n] = False

    # Renumber and describe each node.
    new_id = -np.ones(len(nodes), int)
    new_id[alive] = np.arange(alive.sum())
    out_nodes = []
    for n in np.flatnonzero(alive):
        member = node_voxels[n]
        gaussians = index[np.isin(voxel_of, member)]
        spread = np.linalg.norm(voxels[member] - nodes[n], axis=1)
        out_nodes.append({
            "position": nodes[n].round(5).tolist(),
            "parent": int(new_id[parents[n]]) if parents[n] >= 0 else -1,
            "radius": float(np.clip(np.median(spread) if len(spread) > 2 else 0.004, 0.003, 0.012)),
            "colour_rgb": rgb[gaussians].mean(0).round(3).tolist(),
            "gaussians": gaussians.tolist(),
        })
    children = np.zeros(len(out_nodes), int)
    for node in out_nodes:
        if node["parent"] >= 0:
            children[node["parent"]] += 1
    leaves = int((children == 0).sum())
    total = sum(np.linalg.norm(np.array(n["position"]) - np.array(out_nodes[n["parent"]]["position"]))
                for n in out_nodes if n["parent"] >= 0)
    (split / "bunch_skeleton.json").write_text(json.dumps({
        "frame": "scene metric frame (first left camera, ROS body axes), metres",
        "band": args.band,
        "nodes": out_nodes,
    }))
    print(f"[SKEL] {len(out_nodes)} nodes, {leaves} strand tips, {total:.2f} m of stalk + strands")
    if args.export:
        export_tree_without_bunch(split, out_nodes, positions, args)

    # Previews: skeleton over two real frames.
    cameras = read_cameras(args.scene / "sparse_metric_txt" / "cameras.txt")
    views = {v["name"]: v for v in read_images(args.scene / "sparse_metric_txt" / "images.txt")}
    lefts = sorted(n for n in views if n.startswith("left/"))
    for name in (lefts[0], lefts[len(lefts) // 3]):
        view = views[name]
        fx, fy, cx, cy, _, _ = cameras[int(view["fields"][8])]
        image = (cv2.imread(str(args.scene / "images" / name)) * 0.5).astype(np.uint8)

        def project(p):
            c = view["R"] @ p + view["t"]
            return (int(fx * c[0] / c[2] + cx), int(fy * c[1] / c[2] + cy)) if c[2] > 0.05 else None

        for node in out_nodes:
            if node["parent"] >= 0:
                a, b = project(np.array(node["position"])), project(np.array(out_nodes[node["parent"]]["position"]))
                if a and b:
                    cv2.line(image, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        for n, node in enumerate(out_nodes):
            p = project(np.array(node["position"]))
            if p:
                cv2.circle(image, p, 3, (0, 0, 255) if node["parent"] < 0 else (0, 200, 0), -1)
        cv2.imwrite(str(split / f"bunch_skeleton_{Path(name).stem}.jpg"), image)


if __name__ == "__main__":
    main()
