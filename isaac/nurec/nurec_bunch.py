"""Dynamic date bunch for Isaac Sim: stalk and strands as sprung chains, dates hanging on them.

From SCENE/split/bunch_skeleton.json (bunch_skeleton.py). The fine skeleton (~2.5 cm segments)
is grouped into sections of about section_length, cut at branch points and tips; each section
is one rigid body. The fine segments of a section are attached to it rigidly as tube visuals and
capsule colliders, so strands keep their captured curves while the solver only handles a few
dozen bodies (263 separately jointed segments never settled; a single reduced-coordinate
articulation was far too slow).

Each section hangs from its parent section (the first ones from the world at the stalk root)
on a joint with locked translation and sprung rotation. The captured shape already hangs under
the dates' weight, so each spring is preloaded to balance that moment there: the bunch starts
at rest in its real shape and keeps soft, realistic compliance when pushed.
Strand bodies have no gravity of their own (real strands hold their shape by wood stiffness
and weigh little), so the springs carry the dates, which hang from their nearest section on
their breakable stem joints.

Use the matching split/tree_nobunch_nurec.usdz as the static tree. Call add_bunch() before
sim.reset().
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

from nurec_dates import PhotoPalette, _matrix_to_quat, _quat_to_matrix, add_dates

GRAVITY = 9.81
WOOD_DENSITY = 600.0  # kg/m^3


def _segment_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation whose x axis runs from a to b (capsule/tube axis)."""
    x = (b - a) / np.linalg.norm(b - a)
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)


def _set_local_pose(xformable, position: np.ndarray, rotation: np.ndarray) -> None:
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
    xformable.AddOrientOp().Set(Gf.Quatf(*_matrix_to_quat(rotation)))


def _tube(stage, path: str, length: float, radius: float, colour_srgb, palette, sides: int = 8):
    """Tube along x centred at its origin, emitting its captured colour (like the dates)."""
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), radius * np.cos(angles), radius * np.sin(angles)], 1)
    points = np.vstack([ring + [-length / 2, 0, 0], ring + [length / 2, 0, 0]])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [i, j, sides + j, sides + i]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points.tolist()])
    mesh.CreateFaceVertexCountsAttr([4] * sides)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    palette.apply(mesh, np.tile(np.clip(colour_srgb, 0, 1), (sides, 1)), [4] * sides)
    return mesh


def _sections(parent: np.ndarray, world: np.ndarray, section_length: float):
    """Keep the root, branch points, tips and a node every ~section_length along chains.

    Returns (kept flags, coarse parent of each kept node, owning section node of every node).
    """
    children = [[] for _ in parent]
    for n, p in enumerate(parent):
        if p >= 0:
            children[p].append(n)
    kept = np.zeros(len(parent), bool)
    run = np.zeros(len(parent))  # chain length since the last kept ancestor
    for n in range(len(parent)):  # parents precede children
        p = parent[n]
        if p < 0:
            kept[n] = True
            continue
        run[n] = (0.0 if kept[p] else run[p]) + np.linalg.norm(world[n] - world[p])
        kept[n] = len(children[n]) != 1 or run[n] >= section_length
    coarse_parent = np.full(len(parent), -1)
    for n in range(len(parent)):
        if kept[n] and parent[n] >= 0:
            a = parent[n]
            while not kept[a]:
                a = parent[a]
            coarse_parent[n] = a
    owner = np.full(len(parent), -1)
    for n in reversed(range(len(parent))):  # children before parents
        owner[n] = n if kept[n] else owner[children[n][0]]
    return kept, coarse_parent, owner


def add_bunch(
    stage,
    split_dir: Path,
    scene_pose: list[float],
    date_mass: float = 0.01,
    break_force: float = 5.0,
    section_length: float = 0.06,
    sag_rad: float = 0.0005,
    min_stiffness: float = 4.0,
    damping_ratio: float = 1.0,
    root: str = "/World/Bunch",
    visual: bool = True,
    preload: bool = True,
    min_section_mass: float = 0.02,
    tip_radius: float = 0.0018,
    stalk_radius: float = 0.008,
    body_damping: float = 6.0,
    stalk_stiffness: float = 0.1,
    drive_frequency: float = 0.0,
    solver_iterations: int = 255,
    contact_report: bool = False,
) -> dict:
    """Spawn the dynamic bunch and its dates. Returns section paths, date paths, date -> section."""
    split_dir = Path(split_dir)
    skeleton = json.loads((split_dir / "bunch_skeleton.json").read_text())
    physics = json.loads((split_dir / "dates_physics.json").read_text())
    nodes = skeleton["nodes"]
    r_ws = _quat_to_matrix(*scene_pose[3:])
    p_ws = np.array(scene_pose[:3])
    world = np.array([r_ws @ np.array(n["position"]) + p_ws for n in nodes])
    parent = np.array([n["parent"] for n in nodes])
    # Thickness from the branching, not the splat: the point spread around a strand includes the
    # Gaussians' fuzz (3-5x too thick). Tapers from tip_radius at the tips to stalk_radius at the
    # root with the square root of how much of the bunch hangs below.
    below = np.ones(len(nodes))
    for n in reversed(range(len(nodes))):
        if parent[n] >= 0:
            below[parent[n]] += below[n]
    radius_of = tip_radius + (stalk_radius - tip_radius) * np.sqrt(below / below.max())

    kept, coarse_parent, owner = _sections(parent, world, section_length)
    sections = [n for n in range(len(nodes)) if kept[n] and coarse_parent[n] >= 0]
    section_of = {n: i for i, n in enumerate(sections)}
    a = world[coarse_parent[sections]]
    b = world[sections]
    centre = (a + b) / 2
    frames = [_segment_frame(a[i], b[i]) for i in range(len(sections))]
    fine = [n for n in range(len(nodes)) if parent[n] >= 0]  # fine segment n: parent[n] -> n
    fine_section = {n: section_of[owner[n]] for n in fine}

    # Dates hang from the section owning their nearest fine segment.
    fa, fb = world[parent[fine]], world[fine]
    date_world = {d["id"]: r_ws @ np.array(d["centre"]) + p_ws for d in physics["dates"]}
    date_section = {}
    for date_id, c in date_world.items():
        axis = fb - fa
        t = np.clip(((c - fa) * axis).sum(1) / np.maximum((axis * axis).sum(1), 1e-9), 0, 1)
        nearest = int(np.argmin(np.linalg.norm(fa + t[:, None] * axis - c, axis=1)))
        date_section[date_id] = fine_section[fine[nearest]]

    # Gravity moment each joint carries: the dates in its subtree (strand bodies have no gravity).
    subtree = {i: [i] for i in range(len(sections))}
    for i in reversed(range(len(sections))):
        p = coarse_parent[sections[i]]
        if p in section_of:
            subtree[section_of[p]] += subtree[i]
    dates_on = {i: [] for i in range(len(sections))}
    for date_id, i in date_section.items():
        dates_on[i].append(date_world[date_id])

    UsdGeom.Xform.Define(stage, root)
    joints_root = f"{root}/joints"
    UsdGeom.Scope.Define(stage, joints_root)
    group = UsdPhysics.CollisionGroup.Define(stage, f"{root}/collision_group")
    group.CreateFilteredGroupsRel().AddTarget(group.GetPath())  # overlapping strands must not push apart
    group.CreateFilteredGroupsRel().AddTarget(Sdf.Path("/World/Dates/collision_group"))  # created by add_dates
    palette = PhotoPalette(stage, f"{root}/material", split_dir / "photo_palette_strands.png") if visual else None

    fine_of = {i: [m for m in fine if fine_section[m] == i] for i in range(len(sections))}
    section_mass = np.zeros(len(sections))
    for i in range(len(sections)):
        for f in fine_of[i]:
            length = float(np.linalg.norm(world[f] - world[parent[f]]))
            r = float(min(radius_of[f], max(length / 2, 0.002)))
            section_mass[i] += WOOD_DENSITY * np.pi * r * r * length
    # Strand sections are far lighter than a 10 g date; that mass ratio on stiff springs keeps the
    # solver from settling. They have no gravity, so extra inertia changes no static shape.
    section_mass = np.maximum(section_mass, min_section_mass)

    paths, stiffness_used, total_mass = [], [], 0.0
    for i, n in enumerate(sections):
        path = f"{root}/section_{i:03d}"
        paths.append(path)
        body = UsdGeom.Xform.Define(stage, path)
        _set_local_pose(body, centre[i], frames[i])
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx.CreateDisableGravityAttr(True)
        physx.CreateSolverPositionIterationCountAttr(solver_iterations)
        physx.CreateSolverVelocityIterationCountAttr(4)
        # Air/fibre drag on each body: calms the swing to a gentle sway. Unlike more joint damping
        # (which destabilises the light strand tips) this is unconditionally stable.
        physx.CreateLinearDampingAttr(body_damping)
        physx.CreateAngularDampingAttr(body_damping)

        for k, f in enumerate(fine_of[i]):
            fa_, fb_ = world[parent[f]], world[f]
            length = float(np.linalg.norm(fb_ - fa_))
            r = float(min(radius_of[f], max(length / 2, 0.002)))
            local_centre = frames[i].T @ ((fa_ + fb_) / 2 - centre[i])
            local_rot = frames[i].T @ _segment_frame(fa_, fb_)
            capsule = UsdGeom.Capsule.Define(stage, f"{path}/collider_{k:02d}")
            capsule.CreateAxisAttr("X")
            capsule.CreateRadiusAttr(r)
            capsule.CreateHeightAttr(max(length - 2 * r, 0.0))
            capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)
            capsule.MakeInvisible()  # the camera renders guide prims (black); physics ignores visibility
            _set_local_pose(capsule, local_centre, local_rot)
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(capsule.GetPath())
            if visual:
                # Slightly longer than the segment so neighbours overlap at bends (no dark gaps).
                tube = _tube(stage, f"{path}/visual_{k:02d}", length + 2 * r, r, nodes[f]["colour_rgb"], palette)
                _set_local_pose(tube, local_centre, local_rot)
        mass = float(section_mass[i])
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        total_mass += mass

        # Spring joint where this section meets its parent.
        loads = [c for s in subtree[i] for c in dates_on[s]]
        moment = np.zeros(3)
        for c in loads:
            moment += np.cross(c - a[i], [0.0, 0.0, -date_mass * GRAVITY])
        stiffness = max(min_stiffness, float(np.linalg.norm(moment)) / sag_rad)
        # Damping from this section and the dates it carries. (Sizing it from the whole subtree's
        # inertia, which is physically fuller, made the stiff joints near the stalk unstable.)
        inertia = sum(date_mass * float(np.sum((c - a[i]) ** 2)) for c in loads)
        inertia += mass * float(np.sum((b[i] - a[i]) ** 2)) / 3
        damping = 2 * damping_ratio * math.sqrt(stiffness * inertia)
        stiffness_used.append(stiffness)

        joint = UsdPhysics.Joint.Define(stage, f"{joints_root}/j_{i:03d}")
        p = coarse_parent[n]
        if p in section_of:
            j = section_of[p]
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{root}/section_{j:03d}")])  # parents come first
            local0, rot0 = frames[j].T @ (a[i] - centre[j]), frames[j].T @ frames[i]
        else:  # hangs from the world at the stalk root
            local0, rot0 = a[i], frames[i]
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*local0.tolist()))
        joint.CreateLocalRot0Attr(Gf.Quatf(*_matrix_to_quat(rot0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-float(np.linalg.norm(b[i] - a[i])) / 2, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        for axis in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)  # low > high: locked
        # The captured shape already hangs under the dates' weight: preload each spring so it
        # balances that moment at the captured shape (target = -moment / stiffness, joint frame),
        # instead of sagging further or being made unrealistically stiff.
        if drive_frequency > 0.0:
            # Acceleration drives scale with the joint's effective inertia, so every section rings
            # at drive_frequency: stiff woody strands stay stable where force springs stiffer than
            # ~1 N m/rad on these light bodies diverged.
            omega = 2.0 * math.pi * drive_frequency
            drive_type, gain_k, gain_d = "acceleration", omega ** 2, 2.0 * damping_ratio * omega
            preload_rad = -(frames[i].T @ moment) / (gain_k * max(inertia, 1e-6)) if preload else np.zeros(3)
        else:
            drive_type, gain_k, gain_d = "force", stiffness, damping
            preload_rad = -(frames[i].T @ moment) / stiffness if preload else np.zeros(3)
        for k_axis, axis in enumerate(("rotX", "rotY", "rotZ")):
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr(drive_type)
            drive.CreateTargetPositionAttr(float(math.degrees(preload_rad[k_axis])))
            # USD angular drive gains are per degree.
            drive.CreateStiffnessAttr(float(gain_k * math.pi / 180.0))
            drive.CreateDampingAttr(float(gain_d * math.pi / 180.0))

    if palette is not None:
        palette.save()
    attach = {date_id: (paths[i], frames[i], centre[i]) for date_id, i in date_section.items()}
    dates = add_dates(stage, split_dir / "dates_physics.json", scene_pose, mass=date_mass, break_force=break_force,
                      visual_mode="mesh" if visual else "none", attach=attach, stalk_stiffness=stalk_stiffness,
                      contact_report=contact_report)
    stiffness_used = np.array(stiffness_used)
    print(f"[BUNCH] {len(sections)} sections from {len(fine)} skeleton segments "
          f"({np.linalg.norm(fb - fa, axis=1).sum():.2f} m, {total_mass*1000:.0f} g), {len(dates)} dates; "
          f"joint stiffness {stiffness_used.min():.2f}-{stiffness_used.max():.2f} N m/rad")
    return {"segments": paths, "dates": dates, "date_segment": {d: paths[i] for d, i in date_section.items()}}
