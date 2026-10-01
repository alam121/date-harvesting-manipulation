"""Detachable NuRec dates for Isaac Sim: one rigid body per date, pinned by a breakable joint.

Authors USD physics directly (call before sim.reset()):
  /World/Dates/date_XX            rigid body (mass --date_mass) at the date centre
    /visual                       the date's own NuRec asset (split_dates.py), so it looks real
    /collider                     capsule along the date's long axis
  /World/DateStems/date_XX         fixed joint to the world with physics:breakForce / breakTorque

Dates are in one collision group that ignores itself (touching neighbours would otherwise push
apart at start-up); they still collide with the gripper and the ground. The tree stays a static
visual (split/tree_nurec.usdz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


def _quat_to_matrix(w, x, y, z) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> (w, x, y, z)."""
    trace = np.trace(r)
    if trace > 0:
        k = 2.0 * np.sqrt(trace + 1.0)
        q = (0.25 * k, (r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k, (r[1, 0] - r[0, 1]) / k)
    else:
        i = int(np.argmax(np.diag(r)))
        j, l = (i + 1) % 3, (i + 2) % 3
        k = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[l, l])
        q = [0.0] * 4
        q[0] = (r[l, j] - r[j, l]) / k
        q[1 + i] = 0.25 * k
        q[1 + j] = (r[j, i] + r[i, j]) / k
        q[1 + l] = (r[l, i] + r[i, l]) / k
    q = np.array(q) / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))


class PhotoPalette:
    """Unlit photo colours for the date and strand meshes, through one palette texture.

    The colour is emitted, like the NuRec splats (whose colours already contain the real lighting),
    instead of being lit a second time by the scene lights. Each face reads one block of the
    palette through ordinary UVs ("st"). An emissive material reading the colours from a custom
    per-vertex primvar rendered every mesh solid black in some runs (cause not found; re-authoring
    the primvar after start-up did not help); UV textures are the path every textured asset takes.
    Faces are flat-coloured, which at a few millimetres per face is not visible.

    add meshes with apply(), then save() once: it sorts the colours by brightness (so mip levels
    blend similar colours), writes the PNG and sets every mesh's UVs.
    """

    BLOCK = 4  # texels per colour side: neighbouring colours stay apart down to mip level 2

    def __init__(self, stage, path: str, png: Path, gain: float = 1.0):
        self.png = Path(png)
        self.meshes: list[tuple[UsdGeom.Mesh, list[tuple[int, int, int]], list[int]]] = []
        self.material = UsdShade.Material.Define(stage, path)
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"{path}/palette")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(self.png))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(gain, gain, gain, 1.0))
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface = UsdShade.Shader.Define(stage, f"{path}/surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        # No reflections either (the default IOR specular mirrors the dome light as a blue-grey sheen).
        surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        surface.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        self.material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    def apply(self, mesh: UsdGeom.Mesh, face_colours_srgb: np.ndarray, face_sizes) -> None:
        """face_colours_srgb: (F, 3) in [0, 1]; face_sizes: vertices per face."""
        # 6 bits per channel: invisible, and far fewer palette entries.
        quantised = np.clip(np.round(np.asarray(face_colours_srgb) * 63), 0, 63).astype(int)
        self.meshes.append((mesh, [tuple(c) for c in quantised.tolist()], list(face_sizes)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material)

    def save(self) -> None:
        import cv2

        colours = sorted({c for _, faces, _ in self.meshes for c in faces}, key=lambda c: (sum(c), c))
        index = {c: i for i, c in enumerate(colours)}
        per_side = int(np.ceil(np.sqrt(max(len(colours), 1))))
        size = per_side * self.BLOCK
        image = np.zeros((size, size, 3), np.uint8)
        for i, colour in enumerate(colours):
            row, col = divmod(i, per_side)
            image[row * self.BLOCK:(row + 1) * self.BLOCK, col * self.BLOCK:(col + 1) * self.BLOCK] = \
                np.round(np.array(colour) * 255 / 63)
        self.png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.png), image[..., ::-1])
        for mesh, faces, sizes in self.meshes:
            uv = []
            for colour, n in zip(faces, sizes):
                row, col = divmod(index[colour], per_side)
                # Block centre; USD st has its origin at the bottom-left of the image.
                u = (col + 0.5) / per_side
                v = 1.0 - (row + 0.5) / per_side
                uv += [Gf.Vec2f(u, v)] * n
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uv)


def _add_textured_mesh(stage, path: str, mesh_file: Path, palette: PhotoPalette) -> None:
    data = np.load(mesh_file)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in data["vertices"].tolist()])
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in data["normals"].tolist()])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([3] * len(data["faces"]))
    mesh.CreateFaceVertexIndicesAttr(data["faces"].reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    vertex_colours = np.clip(data["colours_srgb"], 0, 1)
    palette.apply(mesh, vertex_colours[data["faces"]].mean(1), [3] * len(data["faces"]))


def add_dates(
    stage,
    manifest_path: Path,
    scene_pose: list[float],
    mass: float = 0.01,
    break_force: float = 5.0,
    break_torque: float = 1.0e6,
    root: str = "/World/Dates",
    stem_root: str = "/World/DateStems",
    visual_mode: str = "mesh",
    emissive_gain: float = 1.0,
    attach: dict | None = None,
    friction: float = 1.0,
    contact_report: bool = False,
) -> list[str]:
    """Spawn the dates of dates_physics.json; scene_pose is the map's --nurec_pose (x y z qw qx qy qz).

    visual_mode "mesh" uses the photo-textured meshes from texture_dates.py (composite correctly
    with the NuRec tree); "nurec" uses the per-date NuRec volumes, which Isaac Sim 5.1 hides
    whenever the camera is inside the tree volume's bounds; "none" adds physics only (tests).

    attach maps date id -> (body path, body world rotation 3x3, body world position): the date's
    stem joint then holds it to that body (a strand segment, nurec_bunch.py) instead of the world.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    directory = Path(manifest_path).parent
    r_world_scene = _quat_to_matrix(*scene_pose[3:])
    p_world_scene = np.array(scene_pose[:3])
    q_scene = Gf.Quatf(float(scene_pose[3]), Gf.Vec3f(*map(float, scene_pose[4:])))

    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Scope.Define(stage, stem_root)  # joints live apart so "/World/Dates/*" matches only bodies
    group = UsdPhysics.CollisionGroup.Define(stage, f"{root}/collision_group")
    group.CreateFilteredGroupsRel().AddTarget(group.GetPath())  # dates never collide with each other
    palette = PhotoPalette(stage, f"{root}/material", directory / "photo_palette_dates.png",
                           emissive_gain) if visual_mode == "mesh" else None
    # Date skin against the Delto's rubber pads; at PhysX's default 0.5 a detached date slid out of
    # the closed hand. "max" so the fingers' default material cannot lower it.
    grip = UsdShade.Material.Define(stage, f"{root}/physics_material")
    grip_api = UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
    grip_api.CreateStaticFrictionAttr(friction)
    grip_api.CreateDynamicFrictionAttr(0.9 * friction)
    grip_api.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(grip.GetPrim()).CreateFrictionCombineModeAttr("max")

    bodies = []
    for date in manifest["dates"]:
        path = f"{root}/date_{date['id']:02d}"
        body = UsdGeom.Xform.Define(stage, path)
        position = r_world_scene @ np.array(date["centre"]) + p_world_scene
        body.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
        # The date asset keeps the scene's orientation (it was only re-centred), so the body does too.
        body.AddOrientOp().Set(q_scene)
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        if contact_report:  # for a ContactSensor on the dates (which robot link touches which date)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

        if visual_mode == "mesh":
            _add_textured_mesh(stage, f"{path}/visual", directory / date["mesh"], palette)
        elif visual_mode == "nurec":
            visual = UsdGeom.Xform.Define(stage, f"{path}/visual")
            visual.GetPrim().GetReferences().AddReference(str(directory / date["asset"]))

        radii = np.array(date["collider_radii"])
        radius = float(np.mean(radii[1:]))
        capsule = UsdGeom.Capsule.Define(stage, f"{path}/collider")
        capsule.CreateAxisAttr("X")  # collider frame x = the date's long principal axis
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(max(2.0 * float(radii[0]) - 2.0 * radius, 0.0))
        capsule.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only
        capsule.MakeInvisible()  # the camera renders guide prims (black); physics ignores visibility
        capsule.AddOrientOp().Set(Gf.Quatf(*map(float, date["collider_quat_wxyz"])))
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(
            grip, UsdShade.Tokens.weakerThanDescendants, "physics")
        group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(capsule.GetPath())

        # Pin to the world where the date hangs; PhysX removes the joint once the pull exceeds the limits.
        joint = UsdPhysics.FixedJoint.Define(stage, f"{stem_root}/date_{date['id']:02d}")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        if attach and date["id"] in attach:
            body_path, r_world_body, p_world_body = attach[date["id"]]
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
            local = r_world_body.T @ (position - p_world_body)
            r_body_date = r_world_body.T @ r_world_scene
            joint.CreateLocalPos0Attr(Gf.Vec3f(*local.tolist()))
            joint.CreateLocalRot0Attr(Gf.Quatf(*_matrix_to_quat(r_body_date)))
        else:
            joint.CreateLocalPos0Attr(Gf.Vec3f(*position.tolist()))
            joint.CreateLocalRot0Attr(Gf.Quatf(q_scene))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateBreakForceAttr(break_force)
        # PhysX trips breakTorque at pulls of ~0.2-0.5 N for 0.5 N m (not a physical lever arm;
        # see date_pull_test.py), so twist-off is effectively disabled by default.
        joint.CreateBreakTorqueAttr(break_torque)
        bodies.append(path)
    if palette is not None:
        palette.save()
    return bodies
