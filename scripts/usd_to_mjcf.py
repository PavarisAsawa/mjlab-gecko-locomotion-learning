"""Convert the gecko USD (Isaac Sim / UsdPhysics) into MJCF + STL meshes for MuJoCo.

MuJoCo cannot load USD, so re-run this whenever the USD changes. It needs Pixar's
`usd-core`, which is not a project dependency; run it in a throwaway env:

  uv run --no-project --python 3.12 --with usd-core --with numpy \
    python scripts/usd_to_mjcf.py \
    src/mj_lab_gecko_project/robots/gecko/gecko-aug-2026.usd \
    src/mj_lab_gecko_project/robots/gecko/xmls

What is converted:
  - Rigid bodies (PhysicsRigidBodyAPI) -> <body>, tree built from the joints.
  - PhysicsRevoluteJoint -> <joint type="hinge"> with range (deg -> rad).
  - D6 PhysicsJoint with free rotX/rotY (feet) -> two passive hinges.
  - PhysicsMassAPI -> <inertial> (bodies without it get inertia from collision
    geoms at density 1000 kg/m^3, matching PhysX's default).
  - visuals/ meshes -> visual geoms; collisions/ meshes -> collision geoms
    (MuJoCo uses their convex hull, like physics:approximation=convexHull).
Actuators are NOT written here; mjlab adds them from gecko_constants.py.
"""

import argparse
import re
import struct
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT_BODY_HEIGHT_MARGIN = 0.005  # Spawn the root this far above the lowest point.
FOOT_JOINT_ARMATURE = 1e-4  # Passive foot hinges: tiny inertia needs regularizing.
FOOT_JOINT_DAMPING = 0.01
# Foot pads are discs. A ~700-vertex convex hull resting flat on the ground makes
# the contact points flicker and the passive ankles chatter (~1 rad/s at rest); a
# cylinder fitted to the pad is contact-stable. Set False to keep the mesh.
FOOT_COLLISION_AS_CYLINDER = True
# Collision uses each mesh's convex hull; CAD meshes give hulls with thousands of
# vertices, which makes GPU (MuJoCo Warp) collision slow. Cap the hull size.
MAX_HULL_VERT = 64


def _mat(m: Gf.Matrix4d) -> np.ndarray:
  return np.array(m, dtype=np.float64)


def _fmt(v, prec: int = 6) -> str:
  return " ".join(f"{x:.{prec}g}" for x in v)


def _quat_wxyz(q) -> list[float]:
  return [q.GetReal(), *q.GetImaginary()]


def _rel_pose(child_world: Gf.Matrix4d, parent_world: Gf.Matrix4d):
  """Pose of child in parent frame as (pos, quat_wxyz). USD uses row vectors."""
  rel = Gf.Transform(child_world * parent_world.GetInverse())
  return list(rel.GetTranslation()), _quat_wxyz(rel.GetRotation().GetQuat())


def _write_stl(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
  v = verts[tris]  # (F, 3, 3)
  n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
  n /= np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-12)
  rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
  rec["n"], rec["v"] = n, v
  with open(path, "wb") as f:
    f.write(b"\0" * 80 + struct.pack("<I", len(tris)) + rec.tobytes())


def _mesh_in_body(mesh: UsdGeom.Mesh, body_world: Gf.Matrix4d, cache):
  pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
  counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
  idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
  # Fan-triangulate polygons.
  tris, start = [], 0
  for c in counts:
    tris.extend((idx[start], idx[start + k], idx[start + k + 1]) for k in range(1, c - 1))
    start += c
  tris = np.array(tris, dtype=np.int64)
  m = _mat(cache.GetLocalToWorldTransform(mesh.GetPrim()) * body_world.GetInverse())
  pts = np.c_[pts, np.ones(len(pts))] @ m
  return pts[:, :3], tris


def _material_rgba(prim) -> str:
  mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
  if mat:
    m = re.fullmatch(r"material_([0-9A-Fa-f]{6})", mat.GetPrim().GetName())
    if m:
      h = m.group(1)
      return _fmt([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0], 3)
  return "0.6 0.6 0.6 1"


def convert(usd_path: Path, out_dir: Path) -> None:
  stage = Usd.Stage.Open(str(usd_path))
  assert UsdGeom.GetStageUpAxis(stage) == "Z", "Expected Z-up stage"
  assert abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-9, "Expected meters"
  cache = UsdGeom.XformCache()
  pred = Usd.TraverseInstanceProxies()

  bodies = {
    str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
  }
  name = {path: p.GetName() for path, p in bodies.items()}
  world = {path: cache.GetLocalToWorldTransform(p) for path, p in bodies.items()}

  # Joints define the kinematic tree.
  parent, joints_of = {}, {path: [] for path in bodies}
  for p in stage.Traverse():
    if not p.IsA(UsdPhysics.Joint):
      continue
    j = UsdPhysics.Joint(p)
    b0 = str(j.GetBody0Rel().GetTargets()[0])
    b1 = str(j.GetBody1Rel().GetTargets()[0])
    parent[b1] = b0
    joints_of[b1].append(p)
  roots = [b for b in bodies if b not in parent]
  assert len(roots) == 1, f"Expected one root body, got {roots}"
  root = roots[0]
  children = {b: [c for c in bodies if parent.get(c) == b] for b in bodies}

  # Check joint frames agree with the authored body poses (zero joint angle).
  max_err = 0.0
  for child, jps in joints_of.items():
    for jp in jps:
      j = UsdPhysics.Joint(jp)
      f0 = Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())), Gf.Vec3d(j.GetLocalPos0Attr().Get()))
      f1 = Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(j.GetLocalRot1Attr().Get())), Gf.Vec3d(j.GetLocalPos1Attr().Get()))
      w0 = (f0 * world[parent[child]]).ExtractTranslation()
      w1 = (f1 * world[child]).ExtractTranslation()
      max_err = max(max_err, (w0 - w1).GetLength())
  print(f"max joint-frame position mismatch: {max_err * 1e3:.3f} mm")

  assets = out_dir / "assets"
  assets.mkdir(parents=True, exist_ok=True)
  mesh_xml, lowest_z = [], 0.0

  def geoms_for(body_path: str) -> list[str]:
    nonlocal lowest_z
    prim, bw = bodies[body_path], world[body_path]
    out, written = [], {}
    for kind in ("visuals", "collisions"):
      sub = prim.GetChild(kind)
      if not sub:
        continue
      meshes = [p for p in Usd.PrimRange(sub, pred) if p.IsA(UsdGeom.Mesh)]
      for i, mp in enumerate(meshes):
        verts, tris = _mesh_in_body(UsdGeom.Mesh(mp), bw, cache)
        key = (verts.shape, tris.shape, round(float(verts.sum()), 6))
        if key in written:  # Collision mesh identical to visual: reuse the file.
          mesh_name = written[key]
        else:
          suffix = "" if len(meshes) == 1 else f"_{i}"
          mesh_name = f"{name[body_path]}{suffix}" + ("" if kind == "visuals" else "_col")
          _write_stl(assets / f"{mesh_name}.stl", verts, tris)
          mesh_xml.append(f'    <mesh name="{mesh_name}" file="{mesh_name}.stl"/>')
          written[key] = mesh_name
        if kind == "visuals":
          out.append(f'<geom class="visual" mesh="{mesh_name}" rgba="{_material_rgba(mp)}"/>')
        else:
          gname = f"{name[body_path]}_collision" + ("" if len(meshes) == 1 else str(i))
          if FOOT_COLLISION_AS_CYLINDER and name[body_path].startswith("foot_"):
            lo, hi = verts.min(0), verts.max(0)
            radius, half_h = max(hi[:2] - lo[:2]) / 2, (hi[2] - lo[2]) / 2
            out.append(
              f'<geom name="{gname}" class="collision" type="cylinder" '
              f'size="{radius:.5g} {half_h:.5g}" pos="{_fmt((lo + hi) / 2)}"/>'
            )
          else:
            out.append(f'<geom name="{gname}" class="collision" mesh="{mesh_name}"/>')
          wz = (np.c_[verts, np.ones(len(verts))] @ _mat(bw))[:, 2]
          lowest_z = min(lowest_z, float(wz.min()))
    return out

  def joint_xml(jp) -> list[str]:
    j = UsdPhysics.Joint(jp)
    pos = list(j.GetLocalPos1Attr().Get())
    rot1 = Gf.Rotation(Gf.Quatd(j.GetLocalRot1Attr().Get()))
    axes = {a: rot1.TransformDir(Gf.Vec3d(*v)) for a, v in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1)))}
    jname = jp.GetName()
    if jp.IsA(UsdPhysics.RevoluteJoint):
      r = UsdPhysics.RevoluteJoint(jp)
      lo, hi = np.radians([r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()])
      ax = axes[r.GetAxisAttr().Get()]
      return [f'<joint name="{jname}" pos="{_fmt(pos)}" axis="{_fmt(ax)}" range="{lo:.6g} {hi:.6g}"/>']
    # Generic D6 joint: a DOF is free when its limit has low <= high (low > high = locked).
    out = []
    for dof in ("rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
      lim = UsdPhysics.LimitAPI(jp, dof)
      if not jp.HasAPI(UsdPhysics.LimitAPI, dof):
        raise NotImplementedError(f"{jp.GetPath()}: unlimited {dof} not supported")
      lo, hi = lim.GetLowAttr().Get(), lim.GetHighAttr().Get()
      if lo > hi:
        continue
      if not dof.startswith("rot"):
        raise NotImplementedError(f"{jp.GetPath()}: free {dof} not supported")
      lo, hi = np.radians([lo, hi])
      out.append(
        f'<joint name="{jname}_{dof[-1].lower()}" class="passive" pos="{_fmt(pos)}" '
        f'axis="{_fmt(axes[dof[-1]])}" range="{lo:.6g} {hi:.6g}"/>'
      )
    return out

  def inertial_xml(body_path: str) -> list[str]:
    prim = bodies[body_path]
    if not prim.HasAPI(UsdPhysics.MassAPI):
      return []
    m = UsdPhysics.MassAPI(prim)
    return [
      f'<inertial pos="{_fmt(m.GetCenterOfMassAttr().Get())}" '
      f'quat="{_fmt(_quat_wxyz(m.GetPrincipalAxesAttr().Get()))}" '
      f'mass="{m.GetMassAttr().Get():.6g}" '
      f'diaginertia="{_fmt(m.GetDiagonalInertiaAttr().Get())}"/>'
    ]

  def body_xml(body_path: str, depth: int) -> list[str]:
    ind = "  " * depth
    if body_path == root:
      head = f'<body name="{name[body_path]}" pos="0 0 {{root_z}}" childclass="gecko">'
      inner = ['<freejoint name="floating_base_joint"/>', f'<site name="imu" pos="0 0 0"/>']
    else:
      pos, quat = _rel_pose(world[body_path], world[parent[body_path]])
      head = f'<body name="{name[body_path]}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">'
      inner = []
      for jp in joints_of[body_path]:
        inner += joint_xml(jp)
      if name[body_path].startswith("foot_"):
        inner.append(f'<site name="{name[body_path]}" pos="0 0 0" size="0.005"/>')
    inner += inertial_xml(body_path) + geoms_for(body_path)
    lines = [ind + head] + [ind + "  " + s for s in inner]
    for c in children[body_path]:
      lines += body_xml(c, depth + 1)
    return lines + [ind + "</body>"]

  body_lines = body_xml(root, 2)
  root_z = ROOT_BODY_HEIGHT_MARGIN - lowest_z
  body_lines = [s.replace("{root_z}", f"{root_z:.4g}") for s in body_lines]

  xml = f"""<mujoco model="gecko">
  <!-- Generated by scripts/usd_to_mjcf.py from {usd_path.name}. Do not edit by hand. -->
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <default>
    <mesh maxhullvert="{MAX_HULL_VERT}"/>
    <default class="gecko">
      <site group="5" rgba="1 0 0 1"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" contype="1" conaffinity="1" rgba="0.8 0.3 0.3 0.4"/>
      </default>
      <default class="passive">
        <joint armature="{FOOT_JOINT_ARMATURE}" damping="{FOOT_JOINT_DAMPING}"/>
      </default>
    </default>
  </default>

  <asset>
{chr(10).join(mesh_xml)}
  </asset>

  <worldbody>
{chr(10).join(body_lines)}
  </worldbody>
</mujoco>
"""
  (out_dir / "gecko.xml").write_text(xml)
  print(f"wrote {out_dir / 'gecko.xml'} ({len(bodies)} bodies, {len(mesh_xml)} meshes), root z = {root_z:.4f} m")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("usd", type=Path)
  ap.add_argument("out_dir", type=Path)
  a = ap.parse_args()
  convert(a.usd, a.out_dir)
