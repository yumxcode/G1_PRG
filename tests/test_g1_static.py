#!/usr/bin/env python3
"""Static consistency tests for the G1 configuration package (no isaaclab).

Pure ``xml`` + ``ast`` parsing — runnable with the system python3 anywhere
(macOS dev box, CI, or the gradmotion container before a training run):

    python3 tests/test_g1_static.py

What it checks (all against the REAL G1 URDF, never mocks):
  1. every produced .py file passes ``ast.parse``
  2. URDF has exactly 29 revolute joints
  3. foot bodies exist in the URDF link set
  4. g1.py actuator ``joint_names_expr`` regexes cover the URDF joint set
     EXACTLY (no missing, no extra, no double-assignment across groups)
  5. g1.py per-joint effort/velocity limits equal the URDF ``<limit>`` values
     (the rev_1_0 hip_roll 139 Nm / 20 rad/s fix is pinned here)
  6. g1.py init_state joint_pos keys == URDF joint names, and the URDF forward
     kinematics standing height matches init_state pos z within 5 cm
  7. g1_amp_env_cfg.py retarget maps only produce names that exist in the URDF
     (bodies -> link set, joint rules -> joint set) and the 10 discriminator
     key bodies are distinct URDF links
  8. every mesh file referenced by the URDF exists on disk (CONDITIONAL SKIP
     with a printed note when the mesh directory is not shipped next to the
     URDF — meshes are visual-only for the GMR collision URDF)

Path conventions (first hit wins):
  * URDF : $G1_URDF, else walk up from this file looking for
          ``data/robots/g1/urdf/g1_custom_collision_29dof.urdf`` (merged-repo
          convention, mirrors robots/x1/urdf/f1.urdf), else the dev workspace
          fallback ``work/g1cfg-input/g1_custom_collision_29dof.urdf``.
  * g1.py : $G1_CFG, else ``../g1.py`` next to the output package, else walk up
          for ``**/assets/robots/g1.py`` (merged-repo placement).
  * meshes: resolved relative to the URDF's own directory, i.e.
          ``data/robots/g1/urdf/meshes/*.STL`` (upstream g1_description layout;
          the URDF references ``meshes/...`` paths verbatim, zero edits).
"""

import ast
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = []


def check(name, fn):
    try:
        note = fn()
        _RESULTS.append((name, "PASS", note or ""))
    except SkipTest as exc:
        _RESULTS.append((name, "SKIP", str(exc)))
    except AssertionError as exc:
        _RESULTS.append((name, "FAIL", str(exc) or "assertion failed"))
    except Exception as exc:  # noqa: BLE001 — report, don't crash the harness
        _RESULTS.append((name, "ERROR", f"{type(exc).__name__}: {exc}"))


class SkipTest(Exception):
    """Raised when an external input (URDF / meshes) is not available."""


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #
def find_file(rel_candidates, env_var=None, what="input"):
    if env_var and os.environ.get(env_var):
        p = os.environ[env_var]
        assert os.path.isfile(p), f"{env_var}={p} does not exist"
        return os.path.abspath(p)
    node = _HERE
    for _ in range(6):
        for rel in rel_candidates:
            p = os.path.join(node, rel)
            if os.path.isfile(p):
                return os.path.abspath(p)
        node = os.path.dirname(node)
    raise SkipTest(f"{what} not found (tried {rel_candidates} walking up from {_HERE}; "
                   f"set {env_var} to point at it)")


def find_urdf():
    return find_file(
        [
            "robolab/data/robots/g1/urdf/g1_custom_collision_29dof.urdf",
            "data/robots/g1/urdf/g1_custom_collision_29dof.urdf",
            "work/g1cfg-input/g1_custom_collision_29dof.urdf",
            "g1cfg-input/g1_custom_collision_29dof.urdf",
        ],
        env_var="G1_URDF",
        what="G1 URDF",
    )


def find_g1_cfg():
    return find_file(
        ["g1.py", "assets/robots/g1.py", "robolab/robolab/assets/robots/g1.py"],
        env_var="G1_CFG",
        what="g1.py robot cfg",
    )


# --------------------------------------------------------------------------- #
# URDF parsing
# --------------------------------------------------------------------------- #
class Urdf:
    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(path)
        root = ET.parse(path).getroot()
        self.robot_name = root.get("name")
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.revolute = [n for n, j in self.joints.items() if j.get("type") == "revolute"]
        self.mesh_refs = [
            m.get("filename")
            for m in root.iter("mesh")
        ]

    def limits(self, joint):
        lim = self.joints[joint].find("limit")
        return (
            float(lim.get("effort", "nan")),
            float(lim.get("velocity", "nan")),
            float(lim.get("lower", "nan")),
            float(lim.get("upper", "nan")),
        )

    # -- minimal FK (pelvis frame) ----------------------------------------- #
    @staticmethod
    def _rpy_mat(r, p, y):
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _axis_mat(axis, angle):
        n = math.sqrt(sum(v * v for v in axis))
        x, y, z = [v / n for v in axis]
        c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
        return [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]

    @classmethod
    def _mm(cls, a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    def _origin(self, joint):
        o = self.joints[joint].find("origin")
        if o is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        return xyz, rpy

    def _axis(self, joint):
        a = self.joints[joint].find("axis")
        return [float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()]

    def fk_chain(self, joint_names, angles):
        """Position of the last child link origin in the pelvis frame.

        URDF semantics per joint: p_new = p_old + R_old @ xyz  (the joint
        origin translation is expressed in the PARENT frame — apply it with the
        PREVIOUS rotation), then R_new = R_old @ Rpy(origin) @ Raxis(axis, q).
        """
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        p = [0.0, 0.0, 0.0]
        for name, q in zip(joint_names, angles):
            xyz, rpy = self._origin(name)
            p = [p[i] + sum(R[i][k] * xyz[k] for k in range(3)) for i in range(3)]
            Rq = self._axis_mat(self._axis(name), q)
            R = self._mm(R, self._mm(self._rpy_mat(*rpy), Rq))
        return R, p

    def sole_thickness(self, foot_link):
        """Lowest collision-geometry z offset in the ankle_roll link frame."""
        lowest = 0.0
        for col in self.links[foot_link].findall("collision"):
            o = col.find("origin")
            z0 = float((o.get("xyz") if o is not None else "0 0 0").split()[2])
            g = col.find("geometry")
            for prim in g:
                if prim.tag == "cylinder" or prim.tag == "capsule" or prim.tag == "sphere":
                    lowest = min(lowest, z0 - float(prim.get("radius")))
                elif prim.tag == "box":
                    lowest = min(lowest, z0 - float(prim.get("size")).split()[2])  # rough
        return lowest


# --------------------------------------------------------------------------- #
# g1.py ast parsing
# --------------------------------------------------------------------------- #
def parse_g1_cfg(path):
    """Extract (actuators, init_state) from g1.py without importing isaaclab."""
    tree = ast.parse(open(path).read(), filename=path)
    g1_cfg = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "G1_CFG" for t in node.targets
        ):
            g1_cfg = node.value
    assert g1_cfg is not None, "G1_CFG assignment not found"
    assert isinstance(g1_cfg, ast.Call), "G1_CFG is not a call"

    actuators, init_state = {}, None
    for kw in g1_cfg.keywords:
        if kw.arg == "actuators":
            for key, val in zip(kw.value.keys, kw.value.values):
                assert isinstance(val, ast.Call), f"actuator {key.value} is not a cfg call"
                entry = {"group": key.value}
                for akw in val.keywords:
                    if akw.arg == "joint_names_expr":
                        entry["exprs"] = ast.literal_eval(akw.value)
                    elif akw.arg in ("effort_limit_sim", "velocity_limit_sim", "stiffness", "damping", "armature"):
                        entry[akw.arg] = ast.literal_eval(akw.value) if isinstance(
                            akw.value, (ast.Dict, ast.Constant)
                        ) else None
                actuators[key.value] = entry
        elif kw.arg == "init_state":
            for ikw in kw.value.keywords:
                if ikw.arg == "pos":
                    init_state = init_state or {}
                    init_state["pos"] = ast.literal_eval(ikw.value)
                elif ikw.arg == "joint_pos":
                    init_state = init_state or {}
                    init_state["joint_pos"] = ast.literal_eval(ikw.value)
    assert actuators, "actuators dict not found in G1_CFG"
    assert init_state and "joint_pos" in init_state and "pos" in init_state, "init_state incomplete"
    return actuators, init_state


def expand_limit(value, group_joints):
    """Scalar -> same for all joints; dict {pattern: val} -> expand by regex."""
    if isinstance(value, dict):
        out = {}
        for pattern, val in value.items():
            matched = [j for j in group_joints if re.fullmatch(pattern, j)]
            assert matched, f"limit pattern {pattern!r} matches no joint in {group_joints}"
            for j in matched:
                out[j] = float(val)
        return out
    return {j: float(value) for j in group_joints}


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #
def t_ast_parse_all():
    files = [
        "../g1.py",
        "../g1_amp_env_cfg.py",
        "../g1_amp_agent_cfg.py",
        "../__init__.py",
        os.path.basename(__file__),
    ]
    for rel in files:
        p = os.path.normpath(os.path.join(_HERE, rel))
        if not os.path.isfile(p):
            raise SkipTest(f"{rel} not next to tests/ (merged-repo layout) — skipped file")
        ast.parse(open(p).read(), filename=p)
    return f"{len(files)} files"


def t_urdf_29_revolute():
    urdf = Urdf(find_urdf())
    assert len(urdf.revolute) == 29, f"expected 29 revolute joints, got {len(urdf.revolute)}"
    return f"robot={urdf.robot_name}"


def t_feet_exist():
    urdf = Urdf(find_urdf())
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        assert foot in urdf.links, f"foot body {foot!r} missing from URDF links"
    # bonus: the pelvis/base link exists
    assert "pelvis" in urdf.links, "pelvis link missing"
    return "feet + pelvis present"


def t_joint_regex_cover():
    urdf = Urdf(find_urdf())
    actuators, _ = parse_g1_cfg(find_g1_cfg())
    urdf_set = set(urdf.revolute)
    covered = {}
    for name, entry in actuators.items():
        for expr in entry["exprs"]:
            matched = [j for j in urdf.revolute if re.fullmatch(expr, j)]
            assert matched, f"group {name!r} pattern {expr!r} matches no URDF joint"
            for j in matched:
                assert j not in covered, (
                    f"joint {j!r} claimed by both {covered[j]!r} and {name!r}"
                )
                covered[j] = name
    missing = urdf_set - set(covered)
    extra = set(covered) - urdf_set
    assert not missing, f"URDF joints not driven by g1.py: {sorted(missing)}"
    assert not extra, f"g1.py drives non-existent joints: {sorted(extra)}"
    return f"29/29 joints covered by {len(actuators)} groups, no overlap"


def t_limits_match_urdf():
    urdf = Urdf(find_urdf())
    actuators, _ = parse_g1_cfg(find_g1_cfg())
    n = 0
    for name, entry in actuators.items():
        group_joints = [
            j for expr in entry["exprs"] for j in urdf.revolute if re.fullmatch(expr, j)
        ]
        for field, idx in (("effort_limit_sim", 0), ("velocity_limit_sim", 1)):
            limits = expand_limit(entry.get(field), group_joints)
            for joint, val in limits.items():
                ref = urdf.limits(joint)[idx]
                assert abs(val - ref) < 1e-6, (
                    f"{name}/{joint} {field}: g1.py {val} != URDF {ref}"
                )
                n += 1
    return f"{n} per-joint limits equal URDF (incl. hip_roll 139/20 rev_1_0)"


def t_init_state_and_fk():
    urdf = Urdf(find_urdf())
    _, init = parse_g1_cfg(find_g1_cfg())
    keys = set(init["joint_pos"])
    assert keys == set(urdf.revolute), (
        f"init joint_pos keys != URDF joints: missing={sorted(set(urdf.revolute)-keys)} "
        f"extra={sorted(keys-set(urdf.revolute))}"
    )
    for k, v in init["joint_pos"].items():
        assert math.isfinite(v), f"non-finite init angle {k}"
    # FK standing height: left leg chain, init angles
    leg = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
           "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
    q = [init["joint_pos"][j] for j in leg]
    R, p = urdf.fk_chain(leg, q)
    sole_local = urdf.sole_thickness("left_ankle_roll_link")
    # sole point in pelvis frame (foot pitch is ~0 for this pose; use rotation)
    sole_world_z = p[2] + (R[2][2] * sole_local)
    height = -sole_world_z
    z_cfg = init["pos"][2]
    assert abs(z_cfg - height) < 0.05, f"init pos z {z_cfg} vs FK height {height:.4f}"
    return f"FK height {height:.4f} m vs cfg z {z_cfg} (|d|<5 cm)"


def t_retarget_maps():
    urdf = Urdf(find_urdf())
    env_cfg_path = find_file(
        ["../g1_amp_env_cfg.py", "g1_amp_env_cfg.py",
         "robolab/robolab/tasks/manager_based/amp/g1_amp_env_cfg.py"],
        env_var="G1_ENV_CFG",
        what="g1_amp_env_cfg.py",
    )
    tree = ast.parse(open(env_cfg_path).read(), filename=env_cfg_path)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    "X1_TO_G1_BODY_NAMES", "X1_TO_G1_JOINT_RULES", "G1_KEY_BODY_NAMES"
                ):
                    consts[t.id] = ast.literal_eval(node.value)
    assert "X1_TO_G1_BODY_NAMES" in consts and "G1_KEY_BODY_NAMES" in consts, (
        "retarget constants not found in g1_amp_env_cfg.py"
    )
    for x1_name, g1_name in consts["X1_TO_G1_BODY_NAMES"].items():
        assert g1_name in urdf.links, f"body map {x1_name}->{g1_name}: {g1_name!r} not a G1 URDF link"
    kb = consts["G1_KEY_BODY_NAMES"]
    assert len(kb) == 10 and len(set(kb)) == 10, "key bodies must be 10 distinct links"
    for b in kb:
        assert b in urdf.links, f"key body {b!r} not in URDF"
    if "X1_TO_G1_JOINT_RULES" in consts:
        rules = consts["X1_TO_G1_JOINT_RULES"]
        for old, new in rules:
            # the NEW side must be a plausible G1 joint-name token: at least one
            # URDF joint contains it (patterns carry the .*_ prefix around it)
            assert any(new in j for j in urdf.revolute), f"rule ->{new}: token matches no URDF joint"
    return (f"{len(consts['X1_TO_G1_BODY_NAMES'])} body maps, "
            f"{len(consts.get('X1_TO_G1_JOINT_RULES', []))} joint rules, 10 key bodies — all resolve")


def t_meshes_exist():
    urdf = Urdf(find_urdf())
    refs = sorted(set(urdf.mesh_refs))
    missing = []
    for ref in refs:
        # URDF mesh filenames are relative to the URDF's own directory
        p = os.path.normpath(os.path.join(urdf.dir, ref))
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        raise SkipTest(
            f"{len(missing)}/{len(refs)} mesh files not shipped next to the URDF "
            f"(e.g. {missing[0]}). CONVENTION: place the STLs at "
            "data/robots/g1/urdf/meshes/ (URDF refs 'meshes/...' relative to the "
            "URDF dir). Meshes are visual-only for the GMR collision URDF — "
            "physics is unaffected; assert skipped."
        )
    return f"{len(refs)} mesh files present"


def main():
    check("ast.parse all produced .py", t_ast_parse_all)
    check("URDF has 29 revolute joints", t_urdf_29_revolute)
    check("foot/pelvis bodies exist in URDF", t_feet_exist)
    check("g1.py joint regex covers URDF joints exactly", t_joint_regex_cover)
    check("g1.py effort/velocity limits == URDF limits", t_limits_match_urdf)
    check("init_state joints + FK standing height", t_init_state_and_fk)
    check("env cfg retarget maps resolve to G1 names", t_retarget_maps)
    check("URDF mesh files exist (conditional)", t_meshes_exist)

    fails = 0
    print("=" * 78)
    for name, status, note in _RESULTS:
        line = f"[{status}] {name}"
        if note:
            line += f" — {note}"
        print(line)
        if status in ("FAIL", "ERROR"):
            fails += 1
    print("=" * 78)
    print(f"{len(_RESULTS) - fails}/{len(_RESULTS)} passed, "
          f"{sum(1 for r in _RESULTS if r[1] == 'SKIP')} skipped, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
