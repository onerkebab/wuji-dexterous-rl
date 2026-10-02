"""B5 constant-pelvis placement study. Targets retain the original home camera pose.

Collision distances use the existing collision geoms, including MuJoCo's convex mesh
representation; this is a kinematic hand/forearm-to-body check, not whole-body safety.
Nothing in this module writes the shared retarget archives.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wuji_cap.common import frames
from wuji_cap.common.joint_order import SIDES
from wuji_cap.sim.arm_ik import ArmIK, reach_failure
from wuji_cap.sim.g1_model import G1Wuji


@dataclass
class Clearance:
    """Existing collision geoms of each hand/distal arm against trunk, head and legs."""
    g: G1Wuji

    def __post_init__(self):
        m = self.g.model
        active = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        names = [m.body(b).name for b in m.geom_bodyid]
        self.body = np.array([i for i in active if names[i] in ('pelvis', 'torso_link', 'head_link',
                              'waist_yaw_link', 'waist_roll_link') or any(
            names[i].startswith(s + '_' + part) for s in SIDES for part in ('hip_', 'knee_', 'ankle_'))], int)
        self.moving = {s: np.array([i for i in active if names[i].startswith(s[0] + '_') or any(
            names[i].startswith(s + '_' + part) for part in ('elbow_', 'wrist_'))], int) for s in SIDES}
        if not len(self.body) or any(not len(a) for a in self.moving.values()):
            raise ValueError('missing required collision geometry')
        self.pairs = {s: np.array([(a, b) for a in ids for b in self.body], int) for s, ids in self.moving.items()}

    def minimum(self, data: mujoco.MjData, side: str) -> tuple[float, tuple[int, int]]:
        """Exact minimum with bounding-sphere pruning and a 10 m query cutoff.

        A positive sphere bound can skip a pair once a penetration is found; a
        negative sphere bound cannot bound penetration depth, so overlapping pairs
        are always queried. No contact-mask filtering hides hand/body penetration.
        """
        m = self.g.model
        pairs = self.pairs[side]
        a, b = pairs.T
        bounds = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        best, pair = 10.0, (-1, -1)
        for k in np.argsort(bounds, kind='stable'):
            if bounds[k] > max(best, 0.0):
                break
            i, j = map(int, pairs[k])
            distance = float(mujoco.mj_geomDistance(m, data, i, j, 10.0, None))
            if distance < best:
                best, pair = distance, (i, j)
        return best, pair

    def inventory(self) -> dict:
        m = self.g.model
        def describe(ids):
            return [{'id': int(i), 'body': m.body(m.geom_bodyid[i]).name,
                     'mesh': m.mesh(m.geom_dataid[i]).name if m.geom_dataid[i] >= 0 else None}
                    for i in ids]
        return {'body': describe(self.body), **{s: describe(v) for s, v in self.moving.items()}}


def sample_rows(rt: dict, stride: int) -> np.ndarray:
    """Every stride-th usable row plus EVERY existing failure row, in time order."""
    if stride < 1:
        raise ValueError('stride must be positive')
    usable = rt['left_valid'] | rt['right_valid']
    failure = np.zeros(len(usable), bool)
    for s in SIDES:
        failure |= rt[s + '_valid'] & rt[s + '_reach_failure']
    sampled = np.zeros(len(usable), bool)
    sampled[np.flatnonzero(usable)[::stride]] = True
    return np.flatnonzero((sampled | failure) & usable)


def objective(summary: dict) -> tuple:
    """Lexicographic W1-B16 objective; deterministic tie-break by offset magnitude."""
    d = summary['offset_m']
    return (summary['reach_failure_hand_frames'], summary['body_collision_hand_frames'],
            summary['p95_position_error_m'], float(np.linalg.norm(d)), tuple(d))


def leg_pose(g: G1Wuji, floor_z: float) -> tuple[np.ndarray, dict]:
    """Fit bilateral hip/knee/ankle pitch with flat feet at a specified sole plane.

    Keep each foot's home x/y relative to the translated pelvis. All other leg
    joints and pelvis orientation stay unchanged. This checks kinematics only.
    The original keyframe's sole plane is 0.2208 m above the XML floor; report both.
    """
    m = g.model
    base = G1Wuji()
    q = g.q_home.copy()
    data = mujoco.MjData(m)
    info = {'floor_z_m': float(floor_z), 'sides': {}, 'feasible': True}
    for side in SIDES:
        names = [side + '_' + name + '_joint' for name in ('hip_pitch', 'knee', 'ankle_pitch')]
        js = np.array([m.joint(n).id for n in names])
        adr = m.jnt_qposadr[js]
        foot = m.body(side + '_ankle_roll_link').id
        sole = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] == foot and m.geom_contype[i]], int)
        target_x = base.data.xpos[foot, 0] + g.pelvis_offset[0]
        target_z = floor_z + 0.035  # four official foot spheres: centre z=-0.03, radius=0.005
        def residual(angles):
            data.qpos[:] = q
            data.qpos[adr] = angles
            mujoco.mj_kinematics(m, data)
            rot = Rotation.from_matrix(data.xmat[foot].reshape(3, 3)).as_rotvec()
            return np.r_[data.xpos[foot, [0, 2]] - [target_x, target_z], rot[1]]
        lo, hi = m.jnt_range[js].T
        fits = [least_squares(residual, np.clip(seed, lo+1e-8, hi-1e-8), bounds=(lo, hi),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12) for seed in
                (q[adr], [-0.7, 1.4, -0.7], [-1.1, 1.8, -0.7])]
        fit = min(fits, key=lambda f: np.linalg.norm(f.fun))
        err = residual(fit.x)
        q[adr] = fit.x
        bottoms = data.geom_xpos[sole, 2] - m.geom_size[sole, 0]
        flat_error = float(np.linalg.norm(Rotation.from_matrix(data.xmat[foot].reshape(3, 3)).as_rotvec()))
        ok = bool(np.max(np.abs(bottoms-floor_z)) < 1e-6 and flat_error < 1e-6 and abs(err[0]) < 1e-6)
        info['feasible'] &= ok
        info['sides'][side] = {'joint_rad': dict(zip(names, fit.x.tolist())), 'feasible': ok,
                              'sole_z_m': bottoms.tolist(), 'flat_error_rad': flat_error,
                              'foot_x_error_m': float(err[0]), 'residual': err.tolist()}
    return q, info


def reference_sole_height(g: G1Wuji) -> float:
    m, d = g.model, g.data
    ids = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.endswith('ankle_roll_link') and m.geom_contype[i]]
    return float(np.mean(d.geom_xpos[ids, 2] - m.geom_size[ids, 0]))


def clearance_arrays(g: G1Wuji, rt: dict, rows: np.ndarray, arrays: dict) -> dict:
    check = Clearance(g)
    d = mujoco.MjData(g.model)
    for s in SIDES:
        arrays[s + '_clearance_m'] = np.full(len(rows), np.nan)
        arrays[s + '_closest_geom_pair'] = np.full((len(rows), 2), -1, int)
    for t, row in enumerate(rows):
        d.qpos[:] = g.q_home
        for s in SIDES:
            d.qpos[g.arm_qpos_adr[s]] = arrays[s + '_q_arm_hold'][t]
            fingers = rt[s + '_q_wuji'][row] if rt[s + '_valid'][row] else rt[s + '_q_wuji_cmd'][row]
            g.hands[s].set_q_sdk(d, fingers)
        mujoco.mj_kinematics(g.model, d)
        for s in SIDES:
            if rt[s + '_valid'][row]:
                arrays[s + '_clearance_m'][t], arrays[s + '_closest_geom_pair'][t] = check.minimum(d, s)
    return arrays


def summarize_session(rt: dict, rows: np.ndarray, arrays: dict) -> dict:
    out = {}
    for s in SIDES:
        valid = rt[s + '_valid'][rows]
        pos, rot = arrays[s + '_pos_err_m'], arrays[s + '_rot_err_rad']
        distance = arrays[s + '_clearance_m']
        if not (np.isfinite(pos[valid]).all() and np.isfinite(rot[valid]).all() and np.isfinite(distance[valid]).all()):
            raise ValueError('non-finite result on a usable frame')
        fail = valid & reach_failure(pos, rot)
        coll = valid & (distance < 0)
        arrays[s + '_reach_failure'] = fail
        arrays[s + '_body_collision'] = coll
        arrays[s + '_valid'] = valid
        out[s] = {'usable_frames': int(valid.sum()), 'reach_failures': int(fail.sum()),
                  'body_collision_frames': int(coll.sum()),
                  'reach_failure_frame_index': rt['frame_index'][rows[fail]].tolist(),
                  'body_collision_frame_index': rt['frame_index'][rows[coll]].tolist(),
                  'p95_position_error_m': float(np.percentile(pos[valid],95)) if valid.any() else None,
                  'max_position_error_m': float(np.max(pos[valid])) if valid.any() else None,
                  'max_rotation_error_deg': float(np.degrees(np.max(rot[valid]))) if valid.any() else None,
                  'minimum_clearance_m': float(np.min(distance[valid])) if valid.any() else None}
    return out


def evaluate(offset: tuple, sessions: dict, stride: int | None = None, stored_baseline: bool = False) -> tuple[dict, dict]:
    """Solve chronologically at each offset; full evaluation uses all original rows."""
    g = G1Wuji(pelvis_offset=offset)
    reference = G1Wuji()
    T_ref = reference.T_world_from_torso() @ frames.T_torso_from_cam()
    ik = ArmIK(g, T_world_from_cam=T_ref)
    summary = {'offset_m': list(map(float, offset)), 'sessions': {}, 'stride': stride,
               'stored_baseline': stored_baseline,
               'head_camera_displacement_m': (g.T_world_from_torso()[:3, 3] - reference.T_world_from_torso()[:3, 3]).tolist()}
    outputs, all_pos = {}, []
    for session, rt in sessions.items():
        rows = np.arange(len(rt['frame_index'])) if stride is None else sample_rows(rt, stride)
        arrays = {'rows': rows, 'frame_index': rt['frame_index'][rows], 'timestamps_s': rt['timestamps_s'][rows]}
        if stored_baseline:
            if np.any(offset):
                raise ValueError('stored baseline requires zero offset')
            for s in SIDES:
                held = np.empty((len(rows),7)); prev = g.q_home[g.arm_qpos_adr[s]].copy()
                for i,row in enumerate(rows):
                    if rt[s+'_valid'][row]:
                        prev = rt[s+'_q_g1_arm'][row].copy()
                    held[i] = prev
                arrays.update({s+'_q_arm': rt[s+'_q_g1_arm'][rows], s+'_q_arm_hold': held,
                               s+'_pos_err_m':rt[s+'_ik_pos_err_m'][rows], s+'_rot_err_rad':rt[s+'_ik_rot_err_rad'][rows]})
        else:
            res = ik.run({s: rt[s+'_T_cam_from_palm'][rows] for s in SIDES}, {s: rt[s+'_valid'][rows] for s in SIDES})
            for s in SIDES:
                for name in ('q_arm','q_arm_hold','pos_err_m','rot_err_rad','n_iters','start_used'):
                    arrays[s+'_'+name] = getattr(res,name)[s]
        clearance_arrays(g, rt, rows, arrays)
        summary['sessions'][session] = summarize_session(rt, rows, arrays)
        all_pos.extend(arrays[s+'_pos_err_m'][rt[s+'_valid'][rows]] for s in SIDES)
        outputs[session] = arrays
    sides = [v for session in summary['sessions'].values() for v in session.values()]
    summary['reach_failure_hand_frames'] = sum(v['reach_failures'] for v in sides)
    summary['body_collision_hand_frames'] = sum(v['body_collision_frames'] for v in sides)
    summary['usable_hand_frames'] = sum(v['usable_frames'] for v in sides)
    summary['p95_position_error_m'] = float(np.percentile(np.concatenate(all_pos),95))
    return summary, outputs


# W1-A06 (accepted 2026-10-01): the robot placement every B playback, metric and render uses.
PELVIS_OFFSET_M = (0.020, 0.0, -0.315)  # world frame, relative to the G1 home keyframe (pelvis z 1.0 -> 0.685)
LEG_PITCH_RAD = {'hip_pitch': -0.564172666, 'knee': 1.177630700, 'ankle_pitch': -0.613458034}  # both legs; feet flat at z=0


def placed_g1(width: int = 640, height: int = 480, placed: bool = True) -> G1Wuji:
    """G1 + Wuji at the W1-A06 placement: pelvis shifted, knees bent so both feet stand flat on the z=0 floor,
    waist upright. ``placed=False`` gives the pre-B5 robot (home keyframe, soles 0.22 m above the floor)."""
    if not placed:
        return G1Wuji(width, height)
    g = G1Wuji(width, height, pelvis_offset=PELVIS_OFFSET_M)
    for s in SIDES:
        for joint, value in LEG_PITCH_RAD.items():
            g.q_home[g.model.joint(f'{s}_{joint}_joint').qposadr[0]] = value
    g.data.qpos[:] = g.q_home
    mujoco.mj_kinematics(g.model, g.data)
    return g


def reference_T_world_from_cam() -> np.ndarray:
    """The palm targets' camera: the head camera of the unshifted home pose (AS-01). Targets stay here (W1-B16)."""
    return G1Wuji().T_world_from_torso() @ frames.T_torso_from_cam()


def move_camera_to_reference(g: G1Wuji, camera: str) -> None:
    """Render-only: move ``camera`` (a torso camera) back to the reference pose, so it sees what the human's camera
    saw. Valid because the torso only translates by the pelvis offset (waist fixed, orientation unchanged)."""
    m = g.model
    c = m.camera(camera).id
    if m.cam_bodyid[c] != g.torso_body_id:
        raise ValueError(f'{camera} is not on the torso')
    shift = reference_T_world_from_cam()[:3, 3] - g.T_world_from_torso()[:3, :3] @ m.cam_pos[c] - g.T_world_from_torso()[:3, 3]
    m.cam_pos[c] += g.T_world_from_torso()[:3, :3].T @ shift
