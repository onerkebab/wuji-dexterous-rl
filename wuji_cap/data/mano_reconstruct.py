"""smplx reconstruction plus an independent NumPy reference implementation."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch


# MANO's native joints: wrist, index, middle, pinky, ring, thumb (3 each).
# These tips are the installed smplx MANO vertex definitions, checked in A1 overlays.
FINGERS = {
    "thumb": ((0, 13, 14, 15), 744),
    "index": ((0, 1, 2, 3), 320),
    "middle": ((0, 4, 5, 6), 443),
    "ring": ((0, 10, 11, 12), 554),
    "pinky": ((0, 7, 8, 9), 671),
}


@dataclass
class Reconstruction:
    vertices: np.ndarray
    joints: np.ndarray
    valid: np.ndarray


def make_layer(model_dir, side, flip_left_shapedirs=False):
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side: {side}")
    layer = smplx.MANO(str(Path(model_dir)), is_rhand=side == "right",
                       use_pca=False, flat_hand_mean=True)
    if flip_left_shapedirs:
        if side != "left":
            raise ValueError("Shape sign diagnostic is left-hand only")
        with torch.no_grad():
            layer.shapedirs[:, 0, :] *= -1
    return layer


def reconstruct(hand, layer, batch_size=64):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = len(hand.valid)
    vertices = np.full((count, 778, 3), np.nan, dtype=np.float32)
    joints = np.full((count, 16, 3), np.nan, dtype=np.float32)
    indices = np.flatnonzero(hand.valid)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            result = layer(
                betas=torch.tensor(np.broadcast_to(hand.betas, (len(rows), 10)), dtype=torch.float32),
                hand_pose=torch.tensor(hand.hand_pose[rows], dtype=torch.float32),
                global_orient=torch.tensor(hand.global_orient[rows], dtype=torch.float32),
                transl=torch.tensor(hand.transl[rows], dtype=torch.float32),
            )
            vertices[rows] = result.vertices.numpy()
            joints[rows] = result.joints.numpy()
    valid = hand.valid & np.isfinite(vertices).all(axis=(1, 2)) & np.isfinite(joints).all(axis=(1, 2))
    vertices[~valid] = np.nan
    joints[~valid] = np.nan
    return Reconstruction(vertices, joints, valid)


def numpy_reference(model, betas, hand_pose, global_orient, transl):
    """Single-frame MANO, independent of torch/smplx skinning code.

    Shape blend, pose blend, forward kinematics, then linear blend skinning.
    Root rotation is around the shaped root joint, as in standard MANO;
    transl is an additive displacement, not the camera-frame wrist position.
    """
    shaped = model["v_template"] + np.einsum("vci,i->vc", model["shapedirs"], betas)
    rest_joints = model["J_regressor"] @ shaped
    pose = np.concatenate((global_orient, hand_pose)).astype(np.float64).reshape(16, 3)
    rotations = np.stack([cv2.Rodrigues(vector)[0] for vector in pose])
    feature = (rotations[1:] - np.eye(3)).reshape(135)
    posed = shaped + np.einsum("vcp,p->vc", model["posedirs"], feature)
    tree = model["kintree_table"]
    id_to_col = {int(joint): i for i, joint in enumerate(tree[1])}
    transforms = np.tile(np.eye(4), (16, 1, 1))
    for joint in range(16):
        transforms[joint, :3, :3] = rotations[joint]
        if joint == 0:
            transforms[joint, :3, 3] = rest_joints[joint]
        else:
            parent = id_to_col[int(tree[0, joint])]
            transforms[joint, :3, 3] = rest_joints[joint] - rest_joints[parent]
            transforms[joint] = transforms[parent] @ transforms[joint]
    joints = transforms[:, :3, 3].copy() + transl
    transforms[:, :3, 3] -= np.einsum("jcd,jd->jc", transforms[:, :3, :3], rest_joints)
    blended = np.einsum("vj,jab->vab", model["weights"], transforms)
    vertices = np.einsum("vab,vb->va", blended, np.column_stack((posed, np.ones(778))))[:, :3]
    return vertices + transl, joints


def finger_points(reconstruction, frame, name):
    chain, tip = FINGERS[name]
    return np.vstack((reconstruction.joints[frame, list(chain)], reconstruction.vertices[frame, tip]))
