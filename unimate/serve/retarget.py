"""Generated UniMate motion -> a clip for a rig's proxy skeleton.

The model's features decode (``recover_unimate_anim_from_rot``) to local
rotations ``N`` of a skeleton whose rest has identity rotations and the
canonical T-pose offsets, so the global rotations ``D`` are each joint's
world-space change from the canonical T-pose. The rig's bind pose is that
T-pose through the similarity ``bind = s R canonical + t``
(:mod:`unimate.serve.rig_config`), with bind world rotations ``T``. On the
proxy (child jointOrient = the bind rotation relative to its parent, root
jointOrient 0) this gives, column vectors:

- child rotate channel: ``Q^-1 N Q`` with ``Q = R^T T``, the formula the
  Blender path keys (``data_process/utils/blender_rig.py``);
- root rotate channel = root world rotation: ``R D_0 R^T T_0``;
- root position: ``s R r_pos + t``.

The clip (``unimate-clip/1``) holds no UniMate specifics: per-frame rotate
channels of named joints as ``x, y, z, w`` quaternions, and the root's world
position in the scene's axes and internal units (cm), at ``fps``.
"""

import numpy as np
from Animation import positions_global, rotations_global

from unimate.serve.rig_config import Z_TO_Y
from unimate.utils.motion_utils import recover_unimate_anim_from_rot
from unimate.utils.rotation_conversions import matrix_to_quaternion_np

CLIP_FORMAT = 'unimate-clip/1'
FPS = 30


def _quat_mats(wxyz):
    """``(..., 4)`` w, x, y, z quaternions -> ``(..., 3, 3)`` rotation matrices."""
    q = np.asarray(wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _continuous_xyzw(mats):
    """``(T, J, 3, 3)`` -> ``(T, J, 4)`` x, y, z, w with no sign flips over time."""
    wxyz = matrix_to_quaternion_np(mats)
    for t in range(1, len(wxyz)):
        flip = (wxyz[t] * wxyz[t - 1]).sum(-1) < 0
        wxyz[t, flip] *= -1
    return wxyz[..., [1, 2, 3, 0]]


def clip_from_features(feats, record, cond, meta=None, frames=None, debug=False):
    """The clip of one generated motion.

    *feats* are denormalized model features ``(T, J, 12)`` in cond joint order,
    *record* the rig's ``rig.json``, *cond* its cond entry; *frames* trims the
    clip to its first frames; *debug* adds ``world_positions`` ``(T, J, 3)``,
    every joint's expected world position, for checking a keyed proxy.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if frames:
        feats = feats[:max(2, int(frames))]
    parents = np.asarray(cond['parents'])
    anim = recover_unimate_anim_from_rot(feats, parents,
                                         np.asarray(cond['tpos_offsets'], np.float64))
    N = anim.rotations.transforms()                       # (T, J, 3, 3)
    D = rotations_global(anim).transforms()
    r_pos = np.asarray(anim.positions[:, 0])              # (T, 3) canonical

    sim = record['similarity']
    R, s, t = np.asarray(sim['R']), float(sim['s']), np.asarray(sim['t'])
    T_bind = _quat_mats(record['bind_rotations'])         # (J, 3, 3) Y-up
    to_scene = Z_TO_Y.T if record.get('up_axis') == 'z' else np.eye(3)

    Q = np.einsum('ab,jbc->jac', R.T, T_bind)
    channels = np.einsum('jba,tjbc,jcd->tjad', Q, N, Q)   # Q^T N Q
    channels[:, 0] = np.einsum('ab,tbc,dc,de->tae', to_scene @ R, D[:, 0], R, T_bind[0])
    root_position = (s * r_pos @ R.T + t) @ to_scene.T

    clip = {
        'format': CLIP_FORMAT,
        'fps': FPS,
        'frame_count': int(len(feats)),
        'joints': list(record['joint_names']),
        'rotations': np.round(_continuous_xyzw(channels), 7).tolist(),
        'root_position': np.round(root_position, 5).tolist(),
        'meta': dict(meta or {}),
    }
    if debug:
        world = (s * positions_global(anim) @ R.T + t) @ to_scene.T
        clip['world_positions'] = world.tolist()
    return clip
