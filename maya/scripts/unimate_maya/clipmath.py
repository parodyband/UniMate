"""Clip math with no Maya dependency (numpy only): resampling to the scene's
frame rate, placing a clip where the previous one ended, and blending the
seam. Quaternions are ``x, y, z, w`` (Maya's order); rotations act on column
vectors.

A clip (``unimate-clip/1``) holds, per frame, each joint's rotate channel as
a quaternion and the root's world position; the root joint's rotate channel
is its world rotation (its parent is an identity group and its jointOrient
is zero).
"""

import numpy as np


def qmul(a, b):
    """Hamilton product of ``x, y, z, w`` quaternions (rotation *b*, then *a*)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def qconj(q):
    q = np.array(q, dtype=np.float64)
    q[..., :3] *= -1
    return q


def qrot(q, v):
    """Rotate vectors *v* ``(..., 3)`` by quaternions *q* ``(..., 4)``."""
    v = np.asarray(v, dtype=np.float64)
    qv = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)
    return qmul(qmul(q, qv), qconj(q))[..., :3]


def qaxis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def slerp(a, b, w):
    """Spherical interpolation from *a* to *b* (``(..., 4)``) by *w* (``(...)``
    or scalar), along the shorter arc."""
    a = np.asarray(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)[..., None]
    dot = (a * b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    near = sin < 1e-6
    safe = np.where(near, 1.0, sin)
    out = np.where(near, (1 - w) * a + w * b,
                   (np.sin((1 - w) * theta) * a + np.sin(w * theta) * b) / safe)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample(rotations, root_position, src_fps, dst_fps):
    """``(rotations, root_position)`` of a clip at *src_fps* resampled to
    *dst_fps*, over the same duration (slerp / lerp between source frames)."""
    rot = np.asarray(rotations, dtype=np.float64)          # (F, J, 4)
    pos = np.asarray(root_position, dtype=np.float64)      # (F, 3)
    n = len(rot)
    if abs(src_fps - dst_fps) < 1e-6 or n < 2:
        return rot.copy(), pos.copy()
    duration = (n - 1) / float(src_fps)
    count = int(np.floor(duration * dst_fps + 1e-6)) + 1
    u = np.arange(count) * (src_fps / float(dst_fps))      # source frame index
    i0 = np.clip(np.floor(u).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    w = u - i0
    out_rot = slerp(rot[i0], rot[i1], np.repeat(w[:, None], rot.shape[1], 1))
    out_pos = pos[i0] * (1 - w)[:, None] + pos[i1] * w[:, None]
    return out_rot, out_pos


def heading(root_rotation, bind_root_rotation, forward, up):
    """Horizontal heading (unit vector, ``(..., 3)``) of root rotations, from
    the rig's *forward* at bind: ``R R_bind^-1 forward`` projected off *up*."""
    rel = qmul(root_rotation, qconj(bind_root_rotation))
    f = qrot(rel, np.broadcast_to(np.asarray(forward, dtype=np.float64), rel.shape[:-1] + (3,)))
    up = np.asarray(up, dtype=np.float64)
    f = f - (f @ up)[..., None] * up
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.where(norm < 1e-9, 1.0, norm)


def continue_from(rotations, root_position, prev_root_rotation, prev_root_position,
                  bind_root_rotation, forward, up):
    """Turn and shift a clip about *up* so its first frame faces and stands
    where the previous clip ended (horizontally; the clip keeps its own
    height). Only the root changes: children's rotate channels are relative
    to their parents."""
    up = np.asarray(up, dtype=np.float64)
    h0 = heading(rotations[0, 0], bind_root_rotation, forward, up)
    h1 = heading(prev_root_rotation, bind_root_rotation, forward, up)
    angle = np.arctan2(np.cross(h0, h1) @ up, h0 @ h1)
    yaw = qaxis_angle(up, angle)
    rot = np.array(rotations, dtype=np.float64)
    pos = np.asarray(root_position, dtype=np.float64)
    rot[:, 0] = qmul(yaw, rot[:, 0])
    pos = qrot(np.broadcast_to(yaw, (len(pos), 4)), pos - pos[0])
    offset = np.asarray(prev_root_position, dtype=np.float64) - np.asarray(root_position[0])
    offset = offset - (offset @ up) * up                      # horizontal only
    pos = pos + np.asarray(root_position[0]) + offset
    return rot, pos


def blend_in(rotations, root_position, prev_rotations, prev_root_position, frames):
    """Ease the first *frames* frames from the previous clip's last pose
    (*prev_rotations* ``(J, 4)``, *prev_root_position*) into the clip."""
    rot = np.array(rotations, dtype=np.float64)
    pos = np.array(root_position, dtype=np.float64)
    k = min(int(frames), len(rot))
    if k <= 0:
        return rot, pos
    for i in range(k):
        x = (i + 1) / float(k + 1)
        w = x * x * (3 - 2 * x)                                 # smoothstep
        rot[i] = slerp(prev_rotations, rot[i], np.full(rot.shape[1], w))
        pos[i] = (1 - w) * np.asarray(prev_root_position) + w * pos[i]
    return rot, pos
