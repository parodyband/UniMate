"""Append clips to a proxy's timeline.

``append_clip`` keys one clip on the proxy after its last key, as one undo
step: resample to the scene's frame rate, place it where the previous clip
ended (``continue``) or leave it where it was generated (``in_place``: at the
rig's bind position, facing the rig's forward), ease the seam, then key every
proxy joint's rotate and the root's translate.
"""

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from unimate_maya import clipmath
from unimate_maya.proxy import (
    cm_to_ui,
    clips,
    euler_to_quat,
    proxy_joints,
    quat_to_euler,
    rad_to_ui,
    rig_info,
    set_clips,
    ui_to_cm,
    ui_to_rad,
    undo_chunk,
)

PLACEMENTS = ('continue', 'in_place')
_UP = {'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


def scene_fps():
    return om.MTime(1.0, om.MTime.kSeconds).asUnits(om.MTime.uiUnit())


def last_key(group):
    nodes = list(proxy_joints(group).values())
    times = cmds.keyframe(nodes, query=True, timeChange=True) if nodes else None
    return max(times) if times else None


def _pose_at(group, frame, names, joints):
    """The proxy's rotate channels as quaternions ``(J, 4)``, its Euler values
    (radians) and the root translate (cm) at *frame*."""
    quats, eulers = [], []
    for name in names:
        node = joints[name]
        e = [ui_to_rad(v) for v in cmds.getAttr(f'{node}.rotate', time=frame)[0]]
        order = cmds.getAttr(f'{node}.rotateOrder')
        eulers.append(e)
        quats.append(euler_to_quat(e, order))
    root = joints[names[0]]
    pos = [ui_to_cm(v) for v in cmds.getAttr(f'{root}.translate', time=frame)[0]]
    return np.array(quats), eulers, np.array(pos)


def prepare(group, clip, placement='continue', blend_frames=6, start=None):
    """``(start, rotations, root_position, prev_euler)`` of *clip* as it will be
    keyed on *group*'s proxy: resampled, placed and blended."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    info = rig_info(group)
    joints = proxy_joints(group)
    names = clip['joints']
    missing = [n for n in names if n not in joints]
    if missing:
        raise ValueError(f"the clip's joints are not on this proxy: {missing[:5]}")
    rot, pos = clipmath.resample(clip['rotations'], clip['root_position'],
                                 float(clip.get('fps', 30)), scene_fps())
    last = last_key(group)
    if start is None:
        start = last + 1 if last is not None else cmds.playbackOptions(query=True, minTime=True)
    prev_euler = None
    if last is not None and start > last:
        prev_rot, prev_euler, prev_pos = _pose_at(group, last, names, joints)
        if placement == 'continue':
            rot, pos = clipmath.continue_from(
                rot, pos, prev_rot[0], prev_pos, info['bind_root_rotation'],
                info['forward'], _UP[info['up_axis']])
        rot, pos = clipmath.blend_in(rot, pos, prev_rot, prev_pos, blend_frames)
    return start, rot, pos, prev_euler


def append_clip(group, clip, placement='continue', blend_frames=6, start=None):
    """Key *clip* on *group*'s proxy from *start* (default: the frame after its
    last key, else the playback start) as one undo step. Returns
    ``(first frame, last frame)``."""
    start, rot, pos, prev_euler = prepare(group, clip, placement, blend_frames, start)
    joints = proxy_joints(group)
    names = clip['joints']
    orders = [cmds.getAttr(f'{joints[n]}.rotateOrder') for n in names]
    prompt = clip.get('meta', {}).get('prompt', 'clip')
    end = start + len(rot) - 1
    with undo_chunk(f'UniMate: append "{prompt}"'):
        near = prev_euler
        for f in range(len(rot)):
            t = start + f
            current = []
            for j, name in enumerate(names):
                e = quat_to_euler(rot[f, j], orders[j], near[j] if near else None)
                current.append(e)
                node = joints[name]
                for axis, value in zip('XYZ', e):
                    cmds.setKeyframe(node, attribute=f'rotate{axis}', time=t,
                                     value=rad_to_ui(value), inTangentType='linear',
                                     outTangentType='linear')
            root = joints[names[0]]
            for axis, value in zip('XYZ', pos[f]):
                cmds.setKeyframe(root, attribute=f'translate{axis}', time=t,
                                 value=cm_to_ui(value), inTangentType='linear',
                                 outTangentType='linear')
            near = current
        if cmds.playbackOptions(query=True, maxTime=True) < end:
            cmds.playbackOptions(maxTime=end)
        if cmds.playbackOptions(query=True, animationEndTime=True) < end:
            cmds.playbackOptions(animationEndTime=end)
        history = clips(group)
        history.append(dict(clip.get('meta', {}), start=start, end=end, placement=placement))
        set_clips(group, history)
    return start, end
