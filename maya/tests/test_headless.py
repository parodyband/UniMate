"""Headless end-to-end check of the Maya tool against a running server.

    mayapy maya/tests/test_headless.py            (from the repository root)

Needs ``python -m unimate.serve`` running, the test rigs from
``rig_preprocess`` (``outputs/rig/eagle_fbx/Eagle.fbx``,
``outputs/rig/shark_test/shark.fbx``) and the README's example samples
(``outputs/samples/custom``: a generated Eagle motion and its Blender-animated
FBX). Exits non-zero on the first failed check.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'maya', 'scripts'))

import maya.standalone  # noqa: E402

maya.standalone.initialize(name='python')

from maya import cmds  # noqa: E402
from maya.api import OpenMaya as om  # noqa: E402

from unimate_maya import client, proxy, skeleton, timeline  # noqa: E402

EAGLE_FBX = os.path.join(ROOT, 'outputs/rig/eagle_fbx/Eagle.fbx')
SHARK_FBX = os.path.join(ROOT, 'outputs/rig/shark_test/shark.fbx')
MOTION = os.path.join(ROOT, 'outputs/samples/custom/motions/'
                            'Eagle-an_object_flaps_its_wings-rep_0-0.npy')
BLENDER_ANIM = os.path.join(ROOT, 'outputs/samples/custom/animated/'
                                  'Eagle-an_object_flaps_its_wings-rep_0-0.fbx')
FAILED = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)


def world(node, frame=None):
    """World matrix (column vectors, cm) of *node* at *frame* (default: now).
    Evaluated with getAttr -time: moving the time slider would add undo entries."""
    if frame is None:
        frame = cmds.currentTime(query=True)
    m = cmds.getAttr(f'{node}.worldMatrix[0]', time=frame)
    return np.array(m).reshape(4, 4).T


def import_fbx(path, namespace):
    # the FBX importer ignores -namespace; it imports into the current one
    cmds.namespace(add=namespace)
    cmds.namespace(set=f':{namespace}')
    try:
        cmds.file(path, i=True, type='FBX', ignoreVersion=True)
    finally:
        cmds.namespace(set=':')
    joints = cmds.ls(f'{namespace}:*', type='joint', long=True)
    return skeleton.find_root(joints[0])


def proxy_positions(group, frames):
    joints = proxy.proxy_joints(group)
    names = proxy.rig_info(group)['joints']
    out = [[world(joints[n], f)[:3, 3] for n in names] for f in frames]
    return np.array(out), names


def main():
    cmds.loadPlugin('fbxmaya', quiet=True)
    cmds.undoInfo(state=True, infinity=True)
    cmds.currentUnit(time='ntsc')                       # 30 fps: clips need no resampling
    api = client.Client()
    health = api.health()
    print(f"server: ready={health['ready']} rigs={health['rigs']}")
    if not health['ready']:
        raise SystemExit("server not ready")

    print("eagle: read skeleton, build proxy")
    root = import_fbx(EAGLE_FBX, 'eagle')
    sk, nodes, notes = skeleton.read_skeleton(root, name='EagleMaya')
    print(f"  {len(sk['joints'])} joints, up={sk['up_axis']}; {notes}")
    analysis = api.analyze(sk)
    check(analysis['n_included'] == 51, f"analyze includes all 51 joints "
                                        f"({analysis['n_included']})")
    record = api.build(sk)
    group = proxy.build_proxy(record, nodes, api.url)
    joints = proxy.proxy_joints(group)
    check(len(joints) == 51, f"proxy has 51 joints ({len(joints)})")
    rest_err = max(np.abs(world(joints[n])[:3, 3] - world(nodes[n])[:3, 3]).max()
                   for n in joints)
    check(rest_err < 1e-3, f"proxy rest matches rig bind positions ({rest_err:.1e} cm)")

    print("eagle: key a known motion in place, compare with the expected positions")
    clip = api._call('/clips/from_features', {'rig': record['name'], 'path': MOTION,
                                             'debug': True})
    t0 = time.perf_counter()
    first, last = timeline.append_clip(group, clip, placement='in_place', blend_frames=0,
                                       start=1)
    print(f"  keyed frames {first:g}-{last:g} in {time.perf_counter() - t0:.2f}s")
    frames = list(range(int(first), int(last) + 1))
    got, names = proxy_positions(group, frames)
    idx = [clip['joints'].index(n) for n in names]
    expected = np.asarray(clip['world_positions'])[:, idx]
    extent = np.ptp(expected[0], axis=0).max()
    err = np.abs(got - expected).max() / extent
    check(err < 1e-4, f"keyed proxy matches the clip's joint positions "
                      f"({err:.1e} of the rig's size)")

    print("eagle: append a second clip, continue + blend, undo / redo")
    end_root = world(joints[names[0]], last)[:3, 3]
    clip2 = api.generate(record['name'], 'An object walks forward.', seed=11)['clips'][0]
    first2, last2 = timeline.append_clip(group, clip2, placement='continue', blend_frames=6)
    check(first2 == last + 1, f"second clip starts right after the first ({first2:g})")
    start_root = world(joints[names[0]], first2)[:3, 3]
    jump = np.abs((start_root - end_root)[[0, 2]]).max() / extent
    check(jump < 0.01, f"root continues horizontally across the seam ({jump:.1e})")
    n_clips = len(proxy.clips(group))
    cmds.undo()
    check(timeline.last_key(group) == last, "undo removes exactly the appended clip")
    check(len(proxy.clips(group)) == n_clips - 1, "undo also drops it from the clip list")
    cmds.redo()
    check(timeline.last_key(group) == last2, "redo restores it")

    print("eagle: link the rig, then bake")
    made = proxy.link(group)
    check(len(made) == 51, f"linked 51 rig joints ({len(made)})")
    worst = 0.0
    for f in (first, (first + last2) // 2, last2):
        for n in names:
            a, b = world(joints[n], f)[:3, :3], world(nodes[n], f)[:3, :3]
            a = a / np.linalg.norm(a, axis=0)
            b = b / np.linalg.norm(b, axis=0)
            worst = max(worst, np.abs(a - b).max())
    check(worst < 1e-4, f"linked rig joints follow the proxy's world rotations ({worst:.1e})")
    baked = proxy.bake(group)
    check(not proxy.links(group) and not cmds.ls(type='constraint'),
          "bake removes the link constraints")
    rig_keys = cmds.keyframe(baked, query=True, timeChange=True) or []
    check(len(rig_keys) > 0 and max(rig_keys) == last2, "the rig carries the keys after bake")
    worst = 0.0
    for f in (first, last2):
        for n in names:
            a, b = world(joints[n], f)[:3, :3], world(nodes[n], f)[:3, :3]
            worst = max(worst, np.abs(a / np.linalg.norm(a, axis=0)
                                      - b / np.linalg.norm(b, axis=0)).max())
    check(worst < 1e-3, f"baked rig matches the proxy ({worst:.1e})")

    print("shark: a moved, turned, scaled rig with a generated clip")
    cmds.file(new=True, force=True)
    cmds.currentUnit(time='film')                       # 24 fps: clips are resampled
    import_fbx(SHARK_FBX, 'shark')
    top = cmds.group(cmds.ls('shark:*', assemblies=True), name='shark_offset')
    cmds.xform(top, rotation=(0, 90, 0), scale=(37, 37, 37), translation=(120, 5, -40))
    root = skeleton.find_root(cmds.ls('shark:*', type='joint', long=True)[0])
    sk, nodes, notes = skeleton.read_skeleton(root, name='SharkMaya')
    print(f"  {notes}")
    record = api.build(sk)
    group = proxy.build_proxy(record, nodes, api.url)
    size = np.ptp([p['world_position'] for p in record['proxy']], axis=0).max()
    rest_err = max(np.abs(world(j)[:3, 3] - world(nodes[n])[:3, 3]).max()
                   for n, j in proxy.proxy_joints(group).items()) / size
    check(rest_err < 1e-4, f"proxy of the moved rig sits on it at rest ({rest_err:.1e})")
    gen = api._call('/generate', {'rig': record['name'], 'prompt':
                                  'An object swims and turns around.', 'seed': 5,
                                  'debug': True})
    clip = gen['clips'][0]
    print(f"  generated in {gen['seconds']:.2f}s")
    first, last = timeline.append_clip(group, clip, placement='in_place', blend_frames=0,
                                       start=1)
    check(abs((last - first + 1) - 48) <= 1, f"60 frames at 30 fps become ~48 at 24 fps "
                                              f"({last - first + 1:g})")
    # every 4th frame at 24 fps lands on a source frame (every 5th at 30 fps)
    got, names = proxy_positions(group, [first, first + 4, first + 44])
    idx = [clip['joints'].index(n) for n in names]
    expected = np.asarray(clip['world_positions'])[[0, 5, 55]][:, idx]
    extent = np.ptp(expected[0], axis=0).max()
    err = np.abs(got - expected).max() / extent
    check(err < 1e-4, f"resampled frames on source frames match the clip ({err:.1e})")
    rig_root = world(nodes[names[0]])[:3, 3]
    off = np.abs((got[0, 0] - rig_root)[[0, 2]]).max() / extent
    check(off < 1e-3, f"an in-place clip starts where the moved rig stands ({off:.1e})")

    print('PASSED' if not FAILED else f'FAILED: {len(FAILED)} check(s)')
    return 1 if FAILED else 0


if __name__ == '__main__':
    code = main()
    maya.standalone.uninitialize()
    os._exit(code)
