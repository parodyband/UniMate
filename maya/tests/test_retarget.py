"""Server-side checks of the rig set-up and retarget math (no server, no Maya).

    python maya/tests/test_retarget.py        (the env's Python, repository root)

1. The Eagle's cond rebuilt from its joints (``unimate.serve.rig_config``)
   matches ``assets/examples/eagle/cond.npy``, also for a moved, turned,
   scaled copy; the rule labels match the reviewed ones.
2. FK of the proxy keyed with a clip (world = parent * jointOrient * rotate,
   as Maya evaluates joints) reproduces the clip's expected joint positions,
   for Y-up and Z-up scenes.
3. Those expected positions equal Blender's animation of the same motion
   (the README's ``run_animate_motion.sh`` GLB), so the Maya tool and the
   Blender path drive a rig identically.

Needs ``outputs/samples/custom`` from the README's example sampling and
animation. Exits non-zero on a failed check.
"""

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from data_process.rig_preprocess.glb import Glb  # noqa: E402
from unimate.serve import rig_config  # noqa: E402
from unimate.serve.retarget import _quat_mats, clip_from_features  # noqa: E402

EAGLE_GLB = 'assets/examples/eagle/Eagle.glb'
MOTION = 'outputs/samples/custom/motions/Eagle-an_object_flaps_its_wings-rep_0-0.npy'
BLENDER_GLB = 'outputs/samples/custom/animated/Eagle-an_object_flaps_its_wings-rep_0-0.glb'
FAILED = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)


def _quat_xyzw_mat(q):
    x, y, z, w = q
    return _quat_mats(np.array([w, x, y, z]))


def glb_world(glb, frame=None):
    """``{joint name: world 4x4}`` of a GLB's skin joints, at keyframe *frame*
    of its first animation (None: rest)."""
    nodes = glb.nodes
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    tracks = {}
    if frame is not None:
        anim = glb.json['animations'][0]
        for ch in anim['channels']:
            s = anim['samplers'][ch['sampler']]
            tracks[(ch['target']['node'], ch['target']['path'])] = glb.accessor(s['output'])
    cache = {}

    def channel(i, path, default):
        track = tracks.get((i, path))
        return np.array(track[frame] if track is not None else default, float)

    def get(i):
        if i not in cache:
            n = nodes[i]
            t = channel(i, 'translation', n.get('translation', [0, 0, 0]))
            r = channel(i, 'rotation', n.get('rotation', [0, 0, 0, 1]))
            s = channel(i, 'scale', n.get('scale', [1, 1, 1]))
            m = np.eye(4)
            m[:3, :3] = _quat_xyzw_mat(r) * s
            m[:3, 3] = t
            if 'matrix' in n:
                m = np.array(n['matrix'], float).reshape(4, 4).T
            cache[i] = m if i not in parent else get(parent[i]) @ m
        return cache[i]
    return {nodes[j]['name']: get(j) for skin in glb.json['skins'] for j in skin['joints']}


def skeleton_from_glb(path, name, scale=1.0, yaw_deg=0.0, shift=(0, 0, 0), z_up=False):
    """The server's skeleton JSON for a GLB's joints, optionally moved / turned
    / scaled, or authored Z-up (turned +90 deg about X)."""
    glb = Glb(path)
    nodes = glb.nodes
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    ids = [j for skin in glb.json['skins'] for j in skin['joints']]
    a = np.radians(yaw_deg)
    turn = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    if z_up:
        turn = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]]) @ turn
    world = glb.world()
    joints = []
    for j in ids:
        p = parent.get(j)
        while p is not None and p not in ids:
            p = parent.get(p)
        m = world[j].copy()
        m[:3, :3] = turn @ m[:3, :3]
        m[:3, 3] = scale * (turn @ m[:3, 3]) + np.asarray(shift)
        joints.append({'name': nodes[j]['name'],
                       'parent': nodes[p]['name'] if p is not None else None,
                       'matrix': m.T.reshape(-1).tolist(), 'skinned': True})
    return {'name': name, 'up_axis': 'z' if z_up else 'y', 'joints': joints}


def proxy_fk(record, clip):
    """Joint world positions ``(T, J, 3)`` of the proxy keyed with *clip*,
    joints in clip order."""
    by = {p['name']: p for p in record['proxy']}
    idx = {n: i for i, n in enumerate(clip['joints'])}
    rot = _quat_mats(np.asarray(clip['rotations'])[..., [3, 0, 1, 2]])
    W, X = {}, {}
    for p in record['proxy']:                         # parents first
        n, i = p['name'], idx[p['name']]
        jo = _quat_xyzw_mat(p['joint_orient'])
        if p['parent'] is None:
            W[n] = jo[None] @ rot[:, i]
            X[n] = np.asarray(clip['root_position'])
        else:
            W[n] = W[p['parent']] @ jo[None] @ rot[:, i]
            X[n] = X[p['parent']] + np.einsum('tab,b->ta', W[p['parent']],
                                              np.asarray(by[n]['translate']))
    return np.stack([X[n] for n in clip['joints']], 1)


def main():
    ref = np.load('assets/examples/eagle/cond.npy', allow_pickle=True).item()['Eagle']
    ann = json.load(open('assets/examples/eagle/annotation.json'))
    labels = {j['raw']: j['label'] for j in ann['joints']}
    face = {'right': ann['face_pair']['right'], 'left': ann['face_pair']['left'],
            'body_axis': ann['face_pair']['body_axis']}
    feats = np.load(MOTION)

    print("1. cond parity with the shipped Eagle")
    analysis = rig_config.analyze(skeleton_from_glb(EAGLE_GLB, 'Eagle'))
    match = sum(j['label'] == labels[j['name']] for j in analysis['joints'])
    check(match == len(labels), f"rule labels match the reviewed ones ({match}/{len(labels)})")
    check((analysis['face_pair']['right'], analysis['face_pair']['left'])
          == (face['right'], face['left']), "rule facing pair matches")
    cases = [(1.0, 0, (0, 0, 0), False), (37.0, 90, (120, 5, -40), False),
             (37.0, 90, (120, 5, -40), True)]
    records = {}
    for case in cases:
        scale, yaw, shift, z_up = case
        out = tempfile.mkdtemp()
        sk = skeleton_from_glb(EAGLE_GLB, 'Eagle', scale, yaw, shift, z_up)
        record = rig_config.build(sk, out, include=[j['name'] for j in sk['joints']],
                                  labels=labels, face_pair=face)
        _, cond = rig_config.load_rig(out, 'Eagle')
        records[case] = (record, cond)
        tag = f"x{scale:g} yaw {yaw} {'Z' if z_up else 'Y'}-up"
        check(set(cond) == set(ref) and cond['joint_names'] == ref['joint_names']
              and cond['clean_joint_names'] == ref['clean_joint_names'],
              f"{tag}: same keys, joint order and labels")
        pos = max(np.abs(np.asarray(cond[k]) - np.asarray(ref[k])).max()
                  for k in ('tpos_first_frame', 'tpos_offsets', 'offsets'))
        rot = max(np.minimum(np.abs(np.asarray(cond[k]) - ref[k]).max(-1),
                             np.abs(np.asarray(cond[k]) + ref[k]).max(-1)).max()
                  for k in ('tpos_local_rotations', 'tpos_global_rotations'))
        topo = all(np.array_equal(cond[k], ref[k]) for k in
                   ('joint_relations', 'joint_graph_dists', 'joint_depths', 'edge_indexs'))
        check(pos < 1e-5 and rot < 1e-5 and topo,
              f"{tag}: T-pose within 1e-5 (pos {pos:.1e}, rot {rot:.1e}), same topology")
        check(abs(record['similarity']['s'] - scale) < 1e-6 * scale,
              f"{tag}: similarity scale {record['similarity']['s']:.4f}")

    print("2. FK of the keyed proxy reproduces the clip")
    for case, (record, cond) in records.items():
        clip = clip_from_features(feats, record, cond, debug=True)
        got = proxy_fk(record, clip)
        expected = np.asarray(clip['world_positions'])
        err = np.abs(got - expected).max() / np.ptp(expected[0], axis=0).max()
        check(err < 1e-5, f"x{case[0]:g} {'Z' if case[3] else 'Y'}-up: {err:.1e} of the size")

    print("3. the clip equals Blender's animation of the same motion")
    record, cond = records[cases[0]]
    clip = clip_from_features(feats, record, cond, debug=True)
    expected = np.asarray(clip['world_positions'])
    blender = Glb(BLENDER_GLB)
    names = clip['joints']
    worst = 0.0
    for f in range(len(expected)):
        world = glb_world(blender, f)
        got = np.array([world[n][:3, 3] for n in names])
        worst = max(worst, np.abs(got - expected[f]).max())
    worst /= np.ptp(expected[0], axis=0).max()
    check(worst < 1e-4, f"every joint, every frame: {worst:.1e} of the size")

    print('PASSED' if not FAILED else f'FAILED: {len(FAILED)} check(s)')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
