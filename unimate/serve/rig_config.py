"""A skeleton sent by a DCC client (joint names, parents, bind-pose world
matrices) -> a UniMate asset directory, without bpy.

This is ``data_process.rig_preprocess`` for a skeleton that is already in
memory: the export stage is replaced by the matrices the client sends, then
the same annotation (``rig_preprocess.annotate``) and stage-4 calls
(``process_tpose`` -> ``build_topology_cond``) build ``cond.npy``. The
similarity from the canonical T-pose to the bind pose is stored beside it in
``rig.json``, so generated motion can be mapped back onto the rig
(:mod:`unimate.serve.retarget`).

Skeleton JSON, as the Maya client sends it::

    {"name": "Wolf", "up_axis": "y",
     "joints": [{"name": "Hips", "parent": null, "skinned": true,
                 "matrix": [16 floats]}, ...]}

``matrix`` is the joint's bind-pose world matrix the way Maya stores it:
row-major with row vectors (translation in the last row), in internal units
(cm). Everything below works in column-vector math, Y-up; a Z-up scene is
turned Y-up on the way in and back on the way out.
"""

import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
from Animation import positions_global, rotations_global

from data_process.joint_annotation.face_select_rule import empty_entry
from data_process.rig_preprocess.annotate import annotate, facing_defined, manual_face
from data_process.rig_preprocess.review import build_annotation
from data_process.utils.motion_features import (
    build_topology_cond,
    process_tpose,
    resolve_joint_names,
)
from data_process.utils.skeleton import similarity_fit
from unimate.utils.rotation_conversions import matrix_to_quaternion_np

# The released UniML3D models train on up to 70 joints (71 slots, one kept
# for the joint-addition augmentation).
MAX_JOINTS = 70
MIN_JOINTS = 4
# Largest bind-pose error of the canonical-to-rig similarity, over the rig's
# extent; canonicalization is itself a similarity, so a real rig fits ~exactly.
FIT_TOL = 5e-3
STATS_DATASET = 'general'
RIG_FORMAT = 'unimate-rig/1'

# Rx(-90 deg): Z-up scene -> Y-up, (x, y, z) -> (x, z, -y).
Z_TO_Y = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])

# Unskinned control / helper joints that rig_preprocess prunes by name
# (data_process/utils/blender_export.py); `_tip` leaves are kept for general.
_CONTROL_RE = re.compile(r'^(MCH-|ORG-|VIS-)|(_ik|\.ik|_pole|\.pole|_target|_ctrl|\.ctrl|'
                         r'_tweak|_helper)$', re.IGNORECASE)

# Joints left out first when a rig is over the joint budget, one group at a
# time, by label: helpers, then face detail, then fingers, then end joints.
_TRIM_GROUPS = (
    ('helper', re.compile(r'\b(Twist|Muscle|Jiggle|Hair|Cloth|Cape|Skirt|Scarf|Strap|Belt|'
                          r'Bag|Prop|Weapon|Shield|Hat|Chain|Rope|Reins|Halter)\b')),
    ('face', re.compile(r'\b(Eye|Eyeball|Eyebrow|Eyelid|Lid|Lip|Mouth|Tongue|Cheek|Chin|'
                        r'Nose|Muzzle|Whisker|Fang)\b')),
    ('finger', re.compile(r'\b(Finger|Thumb|Phalanges|Palm)\b')),
    ('end', re.compile(r'\bEnd$')),
)

_BAD_NAME_RE = re.compile(r'[-/\\\[\]*?\s]+')


class RigError(ValueError):
    """A skeleton or annotation the server cannot build a rig from."""


# ---------------------------------------------------------------------------
# Skeleton parsing
# ---------------------------------------------------------------------------

def asset_name(name):
    """The rig's UniMate object type: *name* with the characters
    ``rig_preprocess`` refuses (``- / \\ [ ] * ?``, whitespace) replaced."""
    clean = _BAD_NAME_RE.sub('_', str(name)).strip('_')
    if not clean:
        raise RigError(f"rig name {name!r} has no usable characters")
    return clean


def _rot_from_rowmajor(m):
    """Rotation (column vectors) of a Maya row-major 4x4 with scale removed."""
    r = np.asarray(m, dtype=np.float64).reshape(4, 4)[:3, :3].T
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0:          # a mirrored joint: keep a proper rotation
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def parse_skeleton(skeleton):
    """``(names, parents, rot, pos, skinned)`` of *skeleton* in Y-up world
    space, ordered so that every parent precedes its children. *rot* is
    ``(J, 3, 3)`` world rotations, *pos* ``(J, 3)`` world positions (cm)."""
    joints = skeleton.get('joints') or []
    if len(joints) < MIN_JOINTS:
        raise RigError(f"skeleton has {len(joints)} joints; at least {MIN_JOINTS} needed")
    names = [str(j['name']) for j in joints]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise RigError(f"joint names must be unique, repeated: {dup[:5]}")
    index = {n: i for i, n in enumerate(names)}
    parent_of = []
    for j in joints:
        p = j.get('parent')
        if p is not None and p not in index:
            raise RigError(f"joint {j['name']!r}: parent {p!r} is not in the skeleton")
        parent_of.append(index[p] if p is not None else -1)

    # parents before children (depth-first from the roots, input order kept)
    children = {i: [] for i in range(len(names))}
    for i, p in enumerate(parent_of):
        if p >= 0:
            children[p].append(i)
    order, stack = [], [i for i, p in enumerate(parent_of) if p < 0][::-1]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i][::-1])
    if len(order) != len(names):
        raise RigError("skeleton has a parent cycle")
    remap = {old: new for new, old in enumerate(order)}

    up = str(skeleton.get('up_axis', 'y')).lower()
    if up not in ('y', 'z'):
        raise RigError(f"up_axis must be 'y' or 'z', got {up!r}")
    to_y = Z_TO_Y if up == 'z' else np.eye(3)
    rot = np.stack([to_y @ _rot_from_rowmajor(joints[i]['matrix']) for i in order])
    pos = np.stack([to_y @ np.asarray(joints[i]['matrix'], np.float64)[12:15] for i in order])
    parents = [remap[parent_of[i]] if parent_of[i] >= 0 else -1 for i in order]
    skinned = [bool(joints[i].get('skinned', True)) for i in order]
    return [names[i] for i in order], parents, rot, pos, skinned


def subset(parents, keep):
    """Parents of the joints *keep* (sorted indices) within the subset: each
    joint hangs from its nearest kept ancestor. Exactly one kept joint may lack
    one (the root)."""
    pos_in = {j: i for i, j in enumerate(keep)}
    out = []
    for j in keep:
        p = parents[j]
        while p >= 0 and p not in pos_in:
            p = parents[p]
        out.append(pos_in[p] if p >= 0 else -1)
    roots = [keep[i] for i, p in enumerate(out) if p < 0]
    if len(roots) != 1:
        raise RigError(f"the included joints must form one hierarchy, found {len(roots)} roots")
    return out


def local_rest(parents, rot, pos):
    """Export-stage rest locals: rotations ``(J, 4)`` wxyz and offsets
    ``(J, 3)`` in the parent's frame (the root's are its world ones)."""
    lrot = np.empty_like(rot)
    lpos = np.empty_like(pos)
    for i, p in enumerate(parents):
        if p < 0:
            lrot[i], lpos[i] = rot[i], pos[i]
        else:
            lrot[i] = rot[p].T @ rot[i]
            lpos[i] = rot[p].T @ (pos[i] - pos[p])
    return matrix_to_quaternion_np(lrot), lpos


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate(name, names, parents, rot, pos, llm_client=None):
    """``(labels, face_entry, notes)`` from ``rig_preprocess.annotate``, which
    reads the rest pose from an export-stage NPZ: write one to a scratch dir."""
    lrot, lpos = local_rest(parents, rot, pos)
    tmp = tempfile.mkdtemp(prefix='unimate_rig_')
    try:
        os.makedirs(os.path.join(tmp, 'motions'))
        np.savez(os.path.join(tmp, 'motions', f'{name}-rest_pose.npz'),
                 names=np.array(names), parents=np.array(parents, np.int32),
                 rest_local_pos=lpos, rest_local_rot=lrot, fps=np.array(30))
        mode = 'llm' if llm_client is not None else 'rule'
        return annotate(mode, 'general', name, names, tmp, llm_client=llm_client,
                        rest_pos=pos)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _default_include(names, labels, skinned, parents):
    """Joints included by default, and why the others are not: control /
    helper joints by name, then, while the rig is over :data:`MAX_JOINTS`,
    whole label groups (:data:`_TRIM_GROUPS`)."""
    has_child = {p for p in parents if p >= 0}
    excluded = {}
    for i, n in enumerate(names):
        if parents[i] >= 0 and not skinned[i] and _CONTROL_RE.search(n.split(':')[-1]):
            excluded[i] = 'control / helper joint'
    for group, pattern in _TRIM_GROUPS:
        if len(names) - len(excluded) <= MAX_JOINTS:
            break
        for i, label in enumerate(labels):
            if i in excluded or parents[i] < 0:
                continue
            if group == 'end' and (i in has_child or skinned[i]):
                continue
            if pattern.search(label):
                excluded[i] = f'over the {MAX_JOINTS}-joint budget ({group})'
    return [i not in excluded for i in range(len(names))], excluded


def analyze(skeleton, llm_client=None):
    """What the client shows before building: every joint with its suggested
    label, flags and default inclusion; the facing pair and its alternatives;
    the joint budget."""
    name = asset_name(skeleton.get('name', 'rig'))
    names, parents, rot, pos, skinned = parse_skeleton(skeleton)
    labels, face, notes = _annotate(name, names, parents, rot, pos, llm_client)
    review = build_annotation(name, names, parents, pos, labels, face,
                              mode='llm' if llm_client is not None else 'rule', notes=notes)
    include, why = _default_include(names, labels, skinned, parents)
    joints = []
    for i, j in enumerate(review['joints']):
        joints.append({'name': j['raw'], 'parent': j['parent'], 'label': j['label'],
                       'source': j['source'], 'check': j['check'],
                       'reasons': j.get('reasons', []), 'skinned': skinned[i],
                       'include': include[i], 'exclude_reason': why.get(i, '')})
    return {'name': name, 'joints': joints, 'face_pair': review['face_pair'],
            'face_pair_alternatives': review['face_pair_alternatives'],
            'max_joints': MAX_JOINTS, 'n_included': int(sum(include)), 'notes': notes}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _face_entry(face_pair, names, labels, pos):
    """A ``face_joint_names.json`` entry for the client's facing pair (raw
    names, right or head first), or the empty pair."""
    r, l = (face_pair or {}).get('right'), (face_pair or {}).get('left')
    if not (r and l):
        return empty_entry()
    for j in (r, l):
        if j not in names:
            raise RigError(f"facing joint {j!r} is not an included joint")
    entry = manual_face(names, labels, r, l, bool(face_pair.get('body_axis')))
    if not facing_defined(entry, names, pos):
        raise RigError(f"facing pair {r} / {l} has no horizontal separation in the bind "
                       f"pose; choose another pair")
    return entry


def build_cond(name, names, parents, rot, pos, labels, face):
    """``(cond_entry, fit)``: the stage-4 cond of the skeleton (the sequence
    ``motion_features.process_object`` runs, on in-memory data) and the
    similarity ``bind ~= s R canonical + t`` over the cond's joints."""
    lrot, lpos = local_rest(parents, rot, pos)
    tpos = {'names': np.array(names), 'parents': np.array(parents, np.int32),
            'rest_local_pos': lpos, 'rest_local_rot': lrot, 'fps': np.array(30)}
    _, _, clean, _, _ = resolve_joint_names(name, tpos, None, list(labels),
                                            expected_names=list(names))
    (ta, offsets, scale, _, cparents, cnames, _, order,
     face_idxs, body_axis) = process_tpose(tpos, face_joints=face, target_diameter=2.0)
    if face.get('r_hip', {}).get('raw') and face_idxs == [-1, -1]:
        raise RigError("the facing pair did not resolve in the skeleton")
    entry = build_topology_cond(
        name, cparents, offsets, cnames, [clean[i] for i in order],
        positions_global(ta)[0], ta.rotations.qs[0], rotations_global(ta).qs[0],
        max_path_len=5, max_freqs=8, face_joint_idxs=face_idxs, body_axis=body_axis,
        scale_factor=scale, ground_height=None)
    entry['captions'] = {}          # per-clip captions; a rig without clips has none
    bind = pos[np.asarray(order)]
    R, s, t, resid = similarity_fit(entry['tpos_first_frame'], bind)
    if resid > FIT_TOL:
        raise RigError(f"canonical T-pose does not map onto the bind pose (error {resid:.2e} "
                       f"of the rig's extent)")
    return entry, {'R': R, 's': s, 't': t, 'residual': resid, 'order': list(order)}


def build(skeleton, rigs_dir, include=None, labels=None, face_pair=None, llm_client=None):
    """Write the asset directory ``<rigs_dir>/<name>/`` (``cond.npy``,
    ``summary.json``, ``annotation.json``, ``rig.json``) for the included
    joints with the client's labels (``{joint: label}``) and facing pair
    (``{right, left, body_axis}``); any of the three left out comes from
    :func:`analyze`. Returns the rig record (``rig.json``'s content), which
    carries the proxy skeleton to build."""
    name = asset_name(skeleton.get('name', 'rig'))
    names, parents, rot, pos, skinned = parse_skeleton(skeleton)
    up = str(skeleton.get('up_axis', 'y')).lower()
    if include is None or labels is None or face_pair is None:
        suggested = analyze(skeleton, llm_client)
        if include is None:
            include = [j['name'] for j in suggested['joints'] if j['include']]
        if labels is None:
            labels = {j['name']: j['label'] for j in suggested['joints']}
        if face_pair is None:
            face_pair = suggested['face_pair']
            if not {face_pair.get('right'), face_pair.get('left')} <= set(include):
                face_pair = None
    include = set(include)
    unknown = include - set(names)
    if unknown:
        raise RigError(f"included joints not in the skeleton: {sorted(unknown)[:5]}")
    keep = [i for i, n in enumerate(names) if n in include]
    if not MIN_JOINTS <= len(keep) <= MAX_JOINTS:
        raise RigError(f"{len(keep)} joints included; the model takes {MIN_JOINTS} to "
                       f"{MAX_JOINTS}")
    k_parents = subset(parents, keep)
    k_names = [names[i] for i in keep]
    k_rot, k_pos = rot[keep], pos[keep]
    labels = labels or {}
    k_labels = [str(labels.get(n) or 'Bone').strip() or 'Bone' for n in k_names]
    face = _face_entry(face_pair, k_names, k_labels, k_pos)

    entry, fit = build_cond(name, k_names, k_parents, k_rot, k_pos, k_labels, face)
    order = fit['order']
    record = _rig_record(name, up, entry, fit, k_rot[order], k_pos[order])

    out = os.path.join(rigs_dir, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'cond.npy'), {name: entry})
    review = build_annotation(name, k_names, k_parents, k_pos, k_labels, face, mode='file')
    _write_json(os.path.join(out, 'annotation.json'), review)
    _write_json(os.path.join(out, 'summary.json'), OrderedDict([
        ('name', name), ('profile', 'general'), ('stats_dataset', STATS_DATASET),
        ('source', 'unimate.serve'), ('n_joints', len(k_names)),
        ('face_pair', face), ('model_joint_limit', {'max_joints': MAX_JOINTS + 1,
                                                     'fits': True}),
    ]))
    _write_json(os.path.join(out, 'rig.json'), record)
    return record


def _rig_record(name, up, entry, fit, rot, pos):
    """``rig.json``: the cond's joints in cond order with the bind pose (Y-up
    world rotations as wxyz, positions in cm), the canonical-to-bind
    similarity, and the proxy skeleton the client builds (scene axes)."""
    parents = list(entry['parents'])
    to_scene = Z_TO_Y.T if up == 'z' else np.eye(3)
    proxy = []
    for i, n in enumerate(entry['joint_names']):
        p = parents[i]
        if p < 0:
            # proxy root: jointOrient 0, the bind world rotation on rotate
            orient = np.eye(3)
            rotate = to_scene @ rot[i]
            translate = to_scene @ pos[i]
        else:
            orient = rot[p].T @ rot[i]
            rotate = np.eye(3)
            translate = rot[p].T @ (pos[i] - pos[p])
        proxy.append({'name': n, 'parent': entry['joint_names'][p] if p >= 0 else None,
                      'translate': translate.tolist(),
                      'joint_orient': _xyzw(orient), 'rotate': _xyzw(rotate),
                      'world_position': (to_scene @ pos[i]).tolist()})
    R = fit['R']
    return {
        'format': RIG_FORMAT,
        'name': name,
        'up_axis': up,
        'joint_names': list(entry['joint_names']),
        'clean_joint_names': list(entry['clean_joint_names']),
        'parents': parents,
        'bind_rotations': matrix_to_quaternion_np(rot).tolist(),     # wxyz, Y-up
        'bind_positions': pos.tolist(),                              # cm, Y-up
        'similarity': {'R': R.tolist(), 's': fit['s'], 't': fit['t'].tolist(),
                       'residual': fit['residual']},
        # the rig's facing at bind: canonical +Z through the similarity (scene axes)
        'forward': (to_scene @ R @ np.array([0., 0., 1.])).tolist(),
        'proxy': proxy,
    }


def _xyzw(m):
    """A rotation matrix as an ``x, y, z, w`` quaternion (Maya's order)."""
    w, x, y, z = matrix_to_quaternion_np(np.asarray(m)[None])[0]
    return [float(x), float(y), float(z), float(w)]


def _write_json(path, data):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def load_rig(rigs_dir, name):
    """``(record, cond_entry)`` of a built rig."""
    folder = os.path.join(rigs_dir, name)
    with open(os.path.join(folder, 'rig.json')) as f:
        record = json.load(f)
    conds = np.load(os.path.join(folder, 'cond.npy'), allow_pickle=True).item()
    return record, conds[name]


def list_rigs(rigs_dir):
    """Names of the rigs built under *rigs_dir*."""
    if not os.path.isdir(rigs_dir):
        return []
    return sorted(d for d in os.listdir(rigs_dir)
                  if os.path.isfile(os.path.join(rigs_dir, d, 'rig.json'))
                  and os.path.isfile(os.path.join(rigs_dir, d, 'cond.npy')))
