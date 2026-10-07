"""The proxy skeleton: UniMate's clean copy of a rig, which clips are keyed on
and which drives the rig.

The proxy lives under a group ``UNIMATE_<rig>`` (identity transform) and has
one joint per joint the rig was built with on the server, at the rig joint's
bind-pose world transform: the root with jointOrient 0 and its bind world
rotation on rotate, every other joint with jointOrient = its bind rotation
relative to its proxy parent and rotate 0. The group stores the set-up as
JSON string attributes, so a saved scene keeps it.

Linking drives rig nodes from the proxy with constraints. A rig joint's own
proxy joint has its bind world transform, so the default links (each rig
joint from its proxy joint) need no offsets and hold in any pose; a custom
target (a control) is linked with maintain-offset while the proxy sits at
rest, so the rig must then be at rest too. Baking moves the motion onto the
targets and removes the links.
"""

import contextlib
import json
import re

from maya import cmds
from maya.api import OpenMaya as om

GROUP_PREFIX = 'UNIMATE_'
RIG_ATTR = 'unimateRig'
CLIPS_ATTR = 'unimateClips'
LINKS_ATTR = 'unimateLinks'
JOINT_ATTR = 'unimateJoint'
MODES = ('orient', 'parent', 'point', 'none')


# ---------------------------------------------------------------------------
# Units and rotations
# ---------------------------------------------------------------------------

def cm_to_ui(value):
    return om.MDistance(value, om.MDistance.kCentimeters).asUnits(om.MDistance.uiUnit())


def ui_to_cm(value):
    return om.MDistance(value, om.MDistance.uiUnit()).asUnits(om.MDistance.kCentimeters)


def rad_to_ui(value):
    return om.MAngle(value, om.MAngle.kRadians).asUnits(om.MAngle.uiUnit())


def ui_to_rad(value):
    return om.MAngle(value, om.MAngle.uiUnit()).asUnits(om.MAngle.kRadians)


def quat_to_euler(xyzw, order=0, near=None):
    """``x, y, z, w`` -> Euler radians in rotate order *order* (Maya's enum),
    the solution closest to *near* (radians) when given."""
    e = om.MQuaternion(*xyzw).asEulerRotation().reorder(order)
    if near is not None:
        e.setToClosestSolution(om.MEulerRotation(near[0], near[1], near[2], order))
    return (e.x, e.y, e.z)


def euler_to_quat(xyz, order=0):
    q = om.MEulerRotation(xyz[0], xyz[1], xyz[2], order).asQuaternion()
    return (q.x, q.y, q.z, q.w)


@contextlib.contextmanager
def undo_chunk(name):
    cmds.undoInfo(openChunk=True, chunkName=name)
    try:
        yield
    finally:
        cmds.undoInfo(closeChunk=True)


# ---------------------------------------------------------------------------
# Stored set-up
# ---------------------------------------------------------------------------

def _get_json(node, attr, default):
    if not cmds.attributeQuery(attr, node=node, exists=True):
        return default
    raw = cmds.getAttr(f'{node}.{attr}')
    return json.loads(raw) if raw else default


def _set_json(node, attr, value):
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, dataType='string')
    cmds.setAttr(f'{node}.{attr}', json.dumps(value), type='string')


def rig_info(group):
    """The set-up stored on a proxy group (see :func:`build_proxy`)."""
    info = _get_json(group, RIG_ATTR, None)
    if info is None:
        raise ValueError(f"{group} is not a UniMate proxy")
    return info


def clips(group):
    return _get_json(group, CLIPS_ATTR, [])


def set_clips(group, value):
    _set_json(group, CLIPS_ATTR, value)


def links(group):
    return _get_json(group, LINKS_ATTR, [])


def proxies():
    """Proxy groups in the scene."""
    return [n for n in cmds.ls(f'{GROUP_PREFIX}*', type='transform', long=True) or []
            if cmds.attributeQuery(RIG_ATTR, node=n, exists=True)]


def proxy_joints(group):
    """``{server joint name: proxy joint long name}``."""
    out = {}
    for j in cmds.listRelatives(group, allDescendents=True, type='joint', fullPath=True) or []:
        if cmds.attributeQuery(JOINT_ATTR, node=j, exists=True):
            out[cmds.getAttr(f'{j}.{JOINT_ATTR}')] = j
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _safe(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


def build_proxy(record, nodes, server_url=None, replace=True):
    """Create the proxy group for a rig *record* (the server's ``rig.json``);
    *nodes* maps joint names to the rig's DAG paths. Returns the group."""
    name = record['name']
    group_name = f'{GROUP_PREFIX}{_safe(name)}'
    with undo_chunk(f'UniMate: build proxy {name}'):
        for old in cmds.ls(group_name, long=True) or []:
            if not replace:
                raise ValueError(f"{group_name} already exists")
            unlink(old)
            cmds.delete(old)
        group = cmds.createNode('transform', name=group_name)
        group = cmds.ls(group, long=True)[0]
        positions = [p['world_position'] for p in record['proxy']]
        extent = max(max(c) - min(c) for c in zip(*positions)) or 1.0
        created = {}
        for p in record['proxy']:
            parent = created[p['parent']] if p['parent'] else group
            j = cmds.createNode('joint', name=f'{_safe(name)}_um_{_safe(p["name"])}',
                                parent=parent)
            j = cmds.ls(j, long=True)[0]
            cmds.setAttr(f'{j}.translate', *[cm_to_ui(v) for v in p['translate']])
            cmds.setAttr(f'{j}.jointOrient', *[rad_to_ui(a) for a in
                                               quat_to_euler(p['joint_orient'])])
            cmds.setAttr(f'{j}.rotate', *[rad_to_ui(a) for a in quat_to_euler(p['rotate'])])
            cmds.setAttr(f'{j}.radius', cm_to_ui(extent / 80.0))
            cmds.addAttr(j, longName=JOINT_ATTR, dataType='string')
            cmds.setAttr(f'{j}.{JOINT_ATTR}', p['name'], type='string')
            cmds.setAttr(f'{j}.{JOINT_ATTR}', lock=True)
            created[p['name']] = cmds.ls(j, long=True)[0]
        cmds.setAttr(f'{group}.overrideEnabled', 1)
        cmds.setAttr(f'{group}.overrideColor', 18)          # light blue
        root = record['proxy'][0]
        _set_json(group, RIG_ATTR, {
            'name': name,
            'server': server_url,
            'up_axis': record['up_axis'],
            'forward': record['forward'],
            'joints': [p['name'] for p in record['proxy']],
            'rest': {p['name']: {'rotate': p['rotate'], 'translate': p['translate']}
                     for p in record['proxy']},
            'bind_root_rotation': root['rotate'],
            'sources': {n: nodes[n] for n in record['joint_names'] if n in nodes},
        })
        set_clips(group, [])
        _set_json(group, LINKS_ATTR, [])
    return group


# ---------------------------------------------------------------------------
# Link / bake
# ---------------------------------------------------------------------------

def default_mapping(group):
    """``{proxy joint name: (target, mode)}``: each rig joint from its proxy
    joint; the root with a parent constraint, the rest orient."""
    info = rig_info(group)
    root = info['joints'][0]
    out = {}
    for name in info['joints']:
        target = info['sources'].get(name)
        if target and cmds.objExists(target):
            out[name] = (target, 'parent' if name == root else 'orient')
    return out


@contextlib.contextmanager
def proxy_at_rest(group):
    """Hold the proxy in its rest pose: disconnect its animation curves and set
    the rest values; reconnect on exit."""
    info = rig_info(group)
    joints = proxy_joints(group)
    saved = []
    for name, node in joints.items():
        conns = cmds.listConnections(node, source=True, destination=False, plugs=True,
                                     connections=True, type='animCurve') or []
        for dst, src in zip(conns[::2], conns[1::2]):
            cmds.disconnectAttr(src, dst)
            saved.append((src, dst))
    try:
        for name, node in joints.items():
            rest = info['rest'][name]
            cmds.setAttr(f'{node}.rotate', *[rad_to_ui(a) for a in quat_to_euler(
                rest['rotate'], cmds.getAttr(f'{node}.rotateOrder'))])
            cmds.setAttr(f'{node}.translate', *[cm_to_ui(v) for v in rest['translate']])
        yield
    finally:
        for src, dst in saved:
            cmds.connectAttr(src, dst, force=True)


def link(group, mapping=None):
    """Constrain rig targets to proxy joints. *mapping*:
    ``{proxy joint name: (target node, mode)}`` (default
    :func:`default_mapping`); mode one of :data:`MODES`. Replaces earlier
    links. Returns the links made."""
    info = rig_info(group)
    joints = proxy_joints(group)
    mapping = default_mapping(group) if mapping is None else mapping
    made = []
    with undo_chunk('UniMate: link rig'):
        unlink(group)
        with proxy_at_rest(group):
            for name, (target, mode) in mapping.items():
                if mode == 'none' or not target or name not in joints:
                    continue
                if mode not in MODES:
                    raise ValueError(f"link mode must be one of {MODES}, got {mode!r}")
                # a rig joint's own proxy joint matches its bind transform exactly
                exact = info['sources'].get(name) == target
                fn = {'orient': cmds.orientConstraint, 'parent': cmds.parentConstraint,
                      'point': cmds.pointConstraint}[mode]
                con = fn(joints[name], target, maintainOffset=not exact)[0]
                made.append({'proxy': name, 'target': target, 'mode': mode,
                             'constraint': cmds.ls(con, long=True)[0]})
        _set_json(group, LINKS_ATTR, made)
    return made


def unlink(group):
    """Delete the link constraints of *group* (the rig keeps its own keys)."""
    for item in links(group):
        if cmds.objExists(item['constraint']):
            cmds.delete(item['constraint'])
    if cmds.attributeQuery(LINKS_ATTR, node=group, exists=True):
        _set_json(group, LINKS_ATTR, [])


def keyed_range(group):
    """``(first, last)`` keyed frame of the proxy, or None."""
    times = cmds.keyframe(list(proxy_joints(group).values()), query=True, timeChange=True)
    return (min(times), max(times)) if times else None


def bake(group, start=None, end=None):
    """Bake the linked targets over *start*..*end* (default: the proxy's keyed
    range) and remove the links. Returns the baked targets."""
    made = links(group)
    if not made:
        raise ValueError("link the rig first")
    span = keyed_range(group)
    if span is None and (start is None or end is None):
        raise ValueError("the proxy has no keys to bake")
    start = span[0] if start is None else start
    end = span[1] if end is None else end
    attrs = {'orient': ['rx', 'ry', 'rz'], 'parent': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'],
             'point': ['tx', 'ty', 'tz']}
    plugs = [f"{m['target']}.{a}" for m in made for a in attrs[m['mode']]]
    with undo_chunk('UniMate: bake to rig'):
        cmds.bakeResults(plugs, time=(start, end), simulation=True, sampleBy=1,
                         disableImplicitControl=True, preserveOutsideKeys=True,
                         minimizeRotation=True)
        unlink(group)
    return sorted({m['target'] for m in made})
