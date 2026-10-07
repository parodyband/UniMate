"""Read a joint hierarchy from the scene in its bind pose, in the form the
UniMate server takes (``unimate.serve.rig_config``).

Bind-pose world matrices come from, in order of preference: the inverse of
the skinCluster ``bindPreMatrix`` the joint is bound with, the ``bindPose``
dagPose, and, for a joint with neither, its current local transform under its
parent's bind pose (reported, since a posed joint then gives a posed rest).

Those bind matrices are where the rig stood when it was bound. A rig moved,
turned or scaled since (its group, or anything above the skeleton) is put
back where it stands now: the topmost bound joint is placed by its bind-time
local transform under its parent's current world matrix, and every bound
joint follows it rigidly. Matrices are read through the API, so they are in
internal units (cm) whatever the UI unit.
"""

from maya import cmds
from maya.api import OpenMaya as om


def _dag_path(node):
    sel = om.MSelectionList()
    sel.add(node)
    return sel.getDagPath(0)


def _mobject(node):
    sel = om.MSelectionList()
    sel.add(node)
    return sel.getDependNode(0)


def _matrix_plug_value(plug):
    data = om.MFnMatrixData(plug.asMObject())
    if data.isTransformation():         # e.g. dagPose.xformMatrix
        return data.transformation().asMatrix()
    return data.matrix()


def joints_under(root):
    """Long names of *root* and every joint below it, parents first."""
    root = cmds.ls(root, long=True, type='joint')
    if not root:
        raise ValueError("pick a joint (the root of the skeleton)")
    out, stack = [], [root[0]]
    while stack:
        node = stack.pop()
        out.append(node)
        children = cmds.listRelatives(node, children=True, type='joint', fullPath=True) or []
        stack.extend(reversed(children))
    return out


def find_root(node):
    """The topmost joint above *node* (or *node* itself)."""
    node = cmds.ls(node, long=True)[0]
    while True:
        parent = cmds.listRelatives(node, parent=True, fullPath=True, type='joint')
        if not parent:
            return node
        node = parent[0]


def _skin_bind_matrices(joints):
    """``{joint: bind world MMatrix}`` from the skinClusters the joints drive."""
    out = {}
    for joint in joints:
        fn = om.MFnDependencyNode(_mobject(joint))
        plug = fn.findPlug('worldMatrix', False)
        for i in range(plug.numElements()):
            element = plug.elementByPhysicalIndex(i)
            for dest in element.connectedTo(False, True):
                node = dest.node()
                if not node.hasFn(om.MFn.kSkinClusterFilter):
                    continue
                skin = om.MFnDependencyNode(node)
                pre = skin.findPlug('bindPreMatrix', False).elementByLogicalIndex(
                    dest.logicalIndex())
                out[joint] = _matrix_plug_value(pre).inverse()
                break
            if joint in out:
                break
    return out


def _dagpose_bind_matrices(joints, attr='worldMatrix'):
    """``{joint: MMatrix}`` from ``bindPose`` dagPose nodes: the joint's
    ``message`` feeds ``members[i]``; its bind world matrix is
    ``worldMatrix[i]``, its bind local transform ``xformMatrix[i]``."""
    out = {}
    for joint in joints:
        plugs = cmds.listConnections(f'{joint}.message', type='dagPose', plugs=True,
                                     source=False, destination=True) or []
        for plug in plugs:
            pose, member = plug.split('.', 1)
            if not member.startswith('members[') or not cmds.getAttr(f'{pose}.bindPose'):
                continue
            index = int(member[len('members['):-1])
            values = om.MFnDependencyNode(_mobject(pose)).findPlug(attr, False)
            out[joint] = _matrix_plug_value(values.elementByLogicalIndex(index))
            break
    return out


def _local_now(joint):
    """*joint*'s current local matrix (relative to its parent)."""
    path = _dag_path(joint)
    return path.inclusiveMatrix() * path.exclusiveMatrixInverse()


def default_rig_name(root):
    """The rig's namespace, else the top transform above *root*, else *root*."""
    short = root.rsplit('|', 1)[-1]
    if ':' in short:
        return short.rsplit(':', 1)[0].replace(':', '_')
    top = root.strip('|').split('|')[0]
    return top.rsplit(':', 1)[-1]


def _flat(m):
    return [m[i] for i in range(16)]


def short_names(joints):
    """``{long name: unique name}`` for the server: the short name, or, where
    short names repeat, the partial path with ``|`` turned into ``__``."""
    shorts = [j.rsplit('|', 1)[-1] for j in joints]
    out = {}
    for long_name, short in zip(joints, shorts):
        if shorts.count(short) == 1:
            out[long_name] = short
        else:
            partial = cmds.ls(long_name, shortNames=True)[0]
            out[long_name] = partial.strip('|').replace('|', '__')
    return out


def read_skeleton(root, name=None):
    """``(skeleton, nodes, notes)``: the server's skeleton JSON for the joint
    hierarchy under *root*, ``{joint name: long DAG path}``, and notes on where
    the bind pose came from."""
    joints = joints_under(root)
    names = short_names(joints)
    skin = _skin_bind_matrices(joints)
    missing = [j for j in joints if j not in skin]
    posed = _dagpose_bind_matrices(missing) if missing else {}
    bind = dict(posed, **skin)
    notes = []
    if skin:
        notes.append(f"{len(skin)} joint(s) from the skinCluster bind matrices")
    if posed:
        notes.append(f"{len(posed)} joint(s) from the bindPose")
    current = [j for j in joints if j not in bind]
    if current:
        notes.append(f"{len(current)} joint(s) without a bind pose take their current local "
                     f"transform; put the rig in its rest pose first")

    # where the bound skeleton stands now: its topmost bound joint, by its
    # bind-time local transform, under its parent's current world matrix
    world = {}
    placed = om.MMatrix()
    top = next((j for j in joints if j in bind), None)
    if top is not None:
        local = (_dagpose_bind_matrices([top], 'xformMatrix').get(top)
                 or _local_now(top))
        placed = bind[top].inverse() * (local * _dag_path(top).exclusiveMatrix())
        if not placed.isEquivalent(om.MMatrix(), 1e-6):
            notes.append("the rig has moved since it was bound: its bind pose is placed "
                         "where it stands now")
    out = []
    for joint in joints:
        parent = cmds.listRelatives(joint, parent=True, fullPath=True, type='joint')
        if joint in bind:
            m = bind[joint] * placed
        elif parent and parent[0] in world:
            m = _local_now(joint) * world[parent[0]]
        else:
            m = _dag_path(joint).inclusiveMatrix()
        world[joint] = m
        out.append({'name': names[joint],
                    'parent': names[parent[0]] if parent and parent[0] in names else None,
                    'matrix': _flat(m), 'skinned': joint in skin})
    rig_name = name or default_rig_name(joints[0])
    skeleton = {'name': rig_name, 'up_axis': cmds.upAxis(query=True, axis=True),
                'joints': out}
    return skeleton, {names[j]: j for j in joints}, notes
