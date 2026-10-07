"""Drive the UniMate panel headless (offscreen Qt in mayapy) through set-up,
generate, append, undo, link and bake, with a running server.

    mayapy maya/tests/test_ui_smoke.py        (from the repository root)
"""

import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'maya', 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

from PySide6 import QtCore, QtWidgets  # noqa: E402

# before maya.standalone, which would otherwise make a widget-less QGuiApplication
APP = QtWidgets.QApplication([])

import maya.standalone  # noqa: E402

maya.standalone.initialize(name='python')

from maya import cmds  # noqa: E402

import test_headless as th  # noqa: E402
from unimate_maya import proxy, timeline, ui  # noqa: E402

FAILED = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)


def wait(panel, seconds=120):
    """Process Qt events until the panel's request is done."""
    end = time.time() + seconds
    QtWidgets.QApplication.processEvents()
    while panel.busy and time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


def main():
    app = APP
    cmds.loadPlugin('fbxmaya', quiet=True)
    cmds.undoInfo(state=True, infinity=True)
    root = th.import_fbx(th.EAGLE_FBX, 'eagle')

    panel = ui.Panel()
    end = time.time() + 10
    while 'ready' not in panel.server_label.text() and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    check('ready' in panel.server_label.text(), f"server status: {panel.server_label.text()}")

    print("set-up")
    cmds.select(cmds.listRelatives(root, allDescendents=True, type='joint', fullPath=True)[0])
    panel.toggle_setup()
    check(panel.root_edit.text() == root, "root found from a selected child joint")
    panel.name_edit.setText('EagleUI')
    panel.analyze()
    wait(panel)
    check(panel.joint_table.rowCount() == 51, f"joint table lists 51 joints "
                                              f"({panel.joint_table.rowCount()})")
    check(panel.face_right.currentText().endswith('Bip01_R_Thigh'),
          f"facing pair suggested ({panel.face_right.currentText()} / "
          f"{panel.face_left.currentText()})")
    # leave out the two wing tips: the budget and facing lists follow
    for r in range(panel.joint_table.rowCount()):
        if panel.joint_table.item(r, 1).text().endswith(('BN_Wing_L_06', 'BN_Wing_R_06')):
            panel.joint_table.item(r, 0).setCheckState(QtCore.Qt.Unchecked)
    check(panel.budget_label.text().startswith('49 '), f"budget: {panel.budget_label.text()}")
    panel.build()
    wait(panel)
    group = panel.group()
    check(bool(group) and len(proxy.proxy_joints(group)) == 49,
          f"proxy built with 49 joints ({panel.status.text()})")

    print("generate and append")
    panel.prompt.setText('An object flaps its wings.')
    panel.takes_spin.setValue(2)
    panel.seed_edit.setText('21')
    panel.generate()
    wait(panel)
    check(panel.take_list.count() == 2, f"2 takes listed ({panel.status.text()})")
    panel.take_list.setCurrentRow(1)
    panel.append_take()
    span1 = timeline.last_key(group)
    panel.prompt.setText('An object walks forward.')
    panel.generate()
    wait(panel)
    panel.append_take()
    check(panel.history.count() == 2, f"clip list shows 2 clips ({panel.history.count()})")
    print(f"  undo queue head: {cmds.undoInfo(query=True, undoName=True)!r}")
    cmds.undo()
    panel.poll.timeout.emit()       # script jobs don't fire headless; the poll catches it
    app.processEvents()
    check(timeline.last_key(group) == span1,
          f"Ctrl+Z takes the last clip back (last key {timeline.last_key(group)}, "
          f"expected {span1})")
    check(panel.history.count() == 1, f"the clip list follows ({panel.history.count()})")

    print("link and bake")
    check(panel.map_table.rowCount() == 49, "mapping table lists the proxy joints")
    panel.link()
    check(len(proxy.links(group)) == 49, f"linked ({panel.status.text()})")
    panel.bake()
    check(not proxy.links(group), f"baked and unlinked ({panel.status.text()})")

    print("close")
    panel.shutdown()
    jobs_left = [j for j in cmds.scriptJob(listJobs=True) or [] if 'refresh' in j]
    check(not jobs_left, "shutdown removes the panel's script jobs")
    panel.deleteLater()
    del panel
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)
    cmds.undo()                     # an Undo event after the panel is gone
    check(True, "the panel goes away cleanly")

    print('PASSED' if not FAILED else f'FAILED: {len(FAILED)} check(s)')
    return 1 if FAILED else 0


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    # Qt + Maya teardown aborts in mayapy (and leaves a recovered scene in
    # TEMP): end the process without running it
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32')
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)
