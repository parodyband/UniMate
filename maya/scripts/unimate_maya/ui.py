"""The UniMate panel: set up a rig, prompt it, append the clips.

Server calls run on a worker thread; their results come back to the main
thread through a queued Qt signal, where every Maya command runs.
"""

import random
import threading

import shiboken6
from maya import cmds
from PySide6 import QtCore, QtGui, QtWidgets

from unimate_maya import client, proxy, skeleton, timeline

WINDOW_TITLE = 'UniMate'
CONTROL_NAME = 'unimateWorkspaceControl'
_panel = None


def _kill_jobs(jobs):
    for job in jobs:
        # None in batch mode, where scriptJob makes no jobs
        if job is not None and cmds.scriptJob(exists=job):
            cmds.scriptJob(kill=job, force=True)


class _Bridge(QtCore.QObject):
    """Carries ``(callback, value)`` from worker threads to the main thread.
    One for the session (:func:`_bridge`), so a request still in flight when
    a panel closes never emits into a deleted object."""
    call = QtCore.Signal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._dispatch, QtCore.Qt.QueuedConnection)

    @staticmethod
    def _dispatch(fn, value):
        owner = getattr(fn, '__self__', None)
        if isinstance(owner, QtCore.QObject) and not shiboken6.isValid(owner):
            return                                      # its panel has closed
        try:
            fn(value)
        except RuntimeError as exc:
            if 'already deleted' not in str(exc):
                raise


_BRIDGE = None


def _bridge():
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = _Bridge()
    return _BRIDGE


class Panel(QtWidgets.QWidget):
    """The panel's contents; :func:`show` docks it in Maya."""

    def __init__(self, parent=None, url=client.DEFAULT_URL):
        super().__init__(parent)
        self.setObjectName('unimatePanel')
        self.client = client.Client(url)
        self.bridge = _bridge()
        self.server_process = None
        self.skeleton = None            # last skeleton read from the scene
        self.nodes = {}                 # its joint name -> DAG path
        self.analysis = None
        self.takes = []                 # clips of the last generation
        self.busy = False
        self._build_ui()
        self.refresh_rigs()
        self.poll = QtCore.QTimer(self)
        self.poll.timeout.connect(self.check_server)
        self.poll.timeout.connect(self.refresh_history)
        self.poll.start(3000)
        self.check_server()
        # keep the clip list right through undo / redo and scene changes
        self.jobs = [cmds.scriptJob(event=[e, self._safe(self.refresh_history)])
                     for e in ('Undo', 'Redo')]
        self.jobs += [cmds.scriptJob(event=[e, self._safe(self.refresh_rigs)])
                      for e in ('SceneOpened', 'NewSceneOpened')]

    def _safe(self, fn):
        """A script-job callback that does nothing once the panel is gone."""
        def call():
            if self.jobs:
                try:
                    fn()
                except RuntimeError:        # Qt object already deleted
                    pass
        return call

    def shutdown(self):
        """Stop the timer and script jobs; call before the panel goes away
        (Maya commands must not run from Qt's destructors)."""
        self.poll.stop()
        _kill_jobs(self.jobs)
        self.jobs = []

    # -- layout ----------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # server
        row = QtWidgets.QHBoxLayout()
        self.server_label = QtWidgets.QLabel('Server: checking…')
        self.server_button = QtWidgets.QPushButton('Start server')
        self.server_button.clicked.connect(self.start_server)
        row.addWidget(self.server_label, 1)
        row.addWidget(self.server_button)
        layout.addLayout(row)

        # rig
        box = QtWidgets.QGroupBox('Rig')
        grid = QtWidgets.QGridLayout(box)
        self.rig_combo = QtWidgets.QComboBox()
        self.rig_combo.currentIndexChanged.connect(self.on_rig_changed)
        refresh = QtWidgets.QPushButton('↻')
        refresh.setToolTip('Find UniMate proxies in the scene')
        refresh.setFixedWidth(28)
        refresh.clicked.connect(self.refresh_rigs)
        grid.addWidget(QtWidgets.QLabel('Proxy'), 0, 0)
        grid.addWidget(self.rig_combo, 0, 1)
        grid.addWidget(refresh, 0, 2)
        self.setup_button = QtWidgets.QPushButton('Set up rig from selected joint…')
        self.setup_button.clicked.connect(self.toggle_setup)
        grid.addWidget(self.setup_button, 1, 0, 1, 3)
        layout.addWidget(box)

        self.setup = self._build_setup()
        self.setup.setVisible(False)
        layout.addWidget(self.setup)

        # generate
        box = QtWidgets.QGroupBox('Generate')
        form = QtWidgets.QGridLayout(box)
        self.prompt = QtWidgets.QLineEdit()
        self.prompt.setPlaceholderText('An object walks forward.')
        self.prompt.returnPressed.connect(self.generate)
        self.generate_button = QtWidgets.QPushButton('Generate')
        self.generate_button.clicked.connect(self.generate)
        form.addWidget(self.prompt, 0, 0, 1, 5)
        form.addWidget(self.generate_button, 0, 5)
        self.takes_spin = QtWidgets.QSpinBox()
        self.takes_spin.setRange(1, 8)
        self.takes_spin.setValue(2)
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText('random')
        self.seed_edit.setValidator(QtGui.QIntValidator(0, 2 ** 31 - 1))
        self.cfg_spin = QtWidgets.QDoubleSpinBox()
        self.cfg_spin.setRange(1.5, 10.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(3.0)
        self.cfg_spin.setToolTip('How strongly the prompt is followed (5-7 if ignored)')
        self.sampler_combo = QtWidgets.QComboBox()
        self.sampler_combo.addItems(['fast', 'quality'])
        self.sampler_combo.setToolTip('fast: ~2.5 s a take; quality: the reference solver, '
                                      '~3x slower')
        params = QtWidgets.QHBoxLayout()
        for label, widget in [('Takes', self.takes_spin), ('Seed', self.seed_edit),
                              ('Guidance', self.cfg_spin), ('Solver', self.sampler_combo)]:
            params.addWidget(QtWidgets.QLabel(label))
            params.addWidget(widget)
        form.addLayout(params, 1, 0, 1, 6)
        self.take_list = QtWidgets.QListWidget()
        self.take_list.setMaximumHeight(110)
        self.take_list.itemDoubleClicked.connect(lambda *_: self.append_take())
        form.addWidget(self.take_list, 2, 0, 1, 6)
        place = QtWidgets.QHBoxLayout()
        self.placement_combo = QtWidgets.QComboBox()
        self.placement_combo.addItem('Continue from last clip', 'continue')
        self.placement_combo.addItem('In place (rig position)', 'in_place')
        self.blend_spin = QtWidgets.QSpinBox()
        self.blend_spin.setRange(0, 30)
        self.blend_spin.setValue(6)
        self.blend_spin.setSuffix(' fr blend')
        self.append_button = QtWidgets.QPushButton('Append to timeline')
        self.append_button.clicked.connect(self.append_take)
        place.addWidget(self.placement_combo, 1)
        place.addWidget(self.blend_spin)
        place.addWidget(self.append_button)
        form.addLayout(place, 3, 0, 1, 6)
        layout.addWidget(box)

        # timeline
        box = QtWidgets.QGroupBox('Timeline (Ctrl+Z undoes an append)')
        col = QtWidgets.QVBoxLayout(box)
        self.history = QtWidgets.QListWidget()
        self.history.setMaximumHeight(110)
        self.history.itemDoubleClicked.connect(self.goto_clip)
        col.addWidget(self.history)
        layout.addWidget(box)

        # link / bake
        box = QtWidgets.QGroupBox('Drive the rig')
        col = QtWidgets.QVBoxLayout(box)
        self.map_table = QtWidgets.QTableWidget(0, 3)
        self.map_table.setHorizontalHeaderLabels(['Proxy joint', 'Rig target', 'Mode'])
        self.map_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.setMaximumHeight(160)
        col.addWidget(self.map_table)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in [
                ('Target ← selection', self.target_from_selection,
                 'Set the target of the highlighted rows to the selected node (e.g. a control)'),
                ('Link', self.link, 'Constrain the targets to the proxy'),
                ('Unlink', self.unlink, 'Remove the constraints'),
                ('Bake to rig', self.bake, 'Key the targets over the clips, then unlink')]:
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        col.addLayout(row)
        layout.addWidget(box)

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _build_setup(self):
        box = QtWidgets.QGroupBox('Rig set-up')
        grid = QtWidgets.QGridLayout(box)
        self.root_edit = QtWidgets.QLineEdit()
        self.root_edit.setPlaceholderText('root joint')
        pick = QtWidgets.QPushButton('← selection')
        pick.clicked.connect(self.pick_root)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText('rig name')
        analyze = QtWidgets.QPushButton('Analyze')
        analyze.clicked.connect(self.analyze)
        grid.addWidget(QtWidgets.QLabel('Root'), 0, 0)
        grid.addWidget(self.root_edit, 0, 1)
        grid.addWidget(pick, 0, 2)
        grid.addWidget(QtWidgets.QLabel('Name'), 1, 0)
        grid.addWidget(self.name_edit, 1, 1)
        grid.addWidget(analyze, 1, 2)
        self.joint_table = QtWidgets.QTableWidget(0, 4)
        self.joint_table.setHorizontalHeaderLabels(['Use', 'Joint', 'Label', 'Check'])
        header = self.joint_table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.joint_table.verticalHeader().setVisible(False)
        self.joint_table.setMinimumHeight(200)
        self.joint_table.itemChanged.connect(self._on_joint_item_changed)
        grid.addWidget(self.joint_table, 2, 0, 1, 3)
        face = QtWidgets.QHBoxLayout()
        self.face_right = QtWidgets.QComboBox()
        self.face_left = QtWidgets.QComboBox()
        self.body_axis = QtWidgets.QCheckBox('head / tail')
        self.body_axis.setToolTip('No mirrored limbs: Right is the head end, Left the tail end')
        face.addWidget(QtWidgets.QLabel('Facing: right'))
        face.addWidget(self.face_right, 1)
        face.addWidget(QtWidgets.QLabel('left'))
        face.addWidget(self.face_left, 1)
        face.addWidget(self.body_axis)
        grid.addLayout(face, 3, 0, 1, 3)
        self.budget_label = QtWidgets.QLabel('')
        build = QtWidgets.QPushButton('Build proxy')
        build.clicked.connect(self.build)
        grid.addWidget(self.budget_label, 4, 0, 1, 2)
        grid.addWidget(build, 4, 2)
        return box

    # -- helpers -----------------------------------------------------------------

    def say(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet('color: #e66;' if error else '')
        if error:
            cmds.warning(f'UniMate: {text}')

    def run(self, work, done, label):
        """Run *work()* on a thread; *done(result)* on the main thread."""
        if self.busy:
            self.say('Busy, wait for the current request.')
            return
        self.busy = True
        self.say(label)

        def target():
            try:
                result = work()
            except Exception as exc:                    # noqa: BLE001 - shown to the user
                self.bridge.call.emit(self._failed, exc)
                return
            self.bridge.call.emit(self._succeeded, (done, result))
        threading.Thread(target=target, daemon=True).start()

    def _succeeded(self, value):
        self.busy = False
        done, result = value
        try:
            done(result)
        except Exception as exc:                        # noqa: BLE001
            self.say(str(exc), error=True)

    def _failed(self, exc):
        self.busy = False
        self.say(str(exc), error=True)

    def group(self):
        return self.rig_combo.currentData()

    # -- server ------------------------------------------------------------------

    def check_server(self):
        def work():
            return self.client.health()

        def done(h):
            if h.get('error'):
                text = f"Server: model failed to load ({h['error']})"
            elif h.get('ready'):
                text = f"Server: ready ({h.get('device')}, {len(h.get('rigs', []))} rig(s))"
            else:
                text = 'Server: loading the model…'
            self.server_label.setText(text)
            self.server_button.setEnabled(False)

        def failed(_):
            self.server_label.setText('Server: not running')
            self.server_button.setEnabled(True)

        def target():
            try:
                self.bridge.call.emit(done, work())
            except client.ServerError as exc:
                self.bridge.call.emit(failed, exc)
        threading.Thread(target=target, daemon=True).start()

    def start_server(self):
        try:
            self.server_process = client.start_server()
        except client.ServerError as exc:
            self.say(str(exc), error=True)
            return
        self.server_button.setEnabled(False)
        self.server_label.setText('Server: starting…')
        self.say('Starting the UniMate server (the model takes ~15 s to load); '
                 'its log is outputs/serve.log in the UniMate checkout.')

    # -- rigs --------------------------------------------------------------------

    def refresh_rigs(self):
        current = self.group()
        self.rig_combo.blockSignals(True)
        self.rig_combo.clear()
        for g in proxy.proxies():
            self.rig_combo.addItem(proxy.rig_info(g)['name'], g)
        index = self.rig_combo.findData(current)
        self.rig_combo.setCurrentIndex(index if index >= 0 else 0)
        self.rig_combo.blockSignals(False)
        self.on_rig_changed()

    def on_rig_changed(self, *_):
        self.refresh_history()
        self.fill_mapping()

    def toggle_setup(self):
        visible = not self.setup.isVisible()
        self.setup.setVisible(visible)
        if visible and not self.root_edit.text():
            self.pick_root()

    def pick_root(self):
        sel = cmds.ls(selection=True, long=True, type='joint')
        if not sel:
            self.say('Select a joint of the skeleton (any joint; its root is used).', error=True)
            return
        root = skeleton.find_root(sel[0])
        self.root_edit.setText(root)
        self.name_edit.setText(skeleton.default_rig_name(root))

    def analyze(self):
        root = self.root_edit.text().strip()
        if not root or not cmds.objExists(root):
            self.say('Pick the root joint first.', error=True)
            return
        try:
            sk, nodes, notes = skeleton.read_skeleton(root, self.name_edit.text().strip() or None)
        except ValueError as exc:
            self.say(str(exc), error=True)
            return
        self.skeleton, self.nodes = sk, nodes

        def done(result):
            self.analysis = result
            self.fill_joint_table(result)
            self.say(' '.join(notes + result.get('notes', [])[:1]) or 'Analyzed.')
        self.run(lambda: self.client.analyze(sk), done, f'Analyzing {len(sk["joints"])} joints…')

    def fill_joint_table(self, result):
        table = self.joint_table
        table.blockSignals(True)
        table.setRowCount(0)
        for j in result['joints']:
            r = table.rowCount()
            table.insertRow(r)
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(QtCore.Qt.Checked if j['include'] else QtCore.Qt.Unchecked)
            if j.get('exclude_reason'):
                use.setToolTip(j['exclude_reason'])
            table.setItem(r, 0, use)
            name = QtWidgets.QTableWidgetItem(j['name'])
            name.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            table.setItem(r, 1, name)
            table.setItem(r, 2, QtWidgets.QTableWidgetItem(j['label']))
            check = QtWidgets.QTableWidgetItem('⚠' if j['check'] else '')
            check.setFlags(QtCore.Qt.ItemIsEnabled)
            if j['reasons']:
                check.setToolTip('\n'.join(j['reasons']))
            table.setItem(r, 3, check)
        table.resizeColumnToContents(0)
        table.resizeColumnToContents(3)
        table.blockSignals(False)
        face = result['face_pair']
        self._fill_face_combos(face.get('right', ''), face.get('left', ''))
        self.body_axis.setChecked(bool(face.get('body_axis')))
        tips = '\n'.join(face.get('reasons', []))
        self.face_right.setToolTip(tips)
        self.face_left.setToolTip(tips)
        self._update_budget()

    def _included(self):
        t = self.joint_table
        return [t.item(r, 1).text() for r in range(t.rowCount())
                if t.item(r, 0).checkState() == QtCore.Qt.Checked]

    def _fill_face_combos(self, right, left):
        names = [''] + self._included()
        for combo, value in ((self.face_right, right), (self.face_left, left)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(max(0, combo.findText(value)))
            combo.blockSignals(False)

    def _on_joint_item_changed(self, item):
        if item.column() == 0:
            self._fill_face_combos(self.face_right.currentText(), self.face_left.currentText())
            self._update_budget()

    def _update_budget(self):
        n = len(self._included())
        limit = (self.analysis or {}).get('max_joints', 70)
        self.budget_label.setText(f'{n} / {limit} joints')
        self.budget_label.setStyleSheet('color: #e66;' if n > limit else '')

    def build(self):
        if self.skeleton is None:
            self.say('Analyze the rig first.', error=True)
            return
        t = self.joint_table
        labels = {t.item(r, 1).text(): t.item(r, 2).text().strip() or 'Bone'
                  for r in range(t.rowCount())}
        face = {'right': self.face_right.currentText(), 'left': self.face_left.currentText(),
                'body_axis': self.body_axis.isChecked()}
        include = self._included()
        sk, nodes = self.skeleton, self.nodes

        def done(record):
            group = proxy.build_proxy(record, nodes, self.client.url)
            self.setup.setVisible(False)
            self.refresh_rigs()
            self.rig_combo.setCurrentIndex(self.rig_combo.findData(group))
            self.say(f"Built proxy {group.rsplit('|', 1)[-1]} with {len(record['proxy'])} "
                     f"joints. Prompt away.")
        self.run(lambda: self.client.build(sk, include, labels, face), done,
                 f'Building {sk["name"]} ({len(include)} joints)…')

    # -- generate / append -------------------------------------------------------

    def generate(self):
        group = self.group()
        prompt = self.prompt.text().strip()
        if not group:
            self.say('Set up a rig first.', error=True)
            return
        if not prompt:
            self.say('Type a prompt, e.g. "An object walks forward."', error=True)
            return
        rig = proxy.rig_info(group)['name']
        seed = int(self.seed_edit.text()) if self.seed_edit.text() else random.randrange(2 ** 31)
        takes, cfg = self.takes_spin.value(), self.cfg_spin.value()
        sampler = self.sampler_combo.currentText()

        def done(result):
            self.takes = result['clips']
            self.take_list.clear()
            for clip in self.takes:
                m = clip['meta']
                self.take_list.addItem(f"take {m['take'] + 1}: {m['prompt']}  "
                                       f"(seed {m['seed']}, {clip['frame_count']} fr @30)")
            self.take_list.setCurrentRow(0)
            self.say(f"{len(self.takes)} take(s) in {result['seconds']:.1f}s. "
                     f"Pick one and append it.")
        self.run(lambda: self.client.generate(rig, prompt, takes=takes, seed=seed,
                                              cfg_scale=cfg, sampler=sampler),
                 done, f'Generating {takes} take(s) of "{prompt}"…')

    def append_take(self):
        group, row = self.group(), self.take_list.currentRow()
        if not group or row < 0 or row >= len(self.takes):
            self.say('Generate and pick a take first.', error=True)
            return
        try:
            first, last = timeline.append_clip(
                group, self.takes[row], placement=self.placement_combo.currentData(),
                blend_frames=self.blend_spin.value(), goto=True)
        except ValueError as exc:
            self.say(str(exc), error=True)
            return
        self.refresh_history()
        self.say(f'Appended frames {first:g}-{last:g}. Ctrl+Z to take it back.')

    def refresh_history(self):
        """List the proxy's clips (on undo / redo, and on the poll timer as a
        fallback); left alone when nothing changed, to keep the selection."""
        group = self.group()
        clips = proxy.clips(group) if group and cmds.objExists(group) else []
        rows = [(f"{c['start']:g}-{c['end']:g}  {c.get('prompt', '')}  "
                 f"(seed {c.get('seed', '?')})", c['start']) for c in clips]
        if rows == getattr(self, '_history_rows', None):
            return
        self._history_rows = rows
        self.history.clear()
        for text, start in rows:
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, start)
            self.history.addItem(item)

    def goto_clip(self, item):
        cmds.currentTime(item.data(QtCore.Qt.UserRole))

    # -- link / bake -------------------------------------------------------------

    def fill_mapping(self):
        table = self.map_table
        table.setRowCount(0)
        group = self.group()
        if not group or not cmds.objExists(group):
            return
        mapping = proxy.default_mapping(group)
        linked = {m['proxy']: (m['target'], m['mode']) for m in proxy.links(group)}
        for name in proxy.rig_info(group)['joints']:
            target, mode = linked.get(name) or mapping.get(name) or ('', 'none')
            r = table.rowCount()
            table.insertRow(r)
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            table.setItem(r, 0, item)
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(target))
            combo = QtWidgets.QComboBox()
            combo.addItems(proxy.MODES)
            combo.setCurrentText(mode)
            table.setCellWidget(r, 2, combo)

    def _mapping(self):
        t = self.map_table
        return {t.item(r, 0).text(): (t.item(r, 1).text().strip(),
                                      t.cellWidget(r, 2).currentText())
                for r in range(t.rowCount())}

    def target_from_selection(self):
        sel = cmds.ls(selection=True, long=True)
        rows = sorted({i.row() for i in self.map_table.selectedIndexes()})
        if not sel or not rows:
            self.say('Highlight rows here and select a node in the scene.', error=True)
            return
        for r in rows:
            self.map_table.item(r, 1).setText(sel[0])

    def link(self):
        group = self.group()
        if not group:
            return
        try:
            made = proxy.link(group, self._mapping())
        except (ValueError, RuntimeError) as exc:
            self.say(str(exc), error=True)
            return
        self.say(f'Linked {len(made)} target(s) to the proxy.')

    def unlink(self):
        group = self.group()
        if group:
            with proxy.undo_chunk('UniMate: unlink rig'):
                proxy.unlink(group)
            self.say('Unlinked.')

    def bake(self):
        group = self.group()
        if not group:
            return
        try:
            baked = proxy.bake(group)
        except ValueError as exc:
            self.say(str(exc), error=True)
            return
        self.say(f'Baked {len(baked)} target(s) and removed the links.')


def show():
    """Open (or raise) the panel docked as a Maya workspace control."""
    global _panel
    from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

    class Dockable(MayaQWidgetDockableMixin, QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setObjectName(CONTROL_NAME.replace('WorkspaceControl', ''))
            self.setWindowTitle(WINDOW_TITLE)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            self.panel = Panel()
            scroll.setWidget(self.panel)
            layout.addWidget(scroll)

        def dockCloseEventTriggered(self):
            self.panel.shutdown()

    name = CONTROL_NAME.replace('WorkspaceControl', '') + 'WorkspaceControl'
    if _panel is not None:
        try:
            _panel.panel.shutdown()
        except RuntimeError:
            pass
    if cmds.workspaceControl(name, exists=True):
        cmds.deleteUI(name)
    _panel = Dockable()
    _panel.show(dockable=True, area='right', floating=False, width=420)
    return _panel
