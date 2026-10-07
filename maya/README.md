# UniMate for Maya

Prompt a rig in Maya and append the generated motion to its timeline.

1. **Set up a rig.** Pick a joint of a rig; the panel labels its joints, finds its facing, and builds a *proxy*: a clean copy of the skeleton (`UNIMATE_<rig>`) sitting on the rig.
2. **Prompt.** Type "An object walks forward." and get one or more takes, about 2.5 s each.
3. **Append.** Pick a take and append it after the proxy's last key. Each append is one undo step, so Ctrl+Z takes a clip back.
4. **Drive the rig.** Link the rig to the proxy, then bake when you are happy. The proxy follows any rig: joint-only skeletons by default, control rigs by pointing each proxy joint at a control.

Clips are generated one by one and never look at the timeline: each is a standalone 2 s motion (60 frames at 30 fps, resampled to the scene's rate).

## Install

The model runs in a separate process: Maya 2027's Python is 3.13 and UniMate needs the checkout's 3.10 environment with torch. From the repository root, in Git Bash:

```bash
bash scripts/setup_windows.sh            # once: the env, Blender 4.0, the checkpoint
bash scripts/install_maya_module.sh      # once: unimate.mod in ~/Documents/maya/modules
```

Restart Maya, then in the Script Editor (Python):

```python
import unimate_maya; unimate_maya.show()
```

Click **Start server** in the panel; the model loads in about 15 s and the log goes to `outputs/serve.log`. Or run it yourself:

```bash
source scripts/env_windows.sh
python -m unimate.serve
```

It listens on `127.0.0.1:8792`, keeps built rigs in `outputs/maya_rigs/`, and has no authentication, so keep it local.

## Use

**Rig set-up.** Select any joint of the skeleton, then click **Set up rig from selected joint…** and **Analyze**. Check the table:

- **Use**: the joints the model sees, at most 70. Over budget, helper, face and finger joints are left out first. A joint left out isn't driven; it keeps its rest pose relative to its parent.
- **Label**: what each joint is. The model reads the labels, so they matter. Follow `data_process/rig_preprocess/AGENTS.md`, for example:
  - sides as a prefix: `Left Thigh`;
  - chains repeat a label: `Spine`, `Spine`;
  - forelegs of quadrupeds are arms.
  
  ⚠ marks labels worth a look.
- **Facing**: two mirrored joints, right first (the thighs, else the shoulders or fins). A body without mirrored limbs uses head / tail.

**Build proxy** writes the rig to the server and creates the proxy. The bind pose comes from the skinCluster, else the bindPose. A rig moved, turned or scaled since binding is fine.

**Generate.** Type a prompt in the training captions' style: start with "An object" and describe one motion. More takes give more variety.

- **Guidance**: how hard the prompt is pushed. The default is 3; raise it to 5–7 when the prompt is ignored.
- **Seed**: makes a take repeatable.
- **Solver**:
  - `fast`: 20 midpoint steps, ~2.5 s a take on an RTX 4070 Ti SUPER;
  - `quality`: the reference adaptive solver, about 3x slower.

**Append.** Pick a take, then **Append to timeline**, or double-click it.

- **Continue from last clip** turns and moves the clip so it starts where and how the last one ended.
- **In place** leaves it where it was generated: at the rig, facing the rig's forward.
- The **blend** frames ease the seam from the previous pose.

The timeline list shows every clip with its prompt and seed; double-click one to jump to it.

**Drive the rig.** The table maps each proxy joint to a rig target and a mode:

| Mode | Use |
|---|---|
| `orient` | rotation only; the default for every joint but the root |
| `parent` | position and rotation; the default for the root |
| `point` | position only |
| `none` | leave the target alone |

The defaults drive each rig joint from its own proxy joint, exactly, in any pose. For a control rig, highlight rows, select a control, then **Target ← selection**. Those links keep their offsets, so link them while the rig and the proxy are both at rest (go to a frame before the first clip). **Link** shows the motion on the rig live. **Bake to rig** keys the targets over the clips' frame range and removes the links.

## How it works

| Piece | Where |
|---|---|
| Server: rig set-up, model, clips | `unimate/serve/`: `rig_config.py`, `engine.py`, `retarget.py`, `server.py` |
| Maya: read the rig, the proxy, the timeline, the panel | `maya/scripts/unimate_maya/` |

**Set-up.** The server builds the rig's UniMate asset without Blender. It runs `rig_preprocess`'s annotation and the stage-4 cond calls on the joints Maya sends, and stores the similarity from the canonical T-pose to the bind pose (`outputs/maya_rigs/<rig>/rig.json`).

**Clips.** A clip (`unimate-clip/1`) is JSON: each proxy joint's rotate channel as a quaternion per frame, and the root's world path. Nothing in it is UniMate-specific, so another generator could feed the same panel.

**Proxy joints.** A proxy joint has its rig joint's bind transform, with jointOrient holding the bind rotation. The rotate channels are then `Q⁻¹ N Q` (`Q = Rᵀ T`), the same formula the Blender animation path keys.

## Tests

With the server running, from the repository root:

```bash
.venv/Scripts/python.exe maya/tests/test_retarget.py
"/c/Program Files/Autodesk/Maya2027/bin/mayapy.exe" maya/tests/test_headless.py
"/c/Program Files/Autodesk/Maya2027/bin/mayapy.exe" maya/tests/test_ui_smoke.py
```

| Test | Checks |
|---|---|
| `test_retarget.py` | The Eagle's cond rebuilt from its joints matches the shipped one. A keyed proxy reproduces the clip, and the clip equals Blender's animation of the same motion (within 1e-5 of the rig's size, every joint and frame). |
| `test_headless.py` | In Maya: keying, append / seam / undo / redo, link and bake, and a moved, scaled rig at 24 fps. |
| `test_ui_smoke.py` | Drives the panel offscreen from set-up to bake. |

They need the example outputs from the top-level README (`outputs/samples/custom`) and the FBX test rigs from `rig_preprocess` (`outputs/rig/eagle_fbx`, `outputs/rig/shark_test`).

## Limits

- **Clip length.** Clips are 2 s; longer actions are several prompts appended.
- **Not driven:** leaf joints (no rotation in the model's output), joints left out of the rig, and the translation of every joint but the root.
- **IK.** IK pole vectors aren't mapped automatically.
- **License.** The released checkpoints are CC BY-NC 4.0 (non-commercial; see the top-level README). This code is MIT.
