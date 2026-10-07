#!/bin/bash
# Windows setup (Git Bash): the Python env, the Blender 4.0 app in place of the
# pip bpy module, and the released checkpoint.
#
# requirements.txt pins the pip bpy==4.0.0 module, which cannot be installed on
# Windows: PyPI no longer has it and download.blender.org/pypi refuses
# non-browser clients. The Blender 4.0.2 app bundles the same Python 3.10, so a
# .pth file in its site-packages puts the repository and the env's
# site-packages on its path, the env first (the cond.npy pickles need the env's
# numpy 2). Scripts that call `blender -b` then work unchanged; run a module
# that imports bpy with scripts/bpy_run.sh instead of `python -m`.
#
# Usage:
#   bash scripts/setup_windows.sh      # once; finished steps are skipped
#   source scripts/env_windows.sh      # in every new shell
#
# Needs uv (https://docs.astral.sh/uv/).
#
# Env overrides:
#   BLENDER_MIRROR   (default: https://mirrors.ocf.berkeley.edu/blender/release/Blender4.0)
#   SKIP_CHECKPOINT  1: do not download the released checkpoint

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_WIN=$(cygpath -w "$PWD")

BLENDER_VERSION=4.0.2
BLENDER_DIR=.tools/blender-$BLENDER_VERSION-windows-x64
BLENDER_MIRROR=${BLENDER_MIRROR:-https://mirrors.ocf.berkeley.edu/blender/release/Blender4.0}
EXP=unimate_uniml3d_f60_v3
CKPT_URL=https://huggingface.co/Linzhan/UniMate/resolve/main/$EXP

echo "== Python env (.venv)"
[ -x .venv/Scripts/python.exe ] || uv venv .venv --python 3.10
uv pip install --python .venv "setuptools<81" wheel
grep -vE '^bpy==|download\.blender\.org' requirements.txt > .venv/requirements-windows.txt
uv pip install --python .venv -r .venv/requirements-windows.txt \
    --index-strategy unsafe-best-match --no-build-isolation

echo "== Blender $BLENDER_VERSION ($BLENDER_DIR)"
if [ ! -x "$BLENDER_DIR/blender.exe" ]; then
    mkdir -p .tools
    zip=.tools/blender-$BLENDER_VERSION-windows-x64.zip
    curl -fL "$BLENDER_MIRROR/blender-$BLENDER_VERSION.sha256" -o .tools/blender-$BLENDER_VERSION.sha256
    curl -fL "$BLENDER_MIRROR/blender-$BLENDER_VERSION-windows-x64.zip" -o "$zip"
    expected=$(awk '/windows-x64\.zip/ {print $1}' .tools/blender-$BLENDER_VERSION.sha256)
    actual=$(sha256sum "$zip" | cut -d' ' -f1)
    if [ "$expected" != "$actual" ]; then
        echo "ERROR: checksum mismatch for $zip" >&2
        exit 1
    fi
    # Windows' tar reads zip archives; Git Bash's GNU tar does not
    /c/Windows/System32/tar.exe -xf "$zip" -C .tools
    rm "$zip"
fi
printf '%s\nimport sys; sys.path.insert(0, r"%s\\.venv\\Lib\\site-packages")\n' "$ROOT_WIN" "$ROOT_WIN" \
    > "$BLENDER_DIR/${BLENDER_VERSION%.*}/python/lib/site-packages/unimate_venv.pth"

if [ "${SKIP_CHECKPOINT:-0}" != 1 ]; then
    echo "== Checkpoint (outputs/$EXP)"
    mkdir -p "outputs/$EXP/checkpoints"
    for f in config.json dataset_stats.npy checkpoints/checkpoint_step_150000.pt; do
        [ -s "outputs/$EXP/$f" ] || curl -fL "$CKPT_URL/$f" -o "outputs/$EXP/$f"
    done
fi

echo "Done. Run 'source scripts/env_windows.sh' in each new shell."
