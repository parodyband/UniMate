#!/bin/bash
# Run a Python module that imports bpy inside the Blender app, for setups
# without the pip bpy module (Windows: the bpy 4.0 wheel is not obtainable).
# Blender 4.0 bundles Python 3.10, the env's version, so with the env's
# site-packages on Blender's path (scripts/setup_windows.sh) the module runs
# exactly as `python -m <module>` would.
#
# Usage:
#   bash scripts/bpy_run.sh <module> [args...]
#   bash scripts/bpy_run.sh data_process.rig_preprocess run --input robot.glb --output_dir outputs/rig/robot
#
# Env overrides:
#   BLENDER   Blender executable (default: blender on PATH)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ "$#" -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

"${BLENDER:-blender}" -b --factory-startup --python-exit-code 1 --python-expr "
import os, runpy, sys
sys.path.insert(0, os.getcwd())
argv = sys.argv[sys.argv.index('--') + 1:]
sys.argv = argv
runpy.run_module(argv[0], run_name='__main__', alter_sys=True)
" -- "$@"
