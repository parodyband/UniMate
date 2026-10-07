# shellcheck shell=bash
# Source in Git Bash before running anything (after scripts/setup_windows.sh):
# activates the env and puts the Blender 4.0 app on PATH as `blender`.
#
#   source scripts/env_windows.sh

_unimate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$_unimate_root/.venv/Scripts/activate"
export PATH="$_unimate_root/.tools/blender-4.0.2-windows-x64:$PATH"
# Windows consoles default to cp1252, which cannot print the logs' arrows and dashes
export PYTHONUTF8=1
unset _unimate_root
