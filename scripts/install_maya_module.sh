#!/bin/bash
# Install the UniMate Maya tool as a Maya module: write unimate.mod into the
# user's Maya modules folder, pointing at this checkout's maya/ directory, so
# Maya puts maya/scripts on its Python path and sets UNIMATE_ROOT.
#
# Usage:
#   bash scripts/install_maya_module.sh [MODULES_DIR]
#
# MODULES_DIR defaults to ~/Documents/maya/modules (Windows) or
# ~/maya/modules. Remove the tool by deleting MODULES_DIR/unimate.mod.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        ROOT_NATIVE="$(cygpath -w "$ROOT")"
        MAYA_DIR="$ROOT_NATIVE\\maya"
        DEFAULT="$(cygpath -u "$USERPROFILE")/Documents/maya/modules"
        ;;
    *)
        ROOT_NATIVE="$ROOT"
        MAYA_DIR="$ROOT/maya"
        DEFAULT="$HOME/maya/modules"
        ;;
esac
DIR="${1:-$DEFAULT}"
mkdir -p "$DIR"
cat > "$DIR/unimate.mod" <<EOF
+ unimate_maya 0.1 $MAYA_DIR
UNIMATE_ROOT=$ROOT_NATIVE
EOF
echo "Wrote $DIR/unimate.mod -> $MAYA_DIR"
echo "Restart Maya, then run in the Script Editor (Python):"
echo "    import unimate_maya; unimate_maya.show()"
