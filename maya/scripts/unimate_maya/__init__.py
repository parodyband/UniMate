"""UniMate for Maya: prompt a rig, append the generated clips to its timeline.

    import unimate_maya
    unimate_maya.show()

The model runs in a separate server process (``python -m unimate.serve`` in
the UniMate checkout's env); this package reads the rig, builds its proxy
skeleton, and keys clips. See ``maya/README.md``.
"""


def show():
    """Open (or raise) the UniMate panel."""
    from unimate_maya import ui
    return ui.show()
