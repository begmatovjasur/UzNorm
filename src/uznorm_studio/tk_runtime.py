"""No GUI imports until requested. Standard Python installations need no overrides."""
import os
import sys
from pathlib import Path


def prepare_tk():
    # An optional, user-supplied Tcl directory supports portable Python runtimes.
    # Never download libraries or alter global Python configuration automatically.
    custom = os.environ.get("UZNORM_TCL_ROOT")
    root = Path(custom) if custom else Path(sys.base_prefix) / "tcl"
    if custom or (root / "tcl8.6/init.tcl").is_file():
        for variable, child, required in (("TCL_LIBRARY", "tcl8.6", "init.tcl"),
                                           ("TK_LIBRARY", "tk8.6", "tk.tcl")):
            directory = Path(root) / child
            if not (directory / required).is_file():
                raise RuntimeError("UZNORM_TCL_ROOT ichida Tcl/Tk fayllari topilmadi.")
            if custom or variable not in os.environ:
                # Portable Tcl on this Windows host fails to resolve some absolute
                # paths. A relative path works without copying or changing libraries.
                try:
                    os.environ[variable] = Path(os.path.relpath(directory.resolve())).as_posix()
                except ValueError:  # Different Windows drives cannot be relativized.
                    os.environ[variable] = directory.resolve().as_posix()
