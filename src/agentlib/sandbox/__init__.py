"""
Sandbox: Filesystem isolation for agent execution using Linux overlay filesystems.

All filesystem modifications are captured in a temporary overlay layer.
When the session ends, changes are available as a tarball that can be
reviewed, applied, or discarded.

Requires Linux with unprivileged user namespaces enabled.
"""

import os
import shutil
import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent
_SOURCE_FILE = _PACKAGE_DIR / "sandbox_helper.c"
_BINARY_FILE = _PACKAGE_DIR / "sandbox_helper"


def _check_requirements() -> tuple[bool, str]:
    """Check if sandbox requirements are met."""
    # Check for Linux
    if os.uname().sysname != "Linux":
        return False, "Sandbox requires Linux (uses user namespaces and overlayfs)"

    # Check for unprivileged user namespaces
    try:
        with open("/proc/sys/kernel/unprivileged_userns_clone") as f:
            if f.read().strip() != "1":
                return False, "Unprivileged user namespaces disabled. Enable with: sysctl kernel.unprivileged_userns_clone=1"
    except FileNotFoundError:
        pass  # File may not exist on some kernels, assume enabled

    return True, ""


def _compile_helper() -> str:
    """Compile sandbox_helper if needed. Returns path to binary."""
    if _BINARY_FILE.exists() and os.access(_BINARY_FILE, os.X_OK):
        # Check if source is newer than binary
        if _SOURCE_FILE.stat().st_mtime <= _BINARY_FILE.stat().st_mtime:
            return str(_BINARY_FILE)

    # Need to compile
    if not _SOURCE_FILE.exists():
        raise FileNotFoundError(f"sandbox_helper.c not found at {_SOURCE_FILE}")

    # Check for gcc
    gcc = shutil.which("gcc")
    if not gcc:
        raise RuntimeError("gcc not found. Install with: apt install build-essential")

    # Compile
    result = subprocess.run(
        [gcc, "-o", str(_BINARY_FILE), str(_SOURCE_FILE)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile sandbox_helper: {result.stderr}")

    # Make executable
    _BINARY_FILE.chmod(0o755)

    return str(_BINARY_FILE)


def get_sandbox_helper() -> str:
    """Get path to sandbox_helper binary, compiling if needed."""
    ok, msg = _check_requirements()
    if not ok:
        raise RuntimeError(msg)

    return _compile_helper()


# Export mixin classes
from .mixin import SandboxMixin, SandboxedToolREPL, SandboxCrashError

__all__ = ["SandboxMixin", "SandboxedToolREPL", "SandboxCrashError", "get_sandbox_helper"]
