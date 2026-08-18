"""Process-start safety checks for framework resources."""

from __future__ import annotations

import os


_fork_child = False


def _after_fork_child() -> None:
    global _fork_child
    _fork_child = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def assert_safe_process_context() -> None:
    """Reject framework resource setup in a process created with fork."""
    if _fork_child:
        raise RuntimeError(
            "Unsafe process start: this process was created with fork after "
            "the framework was imported. Use multiprocessing.get_context('spawn') "
            "or a fresh subprocess before creating framework resources."
        )
