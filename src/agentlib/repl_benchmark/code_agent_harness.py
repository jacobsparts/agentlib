from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class PTYRunResult:
    args: list[str]
    returncode: int
    output: str


class PTYTimeoutError(TimeoutError):
    def __init__(self, message: str, *, output: str = ""):
        super().__init__(message)
        self.output = output


def run_pty_session(
    args: list[str],
    *,
    inputs: Iterable[str] = (),
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 15.0,
    read_chunk: int = 4096,
    input_delay: float = 0.2,
    final_eof: bool = True,
    wait_for: str | None = None,
    stream_output=None,
) -> PTYRunResult:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        args,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=cwd,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave_fd)
    output = bytearray()
    try:
        last_output_at = time.monotonic()
        input_queue = list(inputs)
        last_send = 0.0
        saw_wait_for = wait_for is None
        while True:
            now = time.monotonic()
            if now - last_output_at > timeout:
                raise PTYTimeoutError(
                    f"Timed out after {timeout:g}s without output: {' '.join(args)}",
                    output=output.decode(errors="replace"),
                )

            if input_queue and now - last_send >= input_delay:
                chunk = input_queue.pop(0)
                os.write(master_fd, chunk.encode())
                last_send = now

            if not input_queue and final_eof and saw_wait_for:
                os.write(master_fd, b"\x04")
                final_eof = False

            ready, _, _ = select.select([master_fd], [], [], 0.05)
            if ready:
                try:
                    data = os.read(master_fd, read_chunk)
                except OSError:
                    data = b""
                if data:
                    last_output_at = now
                    output.extend(data)
                    if stream_output is not None:
                        stream_output.write(data.decode(errors="replace"))
                        stream_output.flush()
                    if wait_for and wait_for in output.decode(errors="replace"):
                        saw_wait_for = True
                elif proc.poll() is not None:
                    break

            if proc.poll() is not None:
                # drain any trailing output
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        data = os.read(master_fd, read_chunk)
                    except OSError:
                        break
                    if not data:
                        break
                    output.extend(data)
                    if stream_output is not None:
                        stream_output.write(data.decode(errors="replace"))
                        stream_output.flush()
                break
        return PTYRunResult(args=args, returncode=proc.returncode or 0, output=output.decode(errors="replace"))
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait(timeout=1)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)