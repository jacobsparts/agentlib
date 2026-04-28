from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import time
from dataclasses import dataclass
import json
from typing import Iterable, Optional

EVENT_PREFIX = "[[AGENTLIB_EVENT:"
EVENT_SUFFIX = "]]"
EVENT_PATTERN = re.compile(r"\[\[AGENTLIB_EVENT:(.*?)\]\]")


@dataclass
class PTYRunResult:
    args: list[str]
    returncode: int
    output: str
    events: list[dict] | None = None


class PTYTimeoutError(TimeoutError):
    def __init__(self, message: str, *, output: str = "", events: list[dict] | None = None):
        super().__init__(message)
        self.output = output
        self.events = events or []


def parse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for match in EVENT_PATTERN.finditer(text):
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _prompt_idle(output: bytearray, inputs_done: bool) -> bool:
    if not inputs_done:
        return False
    decoded = output.decode(errors="replace")
    stripped = decoded.rstrip()
    if not stripped.endswith(">"):
        return False
    after_last_prompt = decoded.rsplit("\n>", 1)[-1].strip() if "\n>" in decoded else ""
    return after_last_prompt == "" or after_last_prompt == ">"


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
    prompt_idle_timeout: float = 3.0,
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
        inputs_done = False
        saw_wait_for = wait_for is None
        eof_sent_at = 0.0
        self_terminated = False
        while True:
            now = time.monotonic()
            if now - last_output_at > timeout:
                raise PTYTimeoutError(
                    f"Timed out after {timeout:g}s without output: {' '.join(args)}",
                    output=output.decode(errors="replace"),
                    events=parse_events(output.decode(errors="replace")),
                )

            if input_queue and now - last_send >= input_delay:
                chunk = input_queue.pop(0)
                os.write(master_fd, chunk.encode())
                last_send = now
                if not input_queue:
                    inputs_done = True

            idle = now - last_output_at
            if inputs_done and final_eof and not eof_sent_at:
                if saw_wait_for and _prompt_idle(output, inputs_done) and idle >= prompt_idle_timeout:
                    os.write(master_fd, b"\x04")
                    eof_sent_at = now

            if eof_sent_at and now - eof_sent_at > 5.0 and proc.poll() is None:
                self_terminated = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    proc.terminate()

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
                    if wait_for and not saw_wait_for and wait_for in output.decode(errors="replace"):
                        saw_wait_for = True
                elif proc.poll() is not None:
                    break

            if proc.poll() is not None:
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
        rc = proc.returncode or 0
        if self_terminated and rc < 0:
            rc = 0
        decoded = output.decode(errors="replace")
        return PTYRunResult(args=args, returncode=rc, output=decoded, events=parse_events(decoded))
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


def strip_events(text: str) -> str:
    return EVENT_PATTERN.sub("", text)
