import io
import sys

from agentlib.cli.altmode import AltMode, Session


class DummyStdout(io.StringIO):
    def isatty(self):
        return True

    @property
    def encoding(self):
        return "utf-8"


def test_session_exit_restores_cursor_visible(monkeypatch):
    original = DummyStdout()
    monkeypatch.setattr(sys, "stdout", original)
    alt = AltMode()
    alt._original_stdout = original
    alt._installed = True
    session = Session(alt)
    session._active = True
    session.exit()
    out = original.getvalue()
    assert "\x1b[?25h" in out
    assert "\x1b[?1049l" in out