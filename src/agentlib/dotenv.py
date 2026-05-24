import os
import re
from pathlib import Path


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def find_dotenv(filename=".env", start=None):
    path = Path(start or os.getcwd()).resolve()
    if path.is_file():
        path = path.parent

    for directory in (path, *path.parents):
        candidate = directory / filename
        if candidate.is_file():
            return str(candidate)
    return ""


def _strip_inline_comment(value):
    quote = None
    escaped = False
    for i, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value.rstrip()


def _decode_quoted(value):
    if len(value) < 2 or value[0] != value[-1] or value[0] not in {"'", '"'}:
        return value

    quote = value[0]
    inner = value[1:-1]
    if quote == "'":
        return inner

    escapes = {
        "\\\\": "\\",
        '\\"': '"',
        "\\'": "'",
        "\\n": "\n",
        "\\r": "\r",
        "\\t": "\t",
    }
    for escaped, replacement in escapes.items():
        inner = inner.replace(escaped, replacement)
    return inner


def dotenv_values(dotenv_path=None, encoding="utf-8"):
    if dotenv_path is None:
        dotenv_path = find_dotenv()
    if not dotenv_path:
        return {}

    path = Path(dotenv_path).expanduser()
    if not path.is_file():
        return {}

    values = {}
    for raw_line in path.read_text(encoding=encoding).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not _KEY_RE.match(key):
            continue

        value = _strip_inline_comment(value.strip())
        values[key] = _decode_quoted(value)
    return values


def load_dotenv(dotenv_path=None, override=False, encoding="utf-8"):
    values = dotenv_values(dotenv_path, encoding=encoding)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return bool(values)
