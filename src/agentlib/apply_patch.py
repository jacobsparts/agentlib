#!/usr/bin/env python3

"""
A self-contained **pure-Python 3.9+** utility for applying human-readable
"pseudo-diff" patch files to a collection of text files.
"""

from __future__ import annotations

import difflib
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


# --------------------------------------------------------------------------- #
#  Instructions / Documentation
# --------------------------------------------------------------------------- #
APPLY_PATCH_INSTRUCTIONS = """
# apply_patch - Patch Format Documentation

A file-oriented diff format for adding, removing, moving, or editing code files.

## Patch Structure

Every patch is wrapped in begin/end markers:

    *** Begin Patch
    [ one or more file operations ]
    *** End Patch

## File Operations

Each operation starts with one of three headers:

- `*** Add File: <path>` - Create a new file. Every following line must start with `+`.
- `*** Delete File: <path>` - Remove an existing file. Nothing follows.
- `*** Update File: <path>` - Patch an existing file in place.
  - Optionally followed by `*** Move to: <new path>` to rename the file.
  - Then one or more "hunks", each introduced by `@@`.

## Hunk Format

Within an update hunk, each line starts with:
- ` ` (space) - Context line (unchanged)
- `-` - Line to remove
- `+` - Line to add

### Context Guidelines

- Show 3 lines of context above and below each change.
- If changes are within 3 lines of each other, don't duplicate context.
- Use `@@` with a function/class name if 3 lines aren't enough to uniquely identify location:

    @@ class BaseClass
     [context lines]
    -[old code]
    +[new code]
     [context lines]

- Chain multiple `@@` statements for deeper nesting:

    @@ class BaseClass
    @@     def method():
     [context lines]
    -[old code]
    +[new code]

- Use `*** End of File` to anchor changes at the end of a file.

## Grammar

    Patch     := Begin { FileOp } End
    Begin     := "*** Begin Patch" NEWLINE
    End       := "*** End Patch" NEWLINE
    FileOp    := AddFile | DeleteFile | UpdateFile
    AddFile   := "*** Add File: " path NEWLINE { "+" line NEWLINE }
    DeleteFile:= "*** Delete File: " path NEWLINE
    UpdateFile:= "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
    MoveTo    := "*** Move to: " newPath NEWLINE
    Hunk      := "@@" [ " " context ] NEWLINE { HunkLine } [ EndOfFile ]
    HunkLine  := (" " | "-" | "+") text NEWLINE
    EndOfFile := "*** End of File" NEWLINE

## Example Patch

    *** Begin Patch
    *** Add File: hello.txt
    +Hello world
    *** Update File: src/app.py
    *** Move to: src/main.py
    @@ def greet():
    -print("Hi")
    +print("Hello, world!")
    *** Delete File: obsolete.txt
    *** End Patch

## Python API

### Basic Usage

    from apply_patch import process_patch

    patch_text = '''*** Begin Patch
    *** Update File: example.py
    @@
    -old line
    +new line
    *** End Patch'''

    result = process_patch(patch_text)
    print(result)  # Prints summary of changes

### Preview Changes Without Applying

    from apply_patch import preview_patch

    previews = preview_patch(patch_text)
    for path, preview in previews.items():
        print(f"{preview.type.value}: {path}")
        if preview.unified_diff:
            print(preview.unified_diff)

### Generate Unified Diffs

    from apply_patch import generate_unified_diff

    diff = generate_unified_diff(old_content, new_content, "file.py")
    print(diff)

## Return Values

- `process_patch()` returns a summary string listing affected files (A/M/D prefixes).
- `preview_patch()` returns `Dict[str, FileChangePreview]` with unified diffs and content.
- `apply_commit()` returns `AffectedPaths` with lists of added/modified/deleted paths.

## Error Handling

All errors inherit from `DiffError`:
- `ParseError` - Invalid patch syntax
- `IoError` - File read/write failures
- `ContextNotFoundError` - Could not locate context lines in file

## Notes

- Paths should be relative, not absolute.
- New lines in `*** Add File` must be prefixed with `+`.
- Unicode punctuation (smart quotes, em-dashes, etc.) is normalized during matching.
- Whitespace-insensitive matching is attempted if exact matching fails.
"""


# --------------------------------------------------------------------------- #
#  Domain objects
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


@dataclass
class AffectedPaths:
    """Tracks file paths affected by applying a patch."""
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)


@dataclass
class FileChangePreview:
    """Preview of a file change without applying it."""
    type: ActionType
    path: str
    unified_diff: Optional[str] = None
    new_content: Optional[str] = None
    old_content: Optional[str] = None
    move_path: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """Base error for patch operations."""


class ParseError(DiffError):
    """Error parsing patch syntax."""

    def __init__(self, message: str, line_number: Optional[int] = None):
        self.message = message
        self.line_number = line_number
        if line_number is not None:
            super().__init__(f"line {line_number}: {message}")
        else:
            super().__init__(message)


class IoError(DiffError):
    """Error reading/writing files."""

    def __init__(self, message: str, path: str):
        super().__init__(f"{message}: {path}")
        self.path = path


class ContextNotFoundError(DiffError):
    """Could not find context lines in file."""

    def __init__(self, context: str, path: str):
        super().__init__(f"Failed to find context in {path}:\n{context}")
        self.context = context
        self.path = path


# --------------------------------------------------------------------------- #
#  Helper dataclasses used while parsing patches
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Patch text parser
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)

    # ------------- low-level helpers -------------------------------------- #
    @property
    def line_number(self) -> int:
        """Current line number (1-based)."""
        return self.index + 1

    def _error(self, message: str) -> ParseError:
        """Create a ParseError with the current line number."""
        return ParseError(message, self.line_number)

    def _cur_line(self) -> str:
        if self.index >= len(self.lines):
            raise ParseError("Unexpected end of input while parsing patch", self.line_number)
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """Strip CR so comparisons work for both LF and CRLF input."""
        return line.rstrip("\r")

    # ------------- scanning convenience ----------------------------------- #
    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if (
            prefixes
            and len(prefixes) > 0
            and self._norm(self._cur_line()).startswith(prefixes)
        ):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        Consume the current line if it starts with *prefix* and return the text
        **after** the prefix.  Raises if prefix is empty.
        """
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """Return the current raw line and advance."""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- public entry point -------------------------------------- #
    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            line_num = self.line_number  # Capture before consuming

            # ---------- UPDATE ---------- #
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise ParseError(f"Duplicate update for file: {path}", line_num)
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise ParseError(f"Update File Error - missing file: {path}", line_num)
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- DELETE ---------- #
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise ParseError(f"Duplicate delete for file: {path}", line_num)
                if path not in self.current_files:
                    raise ParseError(f"Delete File Error - missing file: {path}", line_num)
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- ADD ---------- #
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise ParseError(f"Duplicate add for file: {path}", line_num)
                if path in self.current_files:
                    raise ParseError(f"Add File Error - file already exists: {path}", line_num)
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise self._error(f"Unknown line while parsing: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise self._error("Missing *** End Patch sentinel")
        self.index += 1  # consume sentinel

    # ------------- section parsers ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise self._error(f"Invalid line in update section: {self._cur_line()}")

            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [
                    s.strip() for s in lines[:index]
                ]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            found = True
                            break

            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)

            # Pure addition handling: if no context/deletions, insert at EOF
            # (or just before the final empty line if one exists)
            if not next_ctx:
                insertion_idx = len(lines)
                if lines and lines[-1] == "":
                    insertion_idx = len(lines) - 1
                for ch in chunks:
                    ch.orig_index = insertion_idx
                    action.chunks.append(ch)
                self.index = end_idx
                continue

            new_index, matched_len = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise self._error(
                    f"Could not find {'EOF ' if eof else ''}context in file:\n{ctx_txt}"
                )
            # Adjust chunk indices: orig_index was computed relative to parsed context,
            # but if trailing empty line retry matched fewer lines, we need to adjust
            context_adjustment = len(next_ctx) - matched_len
            for ch in chunks:
                ch.orig_index = ch.orig_index - context_adjustment + new_index
                action.chunks.append(ch)
            index = new_index + matched_len
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            line_num = self.line_number
            s = self.read_line()
            if not s.startswith("+"):
                raise ParseError(f"Invalid Add File line (missing '+'): {s}", line_num)
            lines.append(s[1:])  # strip leading '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
_UNICODE_REPLACEMENTS: Dict[str, str] = {
    # Dashes → '-'
    '\u2010': '-', '\u2011': '-', '\u2012': '-',
    '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-',
    # Single quotes → "'"
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
    # Double quotes → '"'
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
    # Spaces → ' '
    '\u00a0': ' ', '\u2002': ' ', '\u2003': ' ', '\u2004': ' ',
    '\u2005': ' ', '\u2006': ' ', '\u2007': ' ', '\u2008': ' ',
    '\u2009': ' ', '\u200a': ' ', '\u202f': ' ', '\u205f': ' ',
    '\u3000': ' ',
}


def _normalize_unicode(s: str) -> str:
    """Normalize Unicode punctuation to ASCII equivalents."""
    return ''.join(_UNICODE_REPLACEMENTS.get(c, c) for c in s.strip())


def find_context_core(lines: List[str], context: List[str], start: int) -> int:
    """
    Find context lines in file, using progressively looser matching.
    Returns the index where context was found, or -1 if not found.
    """
    if not context:
        return start

    # Exact match
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i
    # Rstrip match
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [
            s.rstrip() for s in context
        ]:
            return i
    # Strip match
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [
            s.strip() for s in context
        ]:
            return i
    # Unicode-normalized match
    for i in range(start, len(lines)):
        if [_normalize_unicode(s) for s in lines[i : i + len(context)]] == [
            _normalize_unicode(s) for s in context
        ]:
            return i
    return -1


def find_context(
    lines: List[str], context: List[str], start: int, eof: bool
) -> Tuple[int, int]:
    """
    Find context lines, with special handling for EOF markers.
    For EOF, tries matching at end of file first, then falls back to normal search.
    Also handles trailing empty line retry for end-of-file modifications.

    Returns (index, matched_length) where:
    - index: position where context was found, or -1 if not found
    - matched_length: number of lines that were actually matched (may be less than
      len(context) if trailing empty line retry was used)
    """
    if not context:
        return start, 0

    if eof:
        # Try matching at end of file first
        new_index = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, len(context)
        # Fall back to searching from start
        new_index = find_context_core(lines, context, start)
        if new_index != -1:
            return new_index, len(context)
    else:
        new_index = find_context_core(lines, context, start)
        if new_index != -1:
            return new_index, len(context)

    # Trailing empty line retry: if context ends with empty string and match failed,
    # retry without the trailing empty line (handles end-of-file modifications)
    if context and context[-1] == "":
        trimmed_context = context[:-1]
        if trimmed_context:
            if eof:
                new_index = find_context_core(lines, trimmed_context, len(lines) - len(trimmed_context))
                if new_index != -1:
                    return new_index, len(trimmed_context)
            new_index = find_context_core(lines, trimmed_context, start)
            if new_index != -1:
                return new_index, len(trimmed_context)

    return -1, 0


def peek_next_section(
    lines: List[str], index: int
) -> Tuple[List[str], List[Chunk], int, bool]:
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise ParseError(f"Invalid line: {s}", index + 1)
        index += 1

        last_mode = mode

        # Handle empty lines as context (keep) lines with empty content
        if s == "":
            mode = "keep"
            line_content = ""
        elif s[0] == "+":
            mode = "add"
            line_content = s[1:]
        elif s[0] == "-":
            mode = "delete"
            line_content = s[1:]
        elif s[0] == " ":
            mode = "keep"
            line_content = s[1:]
        else:
            raise ParseError(f"Invalid line (must start with ' ', '+', or '-'): {s}", index)

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(line_content)
            old.append(line_content)
        elif mode == "add":
            ins_lines.append(line_content)
        elif mode == "keep":
            old.append(line_content)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise ParseError("Empty section (no content lines found)", orig_index + 1)
    return old, chunks, index, False


# --------------------------------------------------------------------------- #
#  Patch → Commit and Commit application
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk.orig_index {chunk.orig_index} exceeds file length"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: overlapping chunks at {orig_index} > {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: Dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE, old_content=orig[path]
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD action without file content")
            commit.changes[path] = FileChange(
                type=ActionType.ADD, new_content=action.new_file
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  User-facing helpers
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: Dict[str, str]) -> Patch:
    lines = text.splitlines()  # preserves blank lines, no strip()
    if len(lines) < 2:
        raise ParseError("Patch too short - must have at least Begin and End markers")
    if not Parser._norm(lines[0]).startswith("*** Begin Patch"):
        raise ParseError("Patch must start with '*** Begin Patch'", 1)
    if Parser._norm(lines[-1]) != "*** End Patch":
        raise ParseError("Patch must end with '*** End Patch'", len(lines))

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Update File: ") :]
        for line in lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: ") :]
        for line in lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Add File: ") :]
        for line in lines
        if line.startswith("*** Add File: ")
    ]


def generate_unified_diff(
    old_content: str, new_content: str, path: str = "", context: int = 3
) -> str:
    """Generate unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    # Ensure files end with newline for proper diff formatting
    if old_lines and not old_lines[-1].endswith('\n'):
        old_lines[-1] += '\n'
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'
    return ''.join(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=path, tofile=path,
        n=context
    ))


# --------------------------------------------------------------------------- #
#  File-system operations
# --------------------------------------------------------------------------- #
def _assert_absolute_path(path: str) -> None:
    """Ensure path is absolute. Raises ValueError if not."""
    if not path.startswith("/"):
        raise ValueError(f"Path must be absolute (start with '/'): {path}")


def _read_file(path: str) -> str:
    """Read file contents as UTF-8 text."""
    _assert_absolute_path(path)
    with open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def _write_file(path: str, content: str) -> None:
    """Write content to file, creating parent directories if needed."""
    _assert_absolute_path(path)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wt", encoding="utf-8") as fh:
        fh.write(content)


def _remove_file(path: str) -> None:
    """Remove file if it exists."""
    _assert_absolute_path(path)
    pathlib.Path(path).unlink(missing_ok=True)


def _load_files(paths: List[str]) -> Dict[str, str]:
    """Load multiple files into a dictionary."""
    return {path: _read_file(path) for path in paths}


# --------------------------------------------------------------------------- #
#  Patch application
# --------------------------------------------------------------------------- #
def preview_patch(text: str) -> Dict[str, FileChangePreview]:
    """
    Parse and validate patch, returning preview of changes without applying.

    This allows inspecting what changes would be made before actually applying them.
    """
    paths = identify_files_needed(text)
    orig = _load_files(paths)
    patch = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)

    previews: Dict[str, FileChangePreview] = {}
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            previews[path] = FileChangePreview(
                type=ActionType.DELETE,
                path=path,
                old_content=change.old_content,
            )
        elif change.type is ActionType.ADD:
            previews[path] = FileChangePreview(
                type=ActionType.ADD,
                path=path,
                new_content=change.new_content,
            )
        elif change.type is ActionType.UPDATE:
            unified_diff = generate_unified_diff(
                change.old_content or "",
                change.new_content or "",
                path,
            )
            previews[path] = FileChangePreview(
                type=ActionType.UPDATE,
                path=path,
                unified_diff=unified_diff,
                old_content=change.old_content,
                new_content=change.new_content,
                move_path=change.move_path,
            )
    return previews


def apply_commit(commit: Commit) -> AffectedPaths:
    """Apply a commit to the filesystem and return the affected paths."""
    affected = AffectedPaths()
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            _remove_file(path)
            affected.deleted.append(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"ADD change for {path} has no content")
            _write_file(path, change.new_content)
            affected.added.append(path)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {path} has no new content")
            target = change.move_path or path
            _write_file(target, change.new_content)
            if change.move_path:
                _remove_file(path)
            affected.modified.append(target)
    return affected


def print_summary(affected: AffectedPaths) -> str:
    """Generate a git-style summary of affected files."""
    lines = ["Success. Updated the following files:"]
    for path in affected.added:
        lines.append(f"A {path}")
    for path in affected.modified:
        lines.append(f"M {path}")
    for path in affected.deleted:
        lines.append(f"D {path}")
    return '\n'.join(lines)


def process_patch(text: str) -> str:
    """
    Parse and apply a patch to the filesystem.

    Returns a summary string listing affected files.
    """
    if not text.startswith("*** Begin Patch"):
        raise ParseError("Patch must start with '*** Begin Patch'", 1)
    paths = identify_files_needed(text)
    orig = _load_files(paths)
    patch = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)
    affected = apply_commit(commit)
    return print_summary(affected)

