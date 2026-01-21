"""
Terminal rendering utilities: ANSI codes, markdown renderer, Panel, Console.

This module provides terminal output formatting without external dependencies.
"""

import re
import shutil
from typing import Optional

# =============================================================================
# ANSI Escape Codes
# =============================================================================

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"
STRIKE = "\x1b[9m"

# Basic colors
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
WHITE = "\x1b[37m"
GRAY = "\x1b[90m"

# Extended colors (256-color mode) for markdown
H1 = "\x1b[38;5;226m"
H2 = "\x1b[38;5;214m"
H3 = "\x1b[38;5;118m"
H4 = "\x1b[38;5;21m"
H5 = "\x1b[38;5;93m"
H6 = "\x1b[38;5;239m"
TEXT = "\x1b[38;5;7m"
BLOCKQUOTE_BG = "\x1b[48;5;236m"
CODEBLOCK = "\x1b[97m"
LIST_COLOR = "\x1b[36m"
HR_COLOR = "\x1b[36m"
LINK_COLOR = "\x1b[38;5;45m"

# Python syntax highlighting colors
KW_COLOR = "\x1b[38;5;204m"
STRING_COLOR = "\x1b[38;5;114m"
NUMBER_COLOR = "\x1b[38;5;220m"
COMMENT_COLOR = "\x1b[38;5;245m"
FUNCTION_COLOR = "\x1b[38;5;81m"
CLASS_COLOR = "\x1b[38;5;214m"
BUILTIN_COLOR = "\x1b[38;5;147m"

# Default theme
DEFAULT_THEME = {
    'panel_border': CYAN,
    'panel_title': BOLD,
    'code_border': BLUE,
    'output_border': GREEN,
    'error_border': RED,
    'dim': DIM,
}


# =============================================================================
# Markup Parsing (rich-style tags like [bold], [red], etc.)
# =============================================================================

MARKUP_MAP = {
    'bold': BOLD,
    'dim': DIM,
    'italic': ITALIC,
    'underline': UNDERLINE,
    'strike': STRIKE,
    'red': RED,
    'green': GREEN,
    'yellow': YELLOW,
    'blue': BLUE,
    'magenta': MAGENTA,
    'cyan': CYAN,
    'white': WHITE,
    'gray': GRAY,
}


def parse_markup(text: str) -> str:
    """Convert rich-style markup [bold], [red], etc. to ANSI codes."""
    result = text
    # Handle closing tags first
    result = re.sub(r'\[/[^\]]*\]', RESET, result)
    # Handle opening tags
    for tag, code in MARKUP_MAP.items():
        result = result.replace(f'[{tag}]', code)
    return result


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def get_terminal_width() -> int:
    """Get terminal width with fallback."""
    try:
        size = shutil.get_terminal_size()
        return size.columns if size.columns > 0 else 80
    except Exception:
        return 80


# =============================================================================
# Python Syntax Highlighting
# =============================================================================

PYTHON_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "False", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "None",
    "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
    "while", "with", "yield"
}

PYTHON_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr", "dict",
    "dir", "enumerate", "eval", "exec", "filter", "float", "format", "getattr",
    "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max", "min", "next",
    "object", "oct", "open", "ord", "pow", "print", "property", "range", "repr",
    "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
    "str", "sum", "super", "tuple", "type", "vars", "zip"
}


def highlight_python(code: str) -> str:
    """Apply basic syntax highlighting to Python code.

    Uses placeholder markers to avoid regex matching numbers inside ANSI codes.
    """
    # Use placeholder markers that won't appear in normal code
    MARK_START = '\x00S'  # Start marker
    MARK_END = '\x00E'    # End marker

    lines = []
    for line in code.split('\n'):
        # Handle comments first (they take precedence)
        comment_part = ''
        if '#' in line:
            # Check if # is inside a string
            in_string = False
            quote_char = None
            for i, c in enumerate(line):
                if c in '"\'':
                    if not in_string:
                        in_string = True
                        quote_char = c
                    elif c == quote_char:
                        in_string = False
                elif c == '#' and not in_string:
                    comment_part = line[i:]
                    line = line[:i]
                    break

        # Mark strings with placeholders to protect them
        protected = []
        def protect_string(m):
            protected.append(m.group(0))
            return f'{MARK_START}{len(protected) - 1}{MARK_END}'

        line = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|\"[^\"]*\"|\'[^\']*\')',
                      protect_string, line, flags=re.DOTALL)

        # Handle numbers (now safe since strings are protected)
        line = re.sub(r'\b(\d+\.?\d*)\b', NUMBER_COLOR + r'\1' + CODEBLOCK, line)

        # Handle keywords
        for kw in PYTHON_KEYWORDS:
            line = re.sub(rf'\b({kw})\b', KW_COLOR + r'\1' + CODEBLOCK, line)

        # Handle builtins
        for bi in PYTHON_BUILTINS:
            line = re.sub(rf'\b({bi})\b', BUILTIN_COLOR + r'\1' + CODEBLOCK, line)

        # Restore protected strings with highlighting
        def restore_string(m):
            idx = int(m.group(1))
            return STRING_COLOR + protected[idx] + CODEBLOCK

        line = re.sub(rf'{MARK_START}(\d+){MARK_END}', restore_string, line)

        # Add back comment with highlighting
        if comment_part:
            line = line + COMMENT_COLOR + comment_part + CODEBLOCK

        lines.append(line)
    return '\n'.join(lines)


# =============================================================================
# Markdown Rendering
# =============================================================================

def _colorize_inline(line: str) -> str:
    """Apply inline markdown formatting (bold, italic, code, links, etc.)."""
    # Inline code
    line = re.sub(r'`([^`]+)`', CODEBLOCK + r'`\1`' + RESET + TEXT, line)
    # Bold+Italic
    line = re.sub(r'\*\*\*(.+?)\*\*\*', BOLD + ITALIC + r'\1' + RESET + TEXT, line)
    # Bold
    line = re.sub(r'\*\*(.+?)\*\*', BOLD + r'\1' + RESET + TEXT, line)
    # Italic (avoiding list markers)
    line = re.sub(r'(?<!\*)\*([^\*\n]+)\*(?!\*)', ITALIC + r'\1' + RESET + TEXT, line)
    # Strikethrough
    line = re.sub(r'~~(.+?)~~', STRIKE + r'\1' + RESET + TEXT, line)
    # Links
    line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', LINK_COLOR + UNDERLINE + r'\1' + RESET + TEXT, line)
    return line


def _wrap_text(text: str, width: int) -> list:
    """Wrap text to specified width, respecting ANSI codes."""
    if not text or len(strip_ansi(text)) <= width:
        return [text]

    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    current_len = len(strip_ansi(words[0]))

    for word in words[1:]:
        word_len = len(strip_ansi(word))
        if current_len + word_len + 1 <= width:
            current += " " + word
            current_len += word_len + 1
        else:
            lines.append(current)
            current = word
            current_len = word_len

    lines.append(current)
    return lines


def render_markdown(text: str, width: Optional[int] = None) -> str:
    """Convert markdown text to ANSI-formatted output."""
    if width is None:
        width = get_terminal_width() - 2

    lines = text.split('\n')
    result = []
    in_code_block = False
    code_lang = None
    code_buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code blocks
        fence_match = re.match(r'^(```|~~~)(.*)$', line)
        if fence_match:
            if in_code_block:
                # End of code block - render buffered code
                code = '\n'.join(code_buffer)
                if code_lang and code_lang.lower() in ('python', 'py'):
                    code = highlight_python(code)
                for code_line in code.split('\n'):
                    result.append(f"{CODEBLOCK}{code_line}{RESET}")
                in_code_block = False
                code_lang = None
                code_buffer = []
            else:
                # Start of code block
                in_code_block = True
                code_lang = fence_match.group(2).strip() or None
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # Horizontal rules
        if re.match(r'^(\-{3,}|={3,}|_{3,})\s*$', line):
            result.append(f"{HR_COLOR}{'─' * (width - 1)}{RESET}")
            i += 1
            continue

        # Blockquotes
        if re.match(r'^\s*>', line):
            content = re.sub(r'^\s*> ?', '', line)
            content = _colorize_inline(content)
            for wrapped in _wrap_text(content, width - 4):
                result.append(f"{TEXT}  > {BLOCKQUOTE_BG}{wrapped}{RESET}")
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            text_content = heading_match.group(2)
            colors = [H1, H2, H3, H4, H5, H6]
            color = colors[level - 1]
            text_content = _colorize_inline(text_content)
            result.append(f"{BOLD}{color}{text_content}{RESET}")
            i += 1
            continue

        # Task lists
        task_match = re.match(r'^(\s*)[\-\*]\s+\[([ x])\]\s+(.*)', line)
        if task_match:
            indent = task_match.group(1)
            status = task_match.group(2)
            content = _colorize_inline(task_match.group(3))
            checkbox = f"[{BOLD}x{RESET}{LIST_COLOR}]" if status == 'x' else "[ ]"
            indent_str = "  " * (len(indent) // 2)
            result.append(f"{indent_str}{LIST_COLOR}* {checkbox} {TEXT}{content}{RESET}")
            i += 1
            continue

        # Unordered lists
        list_match = re.match(r'^(\s*)[\-\*]\s+(.*)', line)
        if list_match:
            indent = list_match.group(1)
            content = _colorize_inline(list_match.group(2))
            indent_str = "  " * (len(indent) // 2)
            for j, wrapped in enumerate(_wrap_text(content, width - len(indent_str) - 4)):
                if j == 0:
                    result.append(f"{indent_str}{LIST_COLOR}* {TEXT}{wrapped}{RESET}")
                else:
                    result.append(f"{indent_str}  {wrapped}")
            i += 1
            continue

        # Ordered lists
        ordered_match = re.match(r'^(\s*)(\d+)\.[ \t]+(.*)', line)
        if ordered_match:
            indent = ordered_match.group(1)
            num = ordered_match.group(2)
            content = _colorize_inline(ordered_match.group(3))
            indent_str = "  " * (len(indent) // 2)
            num_width = len(num) + 2
            for j, wrapped in enumerate(_wrap_text(content, width - len(indent_str) - num_width - 2)):
                if j == 0:
                    result.append(f"{indent_str}{LIST_COLOR}{num}. {TEXT}{wrapped}{RESET}")
                else:
                    result.append(f"{indent_str}{' ' * num_width}{wrapped}")
            i += 1
            continue

        # Regular text
        if line.strip():
            colored = f"{TEXT}{_colorize_inline(line)}"
            for wrapped in _wrap_text(colored, width):
                result.append(f"{wrapped}{RESET}")
        else:
            result.append("")

        i += 1

    return '\n'.join(result)


# =============================================================================
# Panel (bordered box)
# =============================================================================

class Panel:
    """A simple bordered panel for terminal output."""

    BORDER_COLORS = {
        'cyan': CYAN,
        'blue': BLUE,
        'green': GREEN,
        'red': RED,
        'magenta': MAGENTA,
        'yellow': YELLOW,
        'gray': GRAY,
        'white': WHITE,
    }

    def __init__(self, content: str, title: Optional[str] = None,
                 border_style: str = "cyan", width: Optional[int] = None,
                 fit: bool = False):
        self.content = content
        self.title = title
        self.border_color = self.BORDER_COLORS.get(border_style, CYAN)
        self.width = width
        self._fit = fit

    @classmethod
    def fit(cls, content: str, **kwargs):
        """Create a panel that fits its content."""
        return cls(content, fit=True, **kwargs)

    def render(self) -> str:
        """Render the panel as a string."""
        bc = self.border_color

        # Parse markup in content
        content = parse_markup(self.content)
        lines = content.split('\n')

        # Calculate width based on content or terminal
        if self.width:
            total_width = self.width
        elif self._fit:
            content_width = max(len(strip_ansi(line)) for line in lines) if lines else 10
            title_width = len(self.title) + 6 if self.title else 0
            total_width = max(content_width, title_width)
        else:
            total_width = get_terminal_width() - 2

        result = []

        # Top bar with title (left-aligned)
        if self.title:
            title_str = f"[ {self.title} ]"
            left_bar = 6
            right_bar = total_width - len(title_str) - left_bar
            if right_bar < 0:
                right_bar = 0
            top = f"{bc}{'─' * left_bar}{RESET}{BOLD}{title_str}{RESET}{bc}{'─' * right_bar}{RESET}"
        else:
            top = f"{bc}{'─' * total_width}{RESET}"
        result.append(top)

        # Content lines
        for line in lines:
            result.append(line)

        # Bottom bar
        result.append(f"{bc}{'─' * total_width}{RESET}")

        return '\n'.join(result)

    def __str__(self) -> str:
        return self.render()


# =============================================================================
# Markdown Wrapper
# =============================================================================

class Markdown:
    """Wrapper for markdown text to be rendered by Console."""

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return render_markdown(self.text)


# =============================================================================
# Console (print wrapper with markup support)
# =============================================================================

class Console:
    """Simple console output with markup support."""

    def __init__(self, theme: Optional[dict] = None):
        self.theme = {**DEFAULT_THEME, **(theme or {})}

    def print(self, *args, **kwargs):
        """Print with markup support."""
        end = kwargs.pop('end', '\n')
        flush = kwargs.pop('flush', False)

        parts = []
        for arg in args:
            if isinstance(arg, Panel):
                parts.append(arg.render())
            elif isinstance(arg, Markdown):
                parts.append(render_markdown(arg.text))
            elif isinstance(arg, str):
                parts.append(parse_markup(arg))
            else:
                parts.append(str(arg))

        print(' '.join(parts), end=end, flush=flush)

    def panel(self, content: str, title: Optional[str] = None,
              border_style: str = "cyan", **kwargs):
        """Print content in a panel."""
        p = Panel(content, title=title, border_style=border_style, **kwargs)
        print(p.render())

    def markdown(self, text: str):
        """Print markdown-formatted text."""
        print(render_markdown(text))

    def status(self, message: str):
        """Print a dim status message (no newline)."""
        print(f"{DIM}{message}{RESET}", end="", flush=True)

    def clear_line(self):
        """Clear the current line."""
        print("\x1b[1G\x1b[K", end="", flush=True)
