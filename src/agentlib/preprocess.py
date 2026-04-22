"""
Preprocessing fixes for LLM-generated Python code.

Transforms are pure structural functions that attempt to fix common LLM mistakes.
They are composed in preprocess() with a single compile gate at the end: if the
transformed code compiles, accept it; otherwise return the original.
"""

import re


def _compiles(code: str) -> bool:
    """Return True if code compiles as Python."""
    try:
        compile(code, '<repl>', 'exec')
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Individual transforms.  Each takes a string and returns a (possibly modified)
# string.  They must NOT have their own compile guards — the final compile
# check lives in preprocess().
# ---------------------------------------------------------------------------

def _extract_native_function_call_code(code: str) -> str:
    """Convert native function-call XML wrappers into inline Python code.

    Targets malformed outputs like::

        I need to call the tool now.
        <function_calls>
        <invoke name="repl">
        <parameter name="code">decide(x=1)</parameter>
        </invoke>
        </function_calls>

    Replaces each matching XML block with the contents of its
    ``<parameter name="code">...</parameter>`` body, preserving surrounding
    text so later preprocessors can operate on it.  Returns the original
    string unchanged when no matching block is found.
    """
    if '<function_calls' not in code or '<parameter' not in code:
        return code

    pattern = re.compile(
        r"""<function_calls\b[^>]*>.*?<invoke\b[^>]*>.*?
        <parameter\b[^>]*\bname=(["'])code\1[^>]*>(.*?)</parameter>.*?
        </invoke>.*?</function_calls>""",
        flags=re.DOTALL | re.IGNORECASE | re.VERBOSE,
    )

    if not pattern.search(code):
        return code

    return pattern.sub(lambda m: m.group(2).strip(), code)


def _comment_leading_non_code_prefix(code: str) -> str:
    """Comment a leading non-code prefix before the first obvious Python line.

    This is a single structural pass that handles both:
    - ordinary prose prefixes, e.g. ``Two fixes: ...``
    - markdown-ish prefixes, e.g. headings, bullets, numbered items, table rows

    It comments the contiguous non-blank prefix before the first obvious code line.
    If no obvious code line exists, the input is returned unchanged.
    """
    lines = code.split('\n')

    code_line_re = re.compile(
        r'^\s*('
        r'(?:from|import|def|class|if|elif|else|for|while|try|except|finally|with|return|raise|assert|pass|break|continue|yield|del|global|nonlocal)\b'
        r'|(?:@\w[\w\.]*\s*(?:\([^\n]*\))?)'
        r'|[A-Za-z_][A-Za-z0-9_]*\s*\('
        r'|[A-Za-z_][A-Za-z0-9_]*\s*='
        r'|[\]\)\}]\s*(?:,)?\s*$'
        r')'
    )
    markdownish_re = re.compile(
        r'^\s*('
        r'#{1,6}\s+'
        r'|[-*+]\s+'
        r'|\d+[.)]\s+'
        r'|\|.*\|\s*$'
        r'|\*\*[^*\n]+\*\*:?\s*$'
        r'|__[^_\n]+__:?\s*$'
        r')'
    )

    first_code_idx = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if code_line_re.match(line):
            first_code_idx = i
            break

    if first_code_idx is None or first_code_idx == 0:
        return code

    prefix = lines[:first_code_idx]
    if not any(line.strip() for line in prefix):
        return code

    # If the prefix contains markdown-ish structure, definitely comment it.
    # Otherwise still comment it as plain leading prose before code.
    if any(markdownish_re.match(line) for line in prefix if line.strip()) or any(
        not code_line_re.match(line) for line in prefix if line.strip()
    ):
        fixed_prefix = [f'# {line}' if line.strip() else '' for line in prefix]
        return '\n'.join(fixed_prefix + lines[first_code_idx:])

    return code

def _strip_markdown_fences(code: str) -> str:
    """Extract Python from markdown code blocks, converting prose to comments.

    When an LLM wraps its response in markdown fences::

        ```python
        x = 1
        ```
        Now let me continue:
        ```python
        y = 2
        ```

    This strips the fences, keeps the code, and turns prose lines into comments.
    Returns the original string unchanged if fewer than 2 fences are found.
    """
    if '```' not in code:
        return code

    lines = code.split('\n')
    result = []
    in_code_block = False
    fence_count = 0

    for line in lines:
        stripped = line.strip()
        if re.match(r'^```\w*$', stripped):
            in_code_block = not in_code_block
            fence_count += 1
            continue

        if in_code_block:
            result.append(line)
        elif stripped:
            result.append(f'# {stripped}')
        else:
            result.append('')

    if fence_count < 2:
        return code

    return '\n'.join(result)


def _fix_js_comments(code: str) -> str:
    r"""Convert JavaScript-style // comments to Python # comments.

    Only targets positions where // cannot be Python floor division:
      - after a comma:       ``value,  // comment``
      - at the start of a line: ``// comment``

    Floor division always has a left operand (``expr // expr``), so these
    positions are structurally impossible in valid Python.
    """
    # // after comma (with optional whitespace)
    code = re.sub(r',(\s*)//\s*', r',\1# ', code)
    # // at start of line (with optional leading whitespace)
    code = re.sub(r'^(\s*)//\s*', r'\1# ', code, flags=re.MULTILINE)
    return code


def _comment_out_non_python(code: str) -> str:
    """Iteratively comment out lines that are provably not Python.

    Two safe rules (zero false-positive risk):

    1. **Invalid character at line start** — Python reports "invalid character"
       and the offset points to the first non-whitespace position.  Characters
       like ✓ ⚠️ — (em dash) • cannot appear in any valid Python program, so
       when a line *starts* with one it is prose, not code with a typo.

    2. **Markdown horizontal rule** — a line whose stripped content is ``---``.
       Three unary-minus operators with no operand is never a valid Python
       statement.  (Inside a parenthesised expression ``---`` *can* be valid,
       but the parser would not report a SyntaxError on that line in that case.)

    """
    lines = code.split('\n')

    for _ in range(len(lines)):            # bounded iteration
        try:
            compile('\n'.join(lines), '<repl>', 'exec')
            break
        except SyntaxError as e:
            if not e.lineno:
                break
            idx = e.lineno - 1
            if idx < 0 or idx >= len(lines):
                break

            line = lines[idx]
            should_comment = False

            # Rule 1: invalid character at the first non-whitespace position
            if 'invalid character' in e.msg and e.offset is not None:
                first_nonws = len(line) - len(line.lstrip()) + 1   # 1-indexed
                if e.offset == first_nonws:
                    should_comment = True

            # Rule 2: markdown horizontal rule
            if line.strip() == '---':
                should_comment = True

            if not should_comment:
                break

            lines[idx] = f'# {line}'

    return '\n'.join(lines)


def _fix_triple_quote_conflict(code: str) -> str:
    '''Fix triple-quote conflicts where outer """ contains inner """ docstrings.

    When an LLM writes code like::

        print("""
            def foo():
                """docstring"""
        """)

    The inner """ prematurely closes the outer string.  This converts outer
    """ to single quotes when doing so produces valid Python.
    '''
    positions = []
    i = 0
    while True:
        pos = code.find('"""', i)
        if pos == -1:
            break
        positions.append(pos)
        i = pos + 3

    if len(positions) < 4:
        return code

    first_pos = positions[0]
    last_pos = positions[-1]

    return (
        code[:first_pos] + "'''" +
        code[first_pos + 3:last_pos] +
        "'''" + code[last_pos + 3:]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(code: str) -> str:
    """Apply all preprocessing transforms, accepting the result only if it compiles.

    Transforms are composed so that, e.g., fence stripping and JS-comment fixing
    work together even though neither alone produces compilable code.
    """
    if _compiles(code):
        return code

    fixed = code
    fixed = _extract_native_function_call_code(fixed)
    fixed = _strip_markdown_fences(fixed)
    fixed = _fix_js_comments(fixed)
    fixed = _fix_triple_quote_conflict(fixed)
    fixed = _comment_leading_non_code_prefix(fixed)
    fixed = _comment_out_non_python(fixed)

    if fixed != code and _compiles(fixed):
        return fixed

    return code
