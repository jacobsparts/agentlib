"""
Preprocessing fixes for LLM-generated Python code.

Each function follows the same contract:
- Takes code as a string, returns fixed code (or original if no fix needed).
- Only intervenes when the code doesn't compile as-is.
- Validates the fix compiles before accepting it.
"""

import re


def fix_triple_quote_conflict(code: str) -> str:
    '''
    Fix triple-quote conflicts where outer """ contains inner """ docstrings.

    When an LLM writes code like:
        print("""
            def foo():
                """docstring"""
        """)

    The inner """ prematurely closes the outer string. This function detects
    and fixes such cases by converting outer """ to single quotes.

    Returns the fixed code, or original if no fix needed/possible.
    '''
    try:
        compile(code, '<repl>', 'exec')
        return code
    except SyntaxError:
        pass

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

    fixed = (
        code[:first_pos] + "'''" +
        code[first_pos + 3:last_pos] +
        "'''" + code[last_pos + 3:]
    )

    try:
        compile(fixed, '<repl>', 'exec')
        return fixed
    except SyntaxError:
        pass

    return code


def fix_markdown_code_blocks(code: str) -> str:
    """Extract Python from markdown code blocks, optionally mixed with prose.

    When an LLM wraps its response in markdown fences:

        ```python
        x = 1
        ```
        Now let me continue:
        ```python
        y = 2
        ```

    This extracts the code blocks and converts prose lines to comments.

    Returns the fixed code, or original if no fix needed/possible.
    """
    try:
        compile(code, '<repl>', 'exec')
        return code
    except SyntaxError:
        pass

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

    fixed = '\n'.join(result)

    try:
        compile(fixed, '<repl>', 'exec')
        return fixed
    except SyntaxError:
        pass

    return code


def preprocess(code: str) -> str:
    """Apply all preprocessing fixes in sequence."""
    code = fix_markdown_code_blocks(code)
    code = fix_triple_quote_conflict(code)
    return code
