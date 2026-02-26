"""Tests for agentlib.preprocess — LLM code preprocessing fixes."""

from agentlib.preprocess import (
    _strip_markdown_fences,
    _fix_js_comments,
    _fix_triple_quote_conflict,
    _comment_out_non_python,
    preprocess,
)


def compiles(code):
    try:
        compile(code, '<test>', 'exec')
        return True
    except SyntaxError:
        return False


# === _strip_markdown_fences ===

class TestStripMarkdownFences:
    def test_valid_code_untouched(self):
        code = "x = 1\nprint(x)"
        assert _strip_markdown_fences(code) == code

    def test_single_fence(self):
        code = '```python\nx = 1\n```'
        assert _strip_markdown_fences(code) == 'x = 1'

    def test_fence_without_language(self):
        code = '```\nx = 1\n```'
        assert _strip_markdown_fences(code) == 'x = 1'

    def test_multi_block_with_prose(self):
        code = '```python\nx = 1\n```\nSome prose\n```python\ny = 2\n```'
        r = _strip_markdown_fences(code)
        assert 'x = 1' in r
        assert 'y = 2' in r
        assert '# Some prose' in r
        assert '```' not in r
        assert compiles(r)

    def test_prose_before_first_block(self):
        code = 'Let me analyze:\n```python\nx = 1\n```'
        r = _strip_markdown_fences(code)
        assert '# Let me analyze:' in r
        assert 'x = 1' in r
        assert compiles(r)

    def test_prose_after_last_block(self):
        code = '```python\nx = 1\n```\nDone!'
        r = _strip_markdown_fences(code)
        assert 'x = 1' in r
        assert '# Done!' in r
        assert compiles(r)

    def test_indentation_preserved(self):
        code = '```python\ndef foo():\n    return 42\n```'
        r = _strip_markdown_fences(code)
        assert '    return 42' in r
        assert compiles(r)

    def test_blank_lines_preserved(self):
        code = '```python\nx = 1\n```\n\n```python\ny = 2\n```'
        r = _strip_markdown_fences(code)
        assert 'x = 1' in r
        assert 'y = 2' in r
        assert compiles(r)

    def test_backticks_in_string_valid_code_untouched(self):
        code = 'msg = "use ``` for code"\nprint(msg)'
        assert _strip_markdown_fences(code) == code

    def test_non_markdown_syntax_error_unchanged(self):
        """Code with ``` in a string that has an unrelated syntax error."""
        code = 'msg = "use ``` for code"\nx = 1 +'
        assert _strip_markdown_fences(code) == code

    def test_backticks_in_string_within_block(self):
        code = '```python\nmsg = "use ``` for code"\nprint(msg)\n```'
        r = _strip_markdown_fences(code)
        assert 'msg = "use ``` for code"' in r
        assert compiles(r)

    def test_haiku_eval_pattern(self):
        """Full pattern from Haiku eval: prose + markdown headers + multiple blocks."""
        code = (
            '# Aggregation Agent\n'
            '\n'
            'Let me begin.\n'
            '\n'
            '```python\n'
            'x = 42\n'
            'print(x)\n'
            '```\n'
            '\n'
            'Now the decision:\n'
            '\n'
            '```python\n'
            'y = x + 1\n'
            'print(y)\n'
            '```\n'
            '\n'
            '## Summary\n'
            '\n'
            'I **cannot approve** this.'
        )
        r = _strip_markdown_fences(code)
        assert 'x = 42' in r
        assert 'y = x + 1' in r
        assert '# Let me begin.' in r
        assert '```' not in r
        assert compiles(r)

    def test_empty_code_block(self):
        code = '```python\n```'
        r = _strip_markdown_fences(code)
        assert '```' not in r
        assert compiles(r)

    def test_single_fence_only_returns_original(self):
        """A lone ``` without a matching pair should not trigger extraction."""
        code = 'x = 1\n```'
        assert _strip_markdown_fences(code) == code


# === _fix_js_comments ===

class TestFixJsComments:
    def test_no_comments_untouched(self):
        code = 'x = 1\nprint(x)'
        assert _fix_js_comments(code) == code

    def test_floor_division_untouched(self):
        code = 'x = 10 // 3'
        assert _fix_js_comments(code) == code

    def test_floor_division_in_dict_untouched(self):
        code = '{"a": 10 // 3, "b": 20 // 7}'
        assert _fix_js_comments(code) == code

    def test_after_comma(self):
        """// after comma is always a JS comment (no left operand for floor div)."""
        code = 'x = {"a": 1,  // comment\n"b": 2}'
        r = _fix_js_comments(code)
        assert '//' not in r
        assert '# comment' in r
        assert compiles(r)

    def test_at_line_start(self):
        """// at start of line is always a JS comment."""
        code = '// this is a comment\nx = 1'
        r = _fix_js_comments(code)
        assert '//' not in r
        assert '# this is a comment' in r
        assert compiles(r)

    def test_indented_line_start(self):
        """// at start of line with leading whitespace."""
        code = 'if True:\n    // comment\n    x = 1'
        r = _fix_js_comments(code)
        assert '//' not in r
        assert '    # comment' in r
        assert compiles(r)

    def test_mixed_legit_and_js(self):
        """Floor division on one line, JS comment on another."""
        code = 'x = 10 // 3\ny = {"a": 1,  // note\n"b": 2}'
        r = _fix_js_comments(code)
        assert '10 // 3' in r  # floor div preserved
        assert '# note' in r   # JS comment converted
        assert compiles(r)


# === _comment_out_non_python ===

class TestCommentOutNonPython:
    def test_valid_code_untouched(self):
        code = 'x = 1\nprint(x)'
        assert _comment_out_non_python(code) == code

    def test_markdown_hr(self):
        """--- on its own line is commented out."""
        code = 'x = 1\n---\ny = 2'
        r = _comment_out_non_python(code)
        assert '# ---' in r
        assert compiles(r)

    def test_markdown_hr_with_whitespace(self):
        code = 'x = 1\n  ---  \ny = 2'
        r = _comment_out_non_python(code)
        assert compiles(r)

    def test_invalid_char_at_line_start(self):
        """\u2713 at start of line is commented out."""
        code = 'x = 1\n\u2713 check passed\ny = 2'
        r = _comment_out_non_python(code)
        assert '# \u2713' in r
        assert compiles(r)

    def test_invalid_char_mid_line_not_touched(self):
        """Em dash mid-line is NOT commented out (could be code with a typo)."""
        code = 'profit = revenue \u2014 cost'
        r = _comment_out_non_python(code)
        assert r == code  # unchanged

    def test_warning_emoji_at_start(self):
        code = 'x = 1\n\u26a0\ufe0f  Listing 53921: price too high\ny = 2'
        r = _comment_out_non_python(code)
        assert compiles(r)

    def test_bullet_at_start(self):
        code = 'x = 1\n\u2022 Item one\n\u2022 Item two\ny = 2'
        r = _comment_out_non_python(code)
        assert compiles(r)

    def test_multiple_non_python_lines(self):
        """Multiple offending lines are commented out iteratively."""
        code = '\u2713 check 1\n---\n\u2713 check 2\nx = 1'
        r = _comment_out_non_python(code)
        assert compiles(r)
        assert 'x = 1' in r

    def test_dashes_inside_parens_not_touched(self):
        """--- inside an expression is valid Python (1 --- 2 = -1), not our problem."""
        code = 'x = (1\n---\n2)'  # valid: 1 - (-(- 2))
        # This compiles fine, so _comment_out_non_python should not touch it
        assert compiles(code)
        assert _comment_out_non_python(code) == code

    def test_indented_invalid_char(self):
        """Indented line starting with invalid char inside a block."""
        code = 'if True:\n    \u2713 done\n    x = 1'
        r = _comment_out_non_python(code)
        assert compiles(r)


# === _fix_triple_quote_conflict ===

class TestFixTripleQuoteConflict:
    def test_valid_code_untouched(self):
        code = 'x = 1'
        assert _fix_triple_quote_conflict(code) == code

    def test_nested_triple_quotes_fixed(self):
        code = 'print("""\n    def foo():\n        """docstring"""\n""")'
        r = _fix_triple_quote_conflict(code)
        assert compiles(r)

    def test_normal_triple_quote_untouched(self):
        code = '"""\nhello\n"""'
        assert _fix_triple_quote_conflict(code) == code

    def test_unfixable_returns_original(self):
        code = '"""a"""b"""c"""d"""'
        r = _fix_triple_quote_conflict(code)
        # Should return original or a compiling fix
        assert r == code or compiles(r)


# === preprocess (composed pipeline) ===

class TestPreprocess:
    def test_valid_code_untouched(self):
        code = 'x = 1\nprint(x)'
        assert preprocess(code) == code

    def test_markdown_fixed(self):
        code = '```python\nx = 1\n```'
        assert preprocess(code) == 'x = 1'

    def test_triple_quote_fixed(self):
        code = 'print("""\n    """inner"""\n""")'
        assert compiles(preprocess(code))

    def test_fences_plus_js_comments_composed(self):
        """Fence stripping + JS comment fixing must compose.

        Neither transform alone produces compilable code:
        - Fence stripping extracts code with // comments → doesn't compile
        - JS comment fixing on fenced code → // buried inside fences
        Together they work.
        """
        code = (
            '## Step 1\n'
            '\n'
            '```\n'
            'analyze({\n'
            '    "key": "value",  // explanation\n'
            '    "n": null,       // note\n'
            '})\n'
            '```\n'
        )
        r = preprocess(code)
        assert '```' not in r
        assert '//' not in r
        assert compiles(r)

    def test_fences_plus_js_comments_real_world(self):
        """Real-world pattern from DCMOD-M listing agent trace.

        The agent wraps a JS-style object literal in markdown fences with //
        comments.  Markdown headings (##) become Python comments after stripping.
        """
        code = (
            '## Step 1: analyze()\n'
            '\n'
            '```\n'
            'analyze({\n'
            '  listings: [\n'
            '    {\n'
            '      listing_id: "10201",\n'
            '      velocity_7d: 0.00,\n'
            '      acceleration_ratio: null,          // 0/0 — dormant\n'
            '      velocity_trend: "DECELERATING",    // zero across all periods\n'
            '      demand_share: 0.0,                 // 0 of 30 units_sku 30d\n'
            '    }\n'
            '  ]\n'
            '})\n'
            '```\n'
        )
        r = preprocess(code)
        assert '```' not in r
        assert '//' not in r
        assert 'analyze({' in r
        assert compiles(r)

    def test_chain_markdown_then_triple_quote(self):
        """Markdown extraction + triple-quote fix compose."""
        code = '```python\nprint("""\n    """inner"""\n""")\n```'
        r = preprocess(code)
        assert '```' not in r
        assert compiles(r)

    def test_floor_division_not_corrupted(self):
        """Legit floor division must survive preprocessing."""
        code = 'x = 10 // 3\nprint(x)'
        assert preprocess(code) == code

    def test_floor_division_in_fenced_code(self):
        """Floor division inside fences is preserved (not a JS comment)."""
        code = '```python\nx = 10 // 3\nprint(x)\n```'
        r = preprocess(code)
        assert '10 // 3' in r
        assert compiles(r)

    def test_fences_with_simulated_output(self):
        """Fenced code blocks containing simulated REPL output with Unicode.

        The agent places expected output (with \u2713/\u26a0\ufe0f) inside ```python fences.
        After fence stripping those become code lines; _comment_out_non_python
        catches ones starting with invalid characters.
        """
        code = (
            '```python\n'
            'check_fba_fbm()\n'
            '```\n'
            '```python\n'
            '\u2713 No FBM/FBA pairs found for comparison.\n'
            '```\n'
            '```python\n'
            'check_pack_pricing()\n'
            '```\n'
            '```python\n'
            '\u2713 No multi-pack relationships to check.\n'
            '```\n'
        )
        r = preprocess(code)
        assert '```' not in r
        assert 'check_fba_fbm()' in r
        assert 'check_pack_pricing()' in r
        assert compiles(r)

    def test_prose_with_hr_separator(self):
        """Prose line + --- separator + valid code, no fences.

        The --- is caught; the prose line is only caught if it starts with
        an invalid character.
        """
        code = '\u2713 All checks passed\n\n---\n\nanalyze(x=1)'
        r = preprocess(code)
        assert compiles(r)
