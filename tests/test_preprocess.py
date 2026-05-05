"""Tests for agentlib.preprocess — LLM code preprocessing fixes."""

import ast
import json
from pathlib import Path

from agentlib.agents.code_agent import CodeAgent
from agentlib import SandboxMixin
from agentlib.preprocess import (
    _close_unclosed_string,
    _comment_leading_non_code_prefix,
    _extract_native_function_call_code,
    _strip_markdown_fences,
    _strip_leading_prompts,
    _fix_js_comments,
    _fix_triple_quote_conflict,
    _comment_out_non_python,
    preprocess,
)
from agentlib.client import _gemini_transform_schema
from agentlib.client import _gemini_schema_has_unsupported_fieldtypes


def compiles(code):
    try:
        compile(code, '<test>', 'exec')
        return True
    except SyntaxError:
        return False


def same_ast(a, b):
    return ast.dump(ast.parse(a), include_attributes=False) == ast.dump(ast.parse(b), include_attributes=False)


class TestExtractNativeFunctionCallCode:
    def test_valid_code_untouched(self):
        code = "decide(x=1)"
        assert _extract_native_function_call_code(code) == code

    def test_literal_function_call_xml(self):
        code = (
            '<function_calls>\n'
            '<invoke name="repl">\n'
            '<parameter name="code">decide(x=1)</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        assert _extract_native_function_call_code(code) == 'decide(x=1)'

    def test_narration_before_xml(self):
        code = (
            'I need to call decide() now.\n\n'
            '<function_calls>\n'
            '<invoke name="repl">\n'
            '<parameter name="code">decide(x=1)</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        assert _extract_native_function_call_code(code) == (
            'I need to call decide() now.\n\n'
            'decide(x=1)'
        )

    def test_multiline_code_body_preserved(self):
        code = (
            '<function_calls>\n'
            '<invoke name="python">\n'
            '<parameter name="code">decide(\n'
            '    x=1,\n'
            '    y=2,\n'
            ')</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        r = _extract_native_function_call_code(code)
        assert 'decide(' in r
        assert 'x=1' in r
        assert compiles(r)

    def test_single_quotes_supported(self):
        code = (
            "<function_calls><invoke name='repl'><parameter name='code'>decide(x=1)</parameter></invoke></function_calls>"
        )
        assert _extract_native_function_call_code(code) == 'decide(x=1)'

    def test_multiple_xml_blocks_are_replaced_inline(self):
        code = (
            'before\n'
            '<function_calls><invoke name="repl"><parameter name="code">x = 1</parameter></invoke></function_calls>\n'
            'middle\n'
            '<function_calls><invoke name="repl"><parameter name="code">print(x)</parameter></invoke></function_calls>\n'
            'after'
        )
        assert _extract_native_function_call_code(code) == (
            'before\n'
            'x = 1\n'
            'middle\n'
            'print(x)\n'
            'after'
        )


class TestCommentLeadingNonCodePrefix:
    def test_valid_code_untouched(self):
        code = "decide(x=1)"
        assert _comment_leading_non_code_prefix(code) == code

    def test_comments_single_leading_prose_line(self):
        code = "Two fixes: use positional args only.\n\ndecide(x=1)"
        assert _comment_leading_non_code_prefix(code) == (
            "# Two fixes: use positional args only.\n\ndecide(x=1)"
        )

    def test_comments_multiple_leading_prose_lines(self):
        code = (
            "I need to call the tool.\n"
            "Let me do that now.\n\n"
            "analyze(x=1)\n"
            "decide(x=1)"
        )
        assert _comment_leading_non_code_prefix(code) == (
            "# I need to call the tool.\n"
            "# Let me do that now.\n\n"
            "analyze(x=1)\n"
            "decide(x=1)"
        )

    def test_does_not_accept_pure_prose(self):
        code = "I need to call this through the actual REPL tool, not output it as text."
        assert _comment_leading_non_code_prefix(code) == code

    def test_does_not_parse_to_decide_where_code_starts(self):
        code = (
            "Two fixes: use positional args only.\n\n"
            "decide(x=1)"
        )
        assert _comment_leading_non_code_prefix(code) == (
            "# Two fixes: use positional args only.\n\n"
            "decide(x=1)"
        )

    def test_comments_leading_markdown_heading_before_code(self):
        code = "## Analysis\n\ndecide(x=1)"
        assert _comment_leading_non_code_prefix(code) == (
            "# ## Analysis\n\ndecide(x=1)"
        )

    def test_comments_leading_table_before_code(self):
        code = "| Col | Value |\n|---|---|\n\nanalyze(x=1)"
        r = _comment_leading_non_code_prefix(code)
        assert "# | Col | Value |" in r
        assert "# |---|---|" in r
        assert compiles(r)

    def test_comments_leading_bold_heading_before_code(self):
        code = "**Plan:**\n\ndecide(x=1)"
        r = _comment_leading_non_code_prefix(code)
        assert r.startswith("# **Plan:**")
        assert compiles(r)


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


# === _strip_leading_prompts ===

class TestStripLeadingPrompts:
    def test_no_prompts_untouched(self):
        code = 'x = 1\nprint(x)'
        assert _strip_leading_prompts(code) == code

    def test_triple_prompt_single_line(self):
        code = '>>> x = 1'
        assert _strip_leading_prompts(code) == 'x = 1'

    def test_triple_prompt_multiple_lines(self):
        code = '>>> x = 1\n>>> print(x)'
        r = _strip_leading_prompts(code)
        assert r == 'x = 1\nprint(x)'
        assert compiles(r)

    def test_triple_prompt_with_continuation(self):
        code = '>>> for i in range(3):\n...     print(i)'
        r = _strip_leading_prompts(code)
        assert r == 'for i in range(3):\n    print(i)'
        assert compiles(r)

    def test_triple_prompt_bare_line(self):
        """A bare >>> with nothing after it."""
        code = '>>> x = 1\n>>>\n>>> y = 2'
        r = _strip_leading_prompts(code)
        assert 'x = 1' in r
        assert 'y = 2' in r

    def test_single_gt_prompt(self):
        code = '>xpath = ""'
        r = _strip_leading_prompts(code)
        assert r == 'xpath = ""'
        assert compiles(r)

    def test_single_gt_with_space(self):
        code = '> xpath = ""'
        r = _strip_leading_prompts(code)
        assert r == 'xpath = ""'
        assert compiles(r)

    def test_single_gt_multiple_lines(self):
        code = '> x = 1\n> print(x)'
        r = _strip_leading_prompts(code)
        assert r == 'x = 1\nprint(x)'
        assert compiles(r)

    def test_gt_in_valid_code_untouched(self):
        """Greater-than in valid Python should not be stripped."""
        code = 'if x > 0:\n    print(x)'
        assert _strip_leading_prompts(code) == code

    def test_triple_prompt_preserves_indentation(self):
        code = '>>> if True:\n...     x = 1\n...     print(x)'
        r = _strip_leading_prompts(code)
        assert compiles(r)
        assert '    x = 1' in r

    def test_mixed_prompt_and_non_prompt_lines(self):
        code = '>>> x = 1\nsome output\n>>> y = 2'
        r = _strip_leading_prompts(code)
        assert 'x = 1' in r
        assert 'some output' in r
        assert 'y = 2' in r


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


# === _close_unclosed_string ===

class TestCloseUnclosedString:
    def test_valid_code_untouched(self):
        code = 'x = 1'
        assert _close_unclosed_string(code) == code

    def test_matched_triple_quotes_untouched(self):
        code = 'x = """hello"""\nprint(x)'
        assert _close_unclosed_string(code) == code

    def test_odd_double_triple_quote_closed(self):
        code = 'emit("""text with """ and """ more'
        r = _close_unclosed_string(code)
        assert r.count('"""') % 2 == 0
        assert r.endswith('""")')

    def test_odd_single_triple_quote_closed(self):
        code = "emit('''text with ''' and ''' more"
        r = _close_unclosed_string(code)
        assert r.count("'''") % 2 == 0
        assert r.endswith("''')")

    def test_already_even_not_modified(self):
        code = 'print("""a """ b """ c """)'
        assert _close_unclosed_string(code) == code


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

    def test_native_function_call_xml_fixed(self):
        code = (
            '<function_calls>\n'
            '<invoke name="repl">\n'
            '<parameter name="code">decide(\n'
            '    baselines={1: {"standard": 9.99}},\n'
            '    reasoning="ok",\n'
            '    tier_strategy="HOLD",\n'
            ')</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        r = preprocess(code)
        assert '<function_calls' not in r
        assert 'decide(' in r
        assert compiles(r)

    def test_native_function_call_with_narration_falls_back_to_original_if_still_invalid(self):
        code = (
            'I need to call decide() using the tool.\n\n'
            '<function_calls>\n'
            '<invoke name="repl">\n'
            '<parameter name="code">decide(x=1)</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        r = preprocess(code)
        assert '# I need to call decide() using the tool.' in r
        assert 'decide(x=1)' in r
        assert compiles(r)

    def test_markdown_prefix_then_valid_code_fixed(self):
        code = (
            "## Summary\n"
            "- first note\n"
            "| col | value |\n"
            "|---|---|\n\n"
            "decide(x=1)\n"
        )
        r = preprocess(code)
        assert "# ## Summary" in r
        assert "# - first note" in r
        assert "# | col | value |" in r
        assert "decide(x=1)" in r
        assert compiles(r)

    def test_prose_preamble_then_valid_code_fixed(self):
        code = (
            'Two fixes: eBay role must be PRIMARY_MOVER, and decide() takes positional args only — trying dict as positional argument.\n\n'
            'analyze(x=1)\n'
            'decide(x=1)'
        )
        r = preprocess(code)
        assert r.startswith('# Two fixes:')
        assert 'analyze(x=1)' in r
        assert 'decide(x=1)' in r
        assert compiles(r)

    def test_pure_prose_not_accepted_as_comment_only_program(self):
        code = 'I need to call this through the actual REPL tool, not output it as text. Let me do that now.'
        assert preprocess(code) == code

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

    def test_preprocess_leaves_preview_read_pattern_alone(self):
        code = 'preview(read("CLAUDE.md"))'
        r = preprocess(code)
        assert same_ast(r, 'preview(read("CLAUDE.md"))')
        assert compiles(r)

    def test_preprocess_leaves_namedexpr_preview_read_pattern_alone(self):
        code = 'preview(somevar := read("CLAUDE.md", offset=1, limit=80))'
        r = preprocess(code)
        assert same_ast(r, 'preview(somevar := read("CLAUDE.md", offset=1, limit=80))')
        assert compiles(r)

    def test_preprocess_leaves_assigned_read_pattern_alone(self):
        code = 'content = read("CLAUDE.md")'
        r = preprocess(code)
        assert same_ast(r, 'content = read("CLAUDE.md")')
        assert compiles(r)

    def test_unclosed_triple_quote_in_emit(self):
        code = 'emit("""Here is how triple quotes work'
        r = preprocess(code)
        assert compiles(r)

    def test_unclosed_with_multiple_inner_triple_quotes(self):
        code = 'emit("""Use """ for docstrings and """ for multiline'
        r = preprocess(code)
        assert compiles(r)

    def test_unclosed_closing_absorbs_paren(self):
        code = 'emit("""text with """ inside, release=True)'
        r = preprocess(code)
        assert isinstance(r, str)

    def test_unclosed_single_triple_quote_in_emit(self):
        code = "emit('''Here is how triple quotes work"
        r = preprocess(code)
        assert compiles(r)


class TestCodeAgentPreprocessCode:
    def test_bare_read_gets_previewed(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('read("file.py")')
        assert same_ast(r, 'view("file.py")')

    def test_print_read_becomes_view(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('print(read("file.py"))')
        assert same_ast(r, 'view("file.py")')

    def test_print_path_read_text_becomes_view(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('print(Path("README.md").read_text())')
        assert same_ast(r, 'view("README.md")')

    def test_print_sliced_path_read_text_gets_warning(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('print(Path("README.md").read_text()[:4000])')
        assert 'Direct file reads bypass code_agent context tools' in r
        assert 'print(Path("README.md").read_text()[:4000])' in r
        assert compiles(r)

    def test_bare_path_read_text_becomes_view(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('Path("README.md").read_text()')
        assert same_ast(r, 'view("README.md")')

    def test_assignment_path_read_text_becomes_read(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('content = Path("README.md").read_text()')
        assert same_ast(r, 'content = read("README.md")')

    def test_assignment_open_read_becomes_read(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('content = open("README.md").read()')
        assert same_ast(r, 'content = read("README.md")')

    def test_bare_path_open_read_becomes_view(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('Path("README.md").open().read()')
        assert same_ast(r, 'view("README.md")')

    def test_ambiguous_direct_file_read_gets_warning(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('content = Path(filename).read_text()')
        assert 'Direct file reads bypass code_agent context tools' in r
        assert 'content = Path(filename).read_text()' in r
        assert compiles(r)

    def test_preview_read_left_alone(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('preview(read("file.py"))')
        assert same_ast(r, 'preview(read("file.py"))')

    def test_print_assigned_preview_uri_read_becomes_view(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code(
            'fail = read("session://preview/abc123")\n'
            'print(fail)'
        )
        assert same_ast(
            r,
            'fail = read("session://preview/abc123")\n'
            'view("session://preview/abc123")',
        )

    def test_bare_bash_gets_assigned_and_previewed(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('bash("echo hi")')
        assert same_ast(r, 'preview(_bash1 := bash("echo hi"))')

    def test_bare_background_bash_gets_assigned_only(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('bash("sleep 10", bg=True)')
        assert same_ast(r, '_bash1 = bash("sleep 10", bg=True)')

    def test_print_bash_gets_assigned_and_previewed(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('print(bash("echo hi"))')
        assert same_ast(r, 'preview(_bash1 := bash("echo hi"))')

    def test_assigned_bash_left_alone(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('proc = bash("sleep 10", bg=True)')
        assert same_ast(r, 'proc = bash("sleep 10", bg=True)')

    def test_preview_accepts_non_string_values(self, monkeypatch):
        sent = []
        import agentlib.agents.code_agent as code_agent
        monkeypatch.setattr(code_agent, "_send_output", lambda msg_type, chunk: sent.append((msg_type, chunk)), raising=False)

        agent = CodeAgent()
        agent.preview({"pid": 123})

        assert sent == [("preview", "{'pid': 123}\n")]

    def test_view_assignment_rejected(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('content = view("file.py")')
        assert same_ast(
            r,
            'raise ValueError("view() is a display tool, not a value. Use read() for file contents as text.")',
        )

    def test_print_view_rejected(self):
        CodeAgent._preview_counter = 0
        agent = CodeAgent()
        r = agent.preprocess_code('print(view("file.py"))')
        assert same_ast(
            r,
            'raise ValueError("view() is a display tool, not a value. Use read() for file contents as text.")',
        )

    def test_attached_file_matches_absolute_written_path(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("before\n")
        rel = Path.cwd().joinpath(target).relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else str(target)
        name = str(rel)

        agent = CodeAgent()
        agent._ensure_setup()
        agent.conversation.usermsg(
            f"[Attachment: {name}]",
            _attachments={name: "    1→before\n"},
        )

        target.write_text("after\n")
        output = agent.build_output_for_llm([("file_written", str(target.resolve()) + "\n")])

        assert f"[Attachment: {name}]" in output
        rendered = agent.conversation._messages()
        user_messages = [msg for msg in rendered if msg.get("role") == "user"]
        assert "before\n" not in json.dumps(user_messages)
        assert agent._read_attachments[name] == "    1→after\n    2→"


    def test_line_patch_allows_consecutive_patches_after_success(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("one\ntwo\nthree\n")
        name = str(target)

        agent = CodeAgent()
        agent._ensure_setup()
        agent.complete = False
        repl = agent._get_tool_repl()
        try:
            output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
                repl,
                f"view({name!r})",
            )
            assert pure_syntax_error is False
            agent.usermsg(agent.build_output_for_llm(output_chunks))

            output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
                repl,
                f"""line_patch({name!r}, \"""replace 2:2
TWO\""")
line_patch({name!r}, \"""replace 3:3
THREE\""")""",
            )

            assert pure_syntax_error is False
            assert "stale after a previous edit" not in output
            assert target.read_text() == "one\nTWO\nTHREE\n"
        finally:
            repl.close()
    def test_unview_does_not_require_repl_send_output(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("content\n")
        name = str(target)

        agent = CodeAgent()
        agent._ensure_setup()
        agent.conversation.usermsg(
            f"[Attachment: {name}]",
            _attachments={name: "    1→content\n    2→"},
        )

        result = agent.unview(name)
        output = agent.build_output_for_llm([])

        assert result == f"Removed from future context: {name}"
        assert output == ""
        assert name not in agent.list_attachments()

    def test_session_uri_attachments_excluded_from_default_attachment_list(self):
        agent = CodeAgent()
        agent._ensure_setup()
        file_name = "file.py"
        preview_name = "session://preview/abc123"
        other_session_name = "session://blob/def456"

        agent.conversation.usermsg(
            f"[Attachment: {file_name}]\n[Attachment: {preview_name}]\n[Attachment: {other_session_name}]",
            _attachments={
                file_name: "    1→file\n",
                preview_name: "    1→blob\n",
                other_session_name: "    1→other\n",
            },
        )

        assert file_name in agent.list_attachments()
        assert preview_name not in agent.list_attachments()
        assert other_session_name not in agent.list_attachments()
        assert preview_name in agent.list_attachments(include_session_blobs=True)
        assert other_session_name in agent.list_attachments(include_session_blobs=True)

    def test_usermsg_sets_empty_ephemeral_when_no_files_in_context(self):
        agent = CodeAgent()
        agent._ensure_setup()

        agent.usermsg("question")

        assert agent.ephemeral == ""

    def test_usermsg_sets_file_context_ephemeral_for_viewed_files(self):
        agent = CodeAgent()
        agent._ensure_setup()
        agent.conversation.usermsg(
            "[Attachment: file.py]",
            _attachments={"file.py": "    1→content\\n"},
        )

        agent.usermsg("next question")

        assert agent.ephemeral == (
            "Files currently in context:\n"
            "- file.py\n"
            "\n"
            "Remove files that are irrelevant to recent conversation state with unview(path)."
        )
        assert "attachment" not in agent.ephemeral.lower()

    def test_usermsg_file_context_ephemeral_includes_new_read_context(self):
        agent = CodeAgent()
        agent._ensure_setup()
        agent._read_attachments = {"new.py": "    1→content\\n"}

        agent.usermsg("[Attachment: new.py]\\n\\nnext question")

        assert "new.py" in agent.ephemeral

    def test_usermsg_file_context_ephemeral_excludes_session_blobs(self):
        agent = CodeAgent()
        agent._ensure_setup()
        agent._read_attachments = {"session://preview/abc123": "    1→blob\\n"}

        agent.usermsg("[Attachment: session://preview/abc123]\\n\\nnext question")

        assert agent.ephemeral == ""

    def test_unview_session_uri_attachment(self):
        agent = CodeAgent()
        agent._ensure_setup()
        preview_name = "session://preview/abc123"

        agent.conversation.usermsg(
            f"[Attachment: {preview_name}]",
            _attachments={preview_name: "    1→blob\n"},
        )

        result = agent.unview(preview_name)

        assert result == f"Removed from future context: {preview_name}"
        assert preview_name not in agent.list_attachments(include_session_blobs=True)

    def test_unview_pending_prevents_same_turn_reattach(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("content\n")
        name = str(target)

        agent = CodeAgent()
        agent._ensure_setup()
        agent.conversation.usermsg(
            f"[Attachment: {name}]",
            _attachments={name: "    1→old\n    2→"},
        )

        agent.unview(name)
        output = agent.build_output_for_llm([
            ("read_attach", name + "\n"),
            ("read", "    1→content\n    2→\n"),
        ])

        assert output == "    1→content\n    2→\n"
        assert agent._read_attachments == {}

    def test_sandboxed_unview_via_repl_does_not_raise_send_output(self, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("content\n")
        name = str(target)

        class SandboxedCodeAgent(SandboxMixin, CodeAgent):
            sandbox_target = str(tmp_path)

        agent = SandboxedCodeAgent()
        try:
            agent._ensure_setup()
            agent.complete = False
            agent.conversation.usermsg(
                f"[Attachment: {name}]",
                _attachments={name: "    1→content\n    2→"},
            )
            repl = agent._get_tool_repl()
            output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
                repl,
                f"unview({name!r})",
            )

            assert pure_syntax_error is False
            assert "_send_output" not in output
            assert any("Removed from future context" in chunk for _, chunk in output_chunks)
            assert name not in agent.list_attachments()
        finally:
            agent._cleanup()


class TestGeminiSchemaTransform:
    def test_removes_additional_properties_from_dict_field(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "title": "Value",
                }
            },
            "required": ["value"],
            "title": "Tool",
        }
        out = _gemini_transform_schema(schema)
        assert out == {
            "type": "object",
            "properties": {
                "value": {
                    "type": "object",
                }
            },
            "required": ["value"],
        }

    def test_flattens_optional_anyof_null(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Query",
                    "description": "Optional query",
                }
            },
            "title": "Tool",
        }
        out = _gemini_transform_schema(schema)
        assert out == {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional query",
                }
            },
        }

    def test_inlines_refs_and_strips_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Nested"}
            },
            "required": ["item"],
            "$defs": {
                "Nested": {
                    "type": "object",
                    "title": "Nested",
                    "properties": {
                        "name": {"type": "string", "title": "Name"}
                    },
                    "required": ["name"],
                }
            },
        }
        out = _gemini_transform_schema(schema)
        assert out == {
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"],
                }
            },
            "required": ["item"],
        }


class TestGeminiSchemaFallbackDetection:
    def test_detects_dict_map_via_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "baselines": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                }
            },
        }
        assert _gemini_schema_has_unsupported_fieldtypes(schema) is True

    def test_detects_underspecified_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "listings": {
                    "type": "array",
                    "items": {},
                }
            },
        }
        assert _gemini_schema_has_unsupported_fieldtypes(schema) is True

    def test_detects_array_of_bare_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "listings": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
        }
        assert _gemini_schema_has_unsupported_fieldtypes(schema) is True

    def test_allows_explicit_array_of_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "listings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "listing_id": {"type": "integer"},
                            "action": {"type": "string"},
                        },
                        "required": ["listing_id", "action"],
                    },
                }
            },
        }
        assert _gemini_schema_has_unsupported_fieldtypes(schema) is False
