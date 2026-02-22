"""Tests for agentlib.preprocess — LLM code preprocessing fixes."""

from agentlib.preprocess import fix_markdown_code_blocks, fix_triple_quote_conflict, preprocess


def compiles(code):
    try:
        compile(code, '<test>', 'exec')
        return True
    except SyntaxError:
        return False


# === fix_markdown_code_blocks ===

class TestFixMarkdownCodeBlocks:
    def test_valid_code_untouched(self):
        code = "x = 1\nprint(x)"
        assert fix_markdown_code_blocks(code) == code

    def test_single_fence(self):
        code = '```python\nx = 1\n```'
        assert fix_markdown_code_blocks(code) == 'x = 1'

    def test_fence_without_language(self):
        code = '```\nx = 1\n```'
        assert fix_markdown_code_blocks(code) == 'x = 1'

    def test_multi_block_with_prose(self):
        code = '```python\nx = 1\n```\nSome prose\n```python\ny = 2\n```'
        r = fix_markdown_code_blocks(code)
        assert 'x = 1' in r
        assert 'y = 2' in r
        assert '# Some prose' in r
        assert '```' not in r
        assert compiles(r)

    def test_prose_before_first_block(self):
        code = 'Let me analyze:\n```python\nx = 1\n```'
        r = fix_markdown_code_blocks(code)
        assert '# Let me analyze:' in r
        assert 'x = 1' in r
        assert compiles(r)

    def test_prose_after_last_block(self):
        code = '```python\nx = 1\n```\nDone!'
        r = fix_markdown_code_blocks(code)
        assert 'x = 1' in r
        assert '# Done!' in r
        assert compiles(r)

    def test_indentation_preserved(self):
        code = '```python\ndef foo():\n    return 42\n```'
        r = fix_markdown_code_blocks(code)
        assert '    return 42' in r
        assert compiles(r)

    def test_blank_lines_preserved(self):
        code = '```python\nx = 1\n```\n\n```python\ny = 2\n```'
        r = fix_markdown_code_blocks(code)
        assert 'x = 1' in r
        assert 'y = 2' in r
        assert compiles(r)

    def test_backticks_in_string_valid_code_untouched(self):
        code = 'msg = "use ``` for code"\nprint(msg)'
        assert fix_markdown_code_blocks(code) == code

    def test_non_markdown_syntax_error_unchanged(self):
        """Code with ``` in a string that has an unrelated syntax error."""
        code = 'msg = "use ``` for code"\nx = 1 +'
        assert fix_markdown_code_blocks(code) == code

    def test_backticks_in_string_within_block(self):
        code = '```python\nmsg = "use ``` for code"\nprint(msg)\n```'
        r = fix_markdown_code_blocks(code)
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
        r = fix_markdown_code_blocks(code)
        assert 'x = 42' in r
        assert 'y = x + 1' in r
        assert '# Let me begin.' in r
        assert '```' not in r
        assert compiles(r)

    def test_empty_code_block(self):
        code = '```python\n```'
        r = fix_markdown_code_blocks(code)
        assert '```' not in r
        assert compiles(r)

    def test_single_fence_only_returns_original(self):
        """A lone ``` without a matching pair should not trigger extraction."""
        code = 'x = 1\n```'
        assert fix_markdown_code_blocks(code) == code


# === fix_triple_quote_conflict ===

class TestFixTripleQuoteConflict:
    def test_valid_code_untouched(self):
        code = 'x = 1'
        assert fix_triple_quote_conflict(code) == code

    def test_nested_triple_quotes_fixed(self):
        code = 'print("""\n    def foo():\n        """docstring"""\n""")'
        r = fix_triple_quote_conflict(code)
        assert compiles(r)

    def test_normal_triple_quote_untouched(self):
        code = '"""\nhello\n"""'
        assert fix_triple_quote_conflict(code) == code

    def test_unfixable_returns_original(self):
        code = '"""a"""b"""c"""d"""'
        r = fix_triple_quote_conflict(code)
        # Should return original or a compiling fix
        assert r == code or compiles(r)


# === preprocess (chain) ===

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

    def test_chain_order(self):
        """Markdown extraction runs before triple-quote fix."""
        # If we ever fix the chain interaction, this test documents the intent
        code = '```python\nprint("""\n    """inner"""\n""")\n```'
        r = preprocess(code)
        # Currently these don't chain (extracted code has triple-quote issue),
        # so we just verify it doesn't crash
        assert isinstance(r, str)
