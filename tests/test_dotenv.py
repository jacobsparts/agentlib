import os

from agentlib.dotenv import dotenv_values, find_dotenv, load_dotenv


def test_dotenv_values_parses_common_env_syntax(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# ignored
export API_KEY=abc123
SPACED = value with spaces # comment
DOUBLE="line\\nvalue"
SINGLE='literal # value'
EMPTY=
NO_EQUALS
BAD-KEY=value
"""
    )

    assert dotenv_values(env_file) == {
        "API_KEY": "abc123",
        "SPACED": "value with spaces",
        "DOUBLE": "line\nvalue",
        "SINGLE": "literal # value",
        "EMPTY": "",
    }


def test_load_dotenv_does_not_override_by_default(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("API_KEY=from-file\nNEW_KEY=value\n")
    monkeypatch.setenv("API_KEY", "existing")
    monkeypatch.delenv("NEW_KEY", raising=False)

    assert load_dotenv(env_file) is True
    assert os.environ["API_KEY"] == "existing"
    assert os.environ["NEW_KEY"] == "value"

    assert load_dotenv(env_file, override=True) is True
    assert os.environ["API_KEY"] == "from-file"


def test_find_dotenv_walks_up_from_cwd(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=value\n")
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)

    assert find_dotenv() == str(env_file)
