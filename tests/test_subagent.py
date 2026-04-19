from agentlib.subagent import _wrap_subagent_task


def test_wrap_subagent_task_adds_repl_completion_requirements():
    wrapped = _wrap_subagent_task("Say 'hello' and nothing else.")

    assert "Task:\nSay 'hello' and nothing else." in wrapped
    assert "Your response must be raw Python code only." in wrapped
    assert "call emit(result, release=True)." in wrapped
    assert "If the task only asks for a text answer, use emit(the_text, release=True)." in wrapped