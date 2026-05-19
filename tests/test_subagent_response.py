from agentlib.subagent import SubagentResponse


class DummyAgent:
    def _poll(self):
        pass


def test_error_is_serialized_into_result_and_wait_does_not_raise():
    response = SubagentResponse(DummyAgent())

    response._set_error("boom")

    assert response.done
    assert response.is_error
    assert response.error == "boom"
    assert response.result == "boom"
    assert response.wait() is response


def test_result_is_text_for_success():
    response = SubagentResponse(DummyAgent())

    response._set_result("ok")

    assert response.done
    assert not response.is_error
    assert response.error is None
    assert response.result == "ok"
