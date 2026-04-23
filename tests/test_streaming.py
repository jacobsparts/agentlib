import json

from agentlib.streaming import (
    StreamingProtocolError,
    reassemble_chat_completions_stream,
    wrap_chat_completions_streaming_response,
)


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status
        self.reason = "OK"
        self.headers = {}

    def read(self):
        return self._body


class TestStreamingReassembly:
    def test_reassembles_content_stream(self):
        body = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            'data: [DONE]\n\n'
        )
        out = reassemble_chat_completions_stream(body)
        assert out["model"] == "gpt-test"
        assert out["choices"][0]["message"]["role"] == "assistant"
        assert out["choices"][0]["message"]["content"] == "Hello"
        assert out["choices"][0]["finish_reason"] == "stop"
        assert out["usage"]["total_tokens"] == 5

    def test_reassembles_tool_calls_stream(self):
        chunks = [
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "look",
                                "arguments": '{"q":"',
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": 'books"}',
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        out = reassemble_chat_completions_stream(body)
        tool_calls = out["choices"][0]["message"]["tool_calls"]
        assert tool_calls == [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "look",
                "arguments": '{"q":"books"}',
            },
        }]
        assert out["choices"][0]["finish_reason"] == "tool_calls"

    def test_wrapper_returns_non_streaming_json_bytes(self):
        body = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: [DONE]\n\n'
        ).encode()
        response = wrap_chat_completions_streaming_response(FakeResponse(body))
        out = json.loads(response.read().decode())
        assert response.status == 200
        assert out["choices"][0]["message"]["content"] == "Hi"
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_missing_done_raises(self):
        body = 'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        try:
            reassemble_chat_completions_stream(body)
            assert False, "expected StreamingProtocolError"
        except StreamingProtocolError:
            pass