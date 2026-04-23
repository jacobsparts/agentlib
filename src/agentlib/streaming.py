import json


class StreamingProtocolError(Exception):
    pass


def _iter_sse_data_events(body):
    event_lines = []
    for line in body.splitlines():
        if not line:
            if event_lines:
                payload = "\n".join(event_lines).strip()
                if payload:
                    yield payload
                event_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            event_lines.append(line[5:].lstrip())
    if event_lines:
        payload = "\n".join(event_lines).strip()
        if payload:
            yield payload


def _merge_tool_call_delta(tool_calls, delta_tool_calls):
    for delta in delta_tool_calls:
        index = delta.get("index", len(tool_calls))
        while len(tool_calls) <= index:
            tool_calls.append({
                "id": None,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            })
        current = tool_calls[index]
        if delta.get("id"):
            current["id"] = delta["id"]
        if delta.get("type"):
            current["type"] = delta["type"]
        if function := delta.get("function"):
            if function.get("name"):
                current["function"]["name"] += function["name"]
            if function.get("arguments"):
                current["function"]["arguments"] += function["arguments"]


def reassemble_chat_completions_stream(body):
    response_id = None
    created = None
    model = None
    system_fingerprint = None
    finish_reason = None
    usage = None
    message = {
        "role": "assistant",
        "content": "",
    }
    tool_calls = []
    saw_done = False
    saw_choice = False

    for payload in _iter_sse_data_events(body):
        if payload == "[DONE]":
            saw_done = True
            break
        obj = json.loads(payload)
        if "error" in obj:
            raise StreamingProtocolError(json.dumps(obj["error"]))
        response_id = obj.get("id", response_id)
        created = obj.get("created", created)
        model = obj.get("model", model)
        system_fingerprint = obj.get("system_fingerprint", system_fingerprint)
        if obj.get("usage") is not None:
            usage = obj["usage"]
        for choice in obj.get("choices", []):
            saw_choice = True
            delta = choice.get("delta") or {}
            if delta.get("role"):
                message["role"] = delta["role"]
            if delta.get("content"):
                message["content"] += delta["content"]
            if delta.get("tool_calls"):
                _merge_tool_call_delta(tool_calls, delta["tool_calls"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    if not saw_choice and usage is None:
        raise StreamingProtocolError("No chat completion chunks found in streaming response.")
    if not saw_done:
        raise StreamingProtocolError("Streaming response ended before [DONE].")

    if tool_calls:
        for tool_call in tool_calls:
            if tool_call["id"] is None:
                tool_call["id"] = f"call_{len(tool_call['function']['name'])}_{len(tool_call['function']['arguments'])}"
        message["tool_calls"] = tool_calls

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def read_chat_completions_stream_as_json_bytes(response):
    body = response.read()
    if isinstance(body, bytes):
        body = body.decode()
    return json.dumps(reassemble_chat_completions_stream(body)).encode()


class BufferedStreamingHTTPResponse:
    def __init__(self, response, reader):
        self._response = response
        self._reader = reader
        self._buffer = None
        self.status = response.status
        self.reason = getattr(response, "reason", None)
        self.headers = getattr(response, "headers", None)

    def read(self, *args, **kwargs):
        if self._buffer is None:
            self._buffer = self._reader(self._response)
        return self._buffer

    def __getattr__(self, name):
        return getattr(self._response, name)


def wrap_chat_completions_streaming_response(response):
    return BufferedStreamingHTTPResponse(response, read_chat_completions_stream_as_json_bytes)