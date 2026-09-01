import json


def _attachment(media_type, data):
    return {
        "type": "attachment",
        "media_type": media_type,
        "data_type": "bytes",
        "data": data,
    }


def _image_attachment(data):
    from .client import BadRequestError

    if not isinstance(data, bytes):
        raise BadRequestError("Image attachment must be bytes")
    media_type = {
        b"\xff\xd8\xff": "image/jpeg",
        b"\x89PN": "image/png",
    }.get(data[:3])
    if media_type is None:
        raise BadRequestError("Unsupported image format")
    return _attachment(media_type, data)


def _audio_attachment(data):
    from .client import BadRequestError

    if not isinstance(data, bytes):
        raise BadRequestError("Audio attachment must be bytes")
    if data[:4] == b"RIFF":
        media_type = "audio/wav"
    elif data[:4] == b"fLaC":
        media_type = "audio/flac"
    elif data[:4] == b"OggS":
        media_type = "audio/ogg"
    elif data[:4] == b"FORM":
        media_type = "audio/aiff"
    elif data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        media_type = "audio/mp3"
    elif data[:2] in (b"\xff\xf1", b"\xff\xf9"):
        media_type = "audio/aac"
    else:
        raise BadRequestError(f"Unsupported audio format (magic: {data[:4].hex()})")
    return _attachment(media_type, data)


def _content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if (
        isinstance(content, list)
        and content
        and all(
            isinstance(block, dict) and isinstance(block.get("type"), str)
            for block in content
        )
    ):
        return content
    return [{"type": "text", "text": json.dumps(content)}]


def _canonical_message(message):
    content = message.get("content")
    if not isinstance(content, list):
        raise TypeError("canonical message content must be a list of typed blocks")
    if any(
        not isinstance(block, dict) or not isinstance(block.get("type"), str)
        for block in content
    ):
        raise TypeError("canonical message content must contain typed blocks")
    return message


class Convo:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        content = _content_blocks(system_prompt)
        self._messages = [{"role": "system", "content": content}]
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)

    def _with_cache_breakpoints(self, messages):
        if self.llm_client is None:
            return messages
        model_name = getattr(self.llm_client, "model_name", None)
        if model_name != self._prompt_cache_model:
            self._prompt_cache = []
            self._prompt_cache_model = model_name
        cache, self._prompt_cache = self._prompt_cache, []
        annotated = []
        for message in messages:
            out = dict(message)
            content = out.get("content")
            if (
                cache is not False
                and isinstance(content, list)
                and out.get("role") in ("system", "user")
            ):
                content_hash = hash(repr(content))
                if cache:
                    if (expected := cache.pop(0)) is None:
                        if not cache:
                            cache = False
                    elif content_hash != expected:
                        cache = [None] * 3
                    elif not cache:
                        cache = [None] * 4
                out["_prompt_cache_breakpoint"] = True
                self._prompt_cache.append(content_hash)
            annotated.append(out)
        return annotated

    def stored_messages(self):
        return self._messages

    def replace_messages(self, messages):
        messages = list(messages)
        for message in messages:
            _canonical_message(message)
        self._messages = messages

    def append_message(self, message):
        self._messages.append(_canonical_message(message))
        return message

    def extend_messages(self, messages):
        messages = list(messages)
        for message in messages:
            _canonical_message(message)
        self._messages.extend(messages)
        return messages

    def insert_message(self, index, message):
        self._messages.insert(index, _canonical_message(message))
        return message

    def pop_message(self, index=-1):
        return self._messages.pop(index)

    def update_message(self, message, **changes):
        updated = {**message, **changes}
        _canonical_message(updated)
        message.update(changes)
        return message

    def remove_message_fields(self, message, *fields):
        for field in fields:
            message.pop(field, None)
        return message

    def projected_messages(self):
        result = []
        for message in self._messages:
            out = dict(message)
            attachments = out.pop("_attachments", None)
            if attachments:
                content = []
                for block in out.get("content", []):
                    projected = dict(block)
                    if projected.get("type") == "text":
                        text = projected.get("text", "")
                        for name, attachment in attachments.items():
                            text = text.replace(
                                f"[Attachment: {name}]",
                                attachment,
                            )
                        projected["text"] = text
                    content.append(projected)
                out["content"] = content
            result.append(out)
        if self.ephemeral:
            for index in range(len(result) - 1, -1, -1):
                if result[index].get("role") == "user":
                    out = dict(result[index])
                    out["content"] = [
                        {"type": "text", "text": self.ephemeral},
                        *out.get("content", []),
                    ]
                    result[index] = out
                    break
        return self._with_cache_breakpoints(result)

    def call(self, messages=None, additional_messages=(), **kwargs):
        if messages is None:
            messages = self.projected_messages()
        else:
            messages = list(messages)
        messages.extend(additional_messages)
        return self.llm_client.call(messages, **kwargs)

    def llm(self, tools=None):
        resp_msg = self.call(tools=tools)
        self.append_message(resp_msg)
        return resp_msg

    def usermsg(self, content, **kwargs):
        blocks = _content_blocks(content)
        blocks.extend(_image_attachment(data) for data in kwargs.pop("images", ()) or ())
        blocks.extend(_audio_attachment(data) for data in kwargs.pop("audio", ()) or ())
        return self.append_message({
            "role": "user",
            "content": blocks,
            **kwargs,
        })

    def toolmsg(self, content, **kwargs):
        return self.append_message({
            "role": "tool",
            "content": _content_blocks(content),
            **kwargs,
        })
