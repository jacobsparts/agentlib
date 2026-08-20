import base64
import json
import logging
from .utils import JSON_INDENT

logger = logging.getLogger("agentlib")
MEDIA_ATTACHMENTS_FIELD = "_media_attachments"


def _detect_audio_type(data):
    if data[:4] == b'RIFF': return "audio/wav"
    if data[:4] == b'fLaC': return "audio/flac"
    if data[:4] == b'OggS': return "audio/ogg"
    if data[:4] == b'FORM': return "audio/aiff"
    if data[:3] == b'ID3' or data[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
        return "audio/mp3"
    if data[:2] in (b'\xff\xf1', b'\xff\xf9'):
        return "audio/aac"
    raise ValueError(f"Unsupported audio format (magic: {data[:4].hex()})")


class Conversation:
    def __init__(self, llm_client, system_prompt, convo=None):
        self.convo = convo or Convo(llm_client, system_prompt)
        if llm_client is not None:
            self.convo.llm_client = llm_client
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)
        self._legacy_targets = {}

    @property
    def llm_client(self):
        return self.convo.llm_client

    @llm_client.setter
    def llm_client(self, value):
        self.convo.llm_client = value

    def _remember(self, legacy, canonical):
        self._legacy_targets[id(legacy)] = (legacy, canonical)
        return legacy

    @staticmethod
    def _attachment(media_type, data_type, data):
        return {
            "type": "attachment",
            "media_type": media_type,
            "data_type": data_type,
            "data": data,
        }

    def _content_block_to_canonical(self, block):
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            return {"type": "text", "text": block.get("text", "")}
        if kind == "commentary":
            return {"type": "commentary", "text": block.get("text", "")}
        if kind == "reasoning":
            res = {"type": "reasoning", "text": block.get("text", "")}
            if "provider_metadata" in block:
                res["provider_metadata"] = block["provider_metadata"]
            return res
        if kind == "tool_call":
            return dict(block)
        if kind == "attachment":
            return dict(block)
        if kind == "input_file":
            return self._attachment(
                block.get("media_type"),
                "provider_id",
                block["file_id"],
            )
        if kind == "image_url":
            image_url = block["image_url"]
            return self._attachment(
                block.get("media_type") or "image/jpeg",
                "url",
                image_url["url"] if isinstance(image_url, dict) else image_url,
            )
        if kind == "input_audio":
            audio = block["input_audio"]
            media_type = {
                "wav": "audio/wav",
                "mp3": "audio/mp3",
                "mpeg": "audio/mp3",
            }.get(audio.get("format"))
            if media_type is None:
                raise NotImplementedError(
                    f"Unknown input_audio format: {audio.get('format')!r}"
                )
            return self._attachment(
                media_type,
                "bytes",
                base64.b64decode(audio["data"]),
            )
        if kind == "image" and "source" in block:
            source = block["source"]
            if source.get("type") == "base64":
                return self._attachment(
                    source.get("media_type"),
                    "bytes",
                    base64.b64decode(source["data"]),
                )
        if kind == "tool_use":
            return {
                "type": "tool_call",
                "id": block["id"],
                "name": block["name"],
                "args": block["input"],
            }
        raise NotImplementedError(f"Unknown legacy content type: {kind!r}")

    def _to_canonical(self, message):
        from .client import BadRequestError

        blocks = []
        content = message.get("content")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                blocks.append(self._content_block_to_canonical(block))
        elif content is not None:
            raise NotImplementedError(
                f"Unsupported message content type: {type(content)!r}"
            )

        for img in message.get("images") or []:
            if not isinstance(img, bytes):
                raise BadRequestError("Image attachment must be bytes")
            mime = {b'\xff\xd8\xff': "image/jpeg", b'\x89PN': "image/png"}.get(img[:3])
            if mime is None:
                raise BadRequestError("Unsupported image format")
            blocks.append(self._attachment(mime, "bytes", img))

        for aud in message.get("audio") or []:
            if not isinstance(aud, bytes):
                raise BadRequestError("Audio attachment must be bytes")
            try:
                mime = _detect_audio_type(aud)
            except ValueError as e:
                raise BadRequestError(str(e))
            blocks.append(self._attachment(mime, "bytes", aud))

        for item in message.get(MEDIA_ATTACHMENTS_FIELD) or []:
            if not isinstance(item, dict):
                raise BadRequestError("Invalid projected media attachment")
            data = item.get("content")
            if not isinstance(data, bytes):
                raise BadRequestError(
                    "Projected media attachment has no binary content"
                )
            blocks.append(
                self._attachment(item.get("media_type"), "bytes", data)
            )

        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                raw_args = function.get("arguments", {})
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else raw_args
                )
                tc_block = {
                    "type": "tool_call",
                    "id": tool_call.get("id"),
                    "name": function.get("name"),
                    "args": args,
                }
                if "thoughtSignature" in tool_call:
                    tc_block["provider_metadata"] = {
                        "thought_signature": tool_call["thoughtSignature"]
                    }
                elif "provider_metadata" in tool_call:
                    tc_block["provider_metadata"] = tool_call["provider_metadata"]
                blocks.append(tc_block)

        out = {"role": message["role"], "content": blocks}
        for key in ("tool_call_id", "name"):
            if key in message:
                out[key] = message[key]
        out.update({
            key: value
            for key, value in message.items()
            if key.startswith("_") and key != MEDIA_ATTACHMENTS_FIELD
        })
        return out

    def _to_legacy(self, message, *, response=False):
        text_parts = []
        tool_calls = []
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        else:
            for block in raw_content or []:
                kind = block.get("type")
                if kind == "text":
                    if block.get("text"):
                        text_parts.append(block["text"])
                elif kind == "commentary":
                    lines = (block.get("text") or "").split("\n")
                    text_parts.append("# " + "\n# ".join(lines))
                elif kind == "reasoning":
                    continue
                elif kind == "tool_call":
                    raw_args = block["args"]
                    args = (
                        raw_args
                        if isinstance(raw_args, str)
                        else json.dumps(raw_args)
                    )
                    tc = {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": args,
                        },
                    }
                    metadata = block.get("provider_metadata") or {}
                    if "thought_signature" in metadata:
                        tc["thoughtSignature"] = metadata["thought_signature"]
                    elif "thoughtSignature" in metadata:
                        tc["thoughtSignature"] = metadata["thoughtSignature"]
                    elif "thoughtSignature" in block:
                        tc["thoughtSignature"] = block["thoughtSignature"]
                    elif "thought_signature" in block:
                        tc["thoughtSignature"] = block["thought_signature"]
                    tool_calls.append(tc)
                elif kind == "attachment":
                    if response:
                        raise NotImplementedError(
                            "Legacy Conversation cannot store attachment responses"
                        )
                    data_type = block.get("data_type")
                    if data_type == "provider_id":
                        text_parts.append(f"[Attachment: file_id={block.get('data')}]")
                    elif data_type == "url":
                        text_parts.append(f"[Attachment: url={block.get('data')}]")
                    else:
                        raise NotImplementedError(
                            "Legacy Conversation cannot represent stored binary attachment blocks"
                        )
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            content = "\n".join(text_parts)

        out = {
            "role": message.get("role", "assistant"),
            "content": content,
        }
        if tool_calls:
            out["tool_calls"] = tool_calls
        elif "tool_calls" in message:
            out["tool_calls"] = message["tool_calls"]
        for key in ("tool_call_id", "name"):
            if key in message:
                out[key] = message[key]
        out.update({
            key: value
            for key, value in message.items()
            if key.startswith("_")
        })
        provider_metadata = message.get("provider_metadata") or {}
        if "stop_reason" in provider_metadata:
            out["_stop_reason"] = provider_metadata["stop_reason"]
        elif "_stop_reason" in message:
            out["_stop_reason"] = message["_stop_reason"]
        return out

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
                and isinstance(content, str)
                and out.get("role") in ("system", "user")
            ):
                content_hash = hash(content)
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
        return [
            self._remember(self._to_legacy(message), message)
            for message in self.convo.stored_messages()
        ]

    @property
    def messages(self):
        return self.stored_messages()

    def replace_messages(self, messages):
        self._legacy_targets.clear()
        canonical_messages = []
        for message in messages:
            canonical = self._to_canonical(message)
            canonical_messages.append(canonical)
            self._remember(message, canonical)
        self.convo.replace_messages(canonical_messages)

    def append_message(self, message):
        canonical = self.convo.append_message(self._to_canonical(message))
        return self._remember(message, canonical)

    def extend_messages(self, messages):
        return [self.append_message(message) for message in messages]

    def insert_message(self, index, message):
        canonical = self.convo.insert_message(index, self._to_canonical(message))
        return self._remember(message, canonical)

    def pop_message(self, index=-1):
        canonical = self.convo.pop_message(index)
        return self._remember(self._to_legacy(canonical), canonical)

    def update_message(self, message, **changes):
        message.update(changes)
        remembered = self._legacy_targets.get(id(message))
        if remembered is not None and remembered[0] is message:
            canonical = remembered[1]
            canonical.clear()
            canonical.update(self._to_canonical(message))
        return message

    def remove_message_fields(self, message, *fields):
        for field in fields:
            message.pop(field, None)
        remembered = self._legacy_targets.get(id(message))
        if remembered is not None and remembered[0] is message:
            canonical = remembered[1]
            canonical.clear()
            canonical.update(self._to_canonical(message))
        return message

    def _append_message(self, message):
        return self.append_message(message)

    def projected_messages(self):
        result = []
        for msg in self.stored_messages():
            out = dict(msg)
            attachments = out.pop("_attachments", None)
            if attachments:
                for name, content in attachments.items():
                    out["content"] = out.get("content", "").replace(
                        f"[Attachment: {name}]", content
                    )
            result.append(out)
        if self.ephemeral:
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    out = dict(result[i])
                    content = out.get("content", "")
                    out["content"] = self.ephemeral + (
                        "\n\n" + content if content else ""
                    )
                    result[i] = out
                    break

        return self._with_cache_breakpoints(result)

    def _messages(self):
        return self.projected_messages()

    def call(self, messages=None, tools=None, **kwargs):
        if messages is None:
            messages = self.projected_messages()
        else:
            messages = list(messages)
        canonical_messages = [self._to_canonical(m) for m in messages]
        response = self.convo.call(
            canonical_messages,
            tools=tools,
            **kwargs,
        )
        return self._to_legacy(response, response=True)

    def llm(self, tools=None):
        resp_msg = self.call(tools=tools)
        self.append_message(resp_msg)
        return resp_msg

    def usermsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        message = {"role": "user", "content": content, **kwargs}
        return self.append_message(message)

    def toolmsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        message = {"role": "tool", "content": content, **kwargs}
        return self.append_message(message)


class Convo:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        content = (
            system_prompt
            if isinstance(system_prompt, list)
            else [{"type": "text", "text": system_prompt}]
        )
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
        self._messages = list(messages)

    def append_message(self, message):
        self._messages.append(message)
        return message

    def extend_messages(self, messages):
        messages = list(messages)
        self._messages.extend(messages)
        return messages

    def insert_message(self, index, message):
        self._messages.insert(index, message)
        return message

    def pop_message(self, index=-1):
        return self._messages.pop(index)

    def update_message(self, message, **changes):
        message.update(changes)
        return message

    def remove_message_fields(self, message, *fields):
        for field in fields:
            message.pop(field, None)
        return message

    def projected_messages(self):
        result = [dict(m) for m in self._messages]
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

    def usermsg(self, content, **kwargs):
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(content)}]
        return self.append_message({"role": "user", "content": content, **kwargs})
