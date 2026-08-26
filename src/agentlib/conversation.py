import json


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
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(content)}]
        return self.append_message({"role": "user", "content": content, **kwargs})

    def toolmsg(self, content, **kwargs):
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(content)}]
        return self.append_message({"role": "tool", "content": content, **kwargs})
