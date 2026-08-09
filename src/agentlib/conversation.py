import json
from .utils import JSON_INDENT


class Conversation:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        self.messages = [ {"role": "system", "content": system_prompt} ]
        self.ephemeral = ""
        self._prompt_cache = []
        self._prompt_cache_model = getattr(llm_client, "model_name", None)

    def _with_cache_breakpoints(self, messages):
        """Annotate projected messages; update continuity for the next call."""
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

    def _messages(self):
        result = []

        for msg in self.messages:
            out = dict(msg)
            attachments = out.pop('_attachments', None)
            if attachments:
                for name, content in attachments.items():
                    out['content'] = out.get('content', '').replace(f'[Attachment: {name}]', content)
            result.append(out)
        if self.ephemeral:
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    out = dict(result[i])
                    content = out.get("content", "")
                    out["content"] = self.ephemeral + ("\n\n" + content if content else "")
                    result[i] = out
                    break

        return self._with_cache_breakpoints(result)

    def _append_message(self, message):
        self.messages.append(message)

    def llm(self, tools=None):
        resp_msg = self.llm_client.call(self._messages(), tools)
        self.messages.append(resp_msg)
        return resp_msg

    def usermsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        message = {"role": 'user', "content": content, **kwargs}
        self._append_message(message)

    def toolmsg(self, content, **kwargs):
        content = content if type(content) is str else json.dumps(content)
        message = {"role": 'tool', "content": content, **kwargs}
        self._append_message(message)
