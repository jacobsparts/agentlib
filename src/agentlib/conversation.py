import json
from .utils import JSON_INDENT

from .preview_refs import render_preview_refs


class Conversation:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        self.messages = [ {"role": "system", "content": system_prompt} ]
        self.ephemeral = ""

    def _messages(self):
        result = []
        expanded_preview_refs = getattr(self, "expanded_preview_refs", {})
        preview_loader = getattr(self, "preview_loader", None)

        for msg in self.messages:
            out = dict(msg)
            attachments = out.pop('_attachments', None)
            if attachments:
                for name, content in attachments.items():
                    out['content'] = out['content'].replace(f'[Attachment: {name}]', content)
            if preview_loader is not None:
                out['content'] = render_preview_refs(out.get('content', ''), expanded_preview_refs, preview_loader)
            result.append(out)

        if self.ephemeral:
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    out = dict(result[i])
                    content = out.get("content", "")
                    out["content"] = self.ephemeral + ("\n\n" + content if content else "")
                    result[i] = out
                    break

        return result

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
