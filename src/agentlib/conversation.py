import json
from .utils import JSON_INDENT

class Conversation:
    def __init__(self, llm_client, system_prompt):
        self.llm_client = llm_client
        self.messages = [ {"role": "system", "content": system_prompt} ]
        self._ephemeral_context = []

    def _messages(self):
        result = []
        for msg in self.messages:
            attachments = msg.get('_attachments')
            if attachments:
                out = {k: v for k, v in msg.items() if k != '_attachments'}
                for name, content in attachments.items():
                    out['content'] = out['content'].replace(f'[Attachment: {name}]', content)
                result.append(out)
            else:
                result.append(msg)

        if self._ephemeral_context:
            ephemeral_context = "\n\n".join(self._ephemeral_context)
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    out = dict(result[i])
                    content = out.get("content", "")
                    out["content"] = ephemeral_context + ("\n\n" + content if content else "")
                    result[i] = out
                    break

        return result

    def _append_message(self, message):
        self.messages.append(message)

    def llm(self, tools=None):
        resp_msg = self.llm_client.call(self._messages(), tools)
        self.clear_ephemeral()
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

    def ephemeral(self, text):
        self._ephemeral_context.append(text)

    def clear_ephemeral(self):
        self._ephemeral_context.clear()

