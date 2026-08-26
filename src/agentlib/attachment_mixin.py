"""
AttachmentMixin - Mixin that adds persistent context attachments to agents.

Attachments are named pieces of content that persist in the conversation context.
They can be added, updated, or removed. Content is injected via placeholders in
message content, replaced per text block by Convo.projected_messages().

Example:
    from agentlib import BaseAgent, AttachmentMixin

    class MyAgent(AttachmentMixin, BaseAgent):
        model = 'sonnet'
        system = "You are a helpful assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        agent.attach("config", {"debug": True, "timeout": 30})
        agent.attach("schema", "CREATE TABLE users (id INT, name TEXT)")
        result = agent.run("Update the timeout to 60")

Behavior:
    - attach(name, content): Add or update — buffers until next usermsg()
    - detach(name): Remove — invalidates across all messages
    - Placeholders [Attachment: name] remain as tiny breadcrumbs when invalidated
    - Content is rendered as delimited blocks (-------- BEGIN/END --------)
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryAttachment:
    content: str


def encode_attachment_ref(ref):
    if isinstance(ref, MemoryAttachment):
        return {"__memory_attachment__": True, "content": ref.content}
    return ref


def decode_attachment_ref(ref):
    if isinstance(ref, dict) and ref.get("__memory_attachment__"):
        return MemoryAttachment(ref.get("content", ""))
    return ref


def encode_attachment_refs(refs):
    return {name: encode_attachment_ref(ref) for name, ref in (refs or {}).items()}


def decode_attachment_refs(refs):
    return {name: decode_attachment_ref(ref) for name, ref in (refs or {}).items()}


class AttachmentMixin:
    """Mixin that adds attachment support. Use with BaseAgent."""

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if hasattr(self, '_pending_attachments'):
            return

        self._pending_attachments = {}

    def attach(self, name: str, content):
        """
        Add or update an attachment.

        Args:
            name: Identifier for this attachment
            content: String, dict, or list content (dicts/lists are JSON-serialized)
        """
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)

        self._invalidate_attachment(name)
        self._pending_attachments[name] = self._render_attachment(name, content)

    def detach(self, name: str):
        """
        Remove an attachment from context.

        Args:
            name: Identifier of attachment to remove
        """
        self._invalidate_attachment(name)
        self._pending_attachments.pop(name, None)

    def list_attachments(self) -> dict[str, str]:
        """Get currently active attachments."""
        active = {}
        for msg in self.conversation.stored_messages():
            for name, content in msg.get('_attachments', {}).items():
                active[name] = content
        active.update(self._pending_attachments)
        return active

    def _invalidate_attachment(self, name: str):
        """Remove an attachment from all messages."""
        for msg in self.conversation.stored_messages():
            attachments = msg.get('_attachments')
            if attachments and name in attachments:
                updated_attachments = dict(attachments)
                del updated_attachments[name]
                if updated_attachments:
                    self.conversation.update_message(msg, _attachments=updated_attachments)
                else:
                    self.conversation.remove_message_fields(msg, '_attachments')

    def _render_attachment(self, name: str, content: str) -> str:
        """Render an attachment as a delimited block."""
        return f"-------- BEGIN {name} --------\n{content}\n-------- END {name} ----------"

    def _render_placeholder(self, name: str) -> str:
        """Render a placeholder for an attachment."""
        return f"[Attachment: {name}]"

    def usermsg(self, content, **kwargs):
        if self._pending_attachments:
            placeholders = "\n\n".join(
                self._render_placeholder(name)
                for name in self._pending_attachments
            )
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": json.dumps(content)}]
            content = [
                {"type": "text", "text": placeholders},
                *content,
            ]
            # Merge with any existing _attachments
            existing = kwargs.get('_attachments', {})
            existing.update(self._pending_attachments)
            kwargs['_attachments'] = existing
            self._pending_attachments.clear()
        return super().usermsg(content, **kwargs)
