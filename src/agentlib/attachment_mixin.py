"""
AttachmentMixin - Mixin that adds persistent context attachments to agents.

Attachments are named pieces of content that persist in the conversation context.
They can be added, updated, or removed. Content is injected via placeholders in
message content, replaced at render time by Conversation._messages().

Example:
    from agentlib import BaseAgent, AttachmentMixin

    class MyAgent(AttachmentMixin, BaseAgent):
        model = 'anthropic/claude-sonnet-4-20250514'
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
        for msg in self.conversation.messages:
            for name, content in msg.get('_attachments', {}).items():
                active[name] = content
        active.update(self._pending_attachments)
        return active

    def _invalidate_attachment(self, name: str):
        """Remove an attachment from all messages."""
        for msg in self.conversation.messages:
            attachments = msg.get('_attachments')
            if attachments and name in attachments:
                del attachments[name]
                if not attachments:
                    del msg['_attachments']

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
            content = placeholders + "\n\n" + (content if isinstance(content, str) else json.dumps(content))
            kwargs['_attachments'] = dict(self._pending_attachments)
            self._pending_attachments.clear()
        super().usermsg(content, **kwargs)
