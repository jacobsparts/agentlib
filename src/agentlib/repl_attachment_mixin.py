"""
REPLAttachmentMixin - Attachments rendered as REPL file reads.

For REPLAgent-based agents, attachments appear as if the user had read the file
on the REPL, maintaining the illusion of a continuous REPL session.

Example:
    from agentlib import REPLAgent, REPLAttachmentMixin

    class MyAgent(REPLAttachmentMixin, REPLAgent):
        pass

    with MyAgent() as agent:
        agent.attach("config.json", '{"debug": true}')
        result = agent.run("What's in the config?")

The agent sees the attachment inline in the user message:
    >>> read('config.json')
        1→{"debug": true}

    What's in the config?

When the attachment is invalidated (re-attached or detached), the content is
removed and only a small placeholder remains: [Attachment: config.json]
"""

import json


class REPLAttachmentMixin:
    """Mixin that adds REPL-style attachment support. Use with REPLAgent."""

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
            name: Identifier for this attachment (used as filename in synthetic read)
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
        """Render content with line numbers like read() output."""
        lines = content.split('\n')
        return '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))

    def _render_placeholder(self, name: str) -> str:
        """Render a placeholder that looks like a REPL read call."""
        return f">>> read({name!r})\n[Attachment: {name}]"

    def usermsg(self, content, **kwargs):
        if self._pending_attachments:
            # Force new message — don't append to previous REPL output
            self._last_was_repl_output = False

            placeholders = "\n\n".join(
                self._render_placeholder(name)
                for name in self._pending_attachments
            )
            content = placeholders + "\n\n" + (content if isinstance(content, str) else json.dumps(content))
            kwargs['_attachments'] = dict(self._pending_attachments)
            self._pending_attachments.clear()
        super().usermsg(content, **kwargs)
