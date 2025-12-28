"""
AttachmentMixin - Mixin that adds persistent context attachments to agents.

Attachments are named pieces of content that persist in the conversation context.
They can be added, updated, or removed, and are rendered into messages for the LLM.

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
    - attach(name, content): Add or update an attachment
    - detach(name): Remove an attachment from context
    - Attachments are rendered as delimited blocks in messages
    - When an attachment changes, the old version is invalidated
"""

import json


class AttachmentMixin:
    """Mixin that adds attachment support. Use with BaseAgent."""

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()
        
        # {message_index: {name: content}} - content is None for invalidation
        self._message_attachments = {}
        self._conversation_wrapped = False

    def _wrap_conversation(self):
        """Wrap conversation._messages lazily on first use."""
        if self._conversation_wrapped:
            return
        self._conversation_wrapped = True
        conv = self.conversation
        original_messages = conv._messages
        def _messages_with_attachments():
            return self._inject_attachments(original_messages())
        conv._messages = _messages_with_attachments

    def attach(self, name: str, content):
        """
        Add or update an attachment.
        
        Args:
            name: Identifier for this attachment
            content: String, dict, or list content (dicts/lists are JSON-serialized)
        """
        self._wrap_conversation()
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        
        idx = len(self.conversation.messages)
        if idx not in self._message_attachments:
            self._message_attachments[idx] = {}
        self._message_attachments[idx][name] = content

    def detach(self, name: str):
        """
        Remove an attachment from context.
        
        Args:
            name: Identifier of attachment to remove
        """
        self._wrap_conversation()
        idx = len(self.conversation.messages)
        if idx not in self._message_attachments:
            self._message_attachments[idx] = {}
        self._message_attachments[idx][name] = None

    def _render_attachment(self, name: str, content: str) -> str:
        """Render an attachment as a delimited block."""
        return f"-------- BEGIN {name} --------\n{content}\n-------- END {name} ----------"

    def _inject_attachments(self, messages: list) -> list:
        """
        Transform messages to include attachment content.
        
        Walks through messages and attachment state, rendering active attachments
        and invalidation markers into the message stream.
        """
        if not self._message_attachments:
            return messages
        
        result = []
        # Track current state of each attachment: {name: (content, last_seen_idx)}
        active = {}
        
        for idx, msg in enumerate(messages):
            msg_attachments = self._message_attachments.get(idx, {})
            
            # Build attachment text for this message
            attachment_parts = []
            
            # Check for invalidations (content changed or removed)
            for name, content in msg_attachments.items():
                if name in active and active[name][0] != content:
                    # Content changed or removed - mark old as invalid
                    attachment_parts.append(f"[Attachment removed: {name}]")
                
                if content is not None:
                    # Add new/updated content
                    attachment_parts.append(self._render_attachment(name, content))
                    active[name] = (content, idx)
                else:
                    # Removal - clear from active
                    if name in active:
                        del active[name]
            
            if attachment_parts and msg["role"] in ("user", "tool"):
                # Prepend attachments to message content
                attachment_text = "\n\n".join(attachment_parts)
                new_msg = msg.copy()
                new_msg["content"] = attachment_text + "\n\n" + msg["content"]
                result.append(new_msg)
            else:
                result.append(msg)
        
        return result
