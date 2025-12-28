"""
REPLAttachmentMixin - Attachments rendered as synthetic REPL exchanges.

For REPLAgent-based agents, attachments appear as if the agent had already
read the file via Python code, maintaining the illusion of a continuous
REPL session.

Example:
    from agentlib import REPLAgent, REPLAttachmentMixin

    class MyAgent(REPLAttachmentMixin, REPLAgent):
        pass

    with MyAgent() as agent:
        agent.attach("config.json", '{"debug": true}')
        result = agent.run("What's in the config?")

The agent sees synthetic history like:
    assistant: with open('config.json') as f:
        print(f.read())

    user: >>> with open('config.json') as f:
    ...     print(f.read())
    {"debug": true}

Behavior:
    - attach(name, content): Add or update - shows as file read
    - detach(name): Remove - shows as commented-out (redacted) read
    - Attachments inject message pairs before the next assistant turn
"""

import json


class REPLAttachmentMixin:
    """Mixin that adds REPL-style attachment support. Use with REPLAgent."""

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if hasattr(self, '_attachment_state'):
            return

        # {message_index: {name: content}} - content is None for detachment
        self._attachment_state = {}
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
            name: Identifier for this attachment (used as filename in synthetic read)
            content: String, dict, or list content (dicts/lists are JSON-serialized)
        """
        self._wrap_conversation()
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)

        idx = len(self.conversation.messages)
        if idx not in self._attachment_state:
            self._attachment_state[idx] = {}
        self._attachment_state[idx][name] = content

    def detach(self, name: str):
        """
        Remove an attachment from context.

        Args:
            name: Identifier of attachment to remove
        """
        self._wrap_conversation()
        idx = len(self.conversation.messages)
        if idx not in self._attachment_state:
            self._attachment_state[idx] = {}
        self._attachment_state[idx][name] = None

    def list_attachments(self) -> dict[str, str]:
        """
        Get currently active attachments.

        Returns:
            Dict mapping attachment names to their content
        """
        if not self._attachment_state:
            return {}

        # Replay attachment history to get current state
        active = {}
        for idx in sorted(self._attachment_state.keys()):
            for name, content in self._attachment_state[idx].items():
                if content is None:
                    # Detachment
                    active.pop(name, None)
                else:
                    # Attachment or update
                    active[name] = content

        return active

    def _render_read(self, name: str, content: str) -> tuple[dict, dict]:
        """Render an attachment as a synthetic file read exchange."""
        code = f"with open({name!r}) as f:\n    print(f.read())"
        output = f">>> with open({name!r}) as f:\n...     print(f.read())\n{content}"

        return (
            {"role": "assistant", "content": code},
            {"role": "user", "content": output},
        )

    def _render_redacted(self, name: str) -> tuple[dict, dict]:
        """Render a detached attachment as commented-out code."""
        code = f"# with open({name!r}) as f:  # [redacted by system]\n#     print(f.read())"
        output = f">>> # with open({name!r}) as f:  # [redacted by system]\n>>> #     print(f.read())"

        return (
            {"role": "assistant", "content": code},
            {"role": "user", "content": output},
        )

    def _inject_attachments(self, messages: list) -> list:
        """
        Transform messages to include synthetic REPL exchanges for attachments.

        Walks through messages and attachment state. When attachments change,
        injects message pairs that look like file reads in the REPL.

        For early attachments (before first user message), injects a synthetic
        user request to maintain LLM provider compatibility.
        """
        if not self._attachment_state:
            return messages

        result = []
        # Track current state: {name: content} where content=None means detached
        active = {}

        # Check if we have early attachments (before any user/assistant messages)
        # Attachments added before first user message will be at index 0 or 1 (after system message)
        needs_synthetic_request = False
        early_attachments = {}

        # Check for attachments at index 0 or 1 when there are only system messages
        has_non_system = any(msg.get("role") not in ("system",) for msg in messages)
        if not has_non_system:
            # Check both index 0 and 1 for attachments
            for idx in [0, 1]:
                attachments = self._attachment_state.get(idx, {})
                if attachments:
                    early_attachments.update(attachments)
            if early_attachments:
                needs_synthetic_request = True

        for idx, msg in enumerate(messages):
            # Special handling for index 0 with early attachments
            if idx == 0 and needs_synthetic_request and msg.get("role") == "system":
                # Add system message first
                result.append(msg)
                # Add synthetic user request
                filenames = [name for name, content in early_attachments.items() if content is not None]
                if filenames:
                    if len(filenames) == 1:
                        request_msg = f"Please load {filenames[0]}"
                    else:
                        file_list = ", ".join(filenames)
                        request_msg = f"Please load the following files: {file_list}"
                    result.append({"role": "user", "content": request_msg})

                # Now render the early attachments as file reads
                for name, content in early_attachments.items():
                    if content is not None:
                        assistant_msg, user_msg = self._render_read(name, content)
                        result.append(assistant_msg)
                        result.append(user_msg)
                        active[name] = content

            # Process attachment changes before this message
            changes = self._attachment_state.get(idx, {})
            for name, content in changes.items():
                prev_content = active.get(name)

                if content is None:
                    # Detaching
                    if name in active:
                        assistant_msg, user_msg = self._render_redacted(name)
                        result.append(assistant_msg)
                        result.append(user_msg)
                        del active[name]
                elif prev_content != content:
                    # New or updated attachment
                    if prev_content is not None:
                        # Content changed - show redaction first
                        assistant_msg, user_msg = self._render_redacted(name)
                        result.append(assistant_msg)
                        result.append(user_msg)
                    # Show new content
                    assistant_msg, user_msg = self._render_read(name, content)
                    result.append(assistant_msg)
                    result.append(user_msg)
                    active[name] = content

            # Add the message to result (unless it's index 0 system message we already added)
            if not (idx == 0 and needs_synthetic_request and msg.get("role") == "system"):
                result.append(msg)

        return result
