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
        agent.attach("settings.yaml", 'key: value')
        result = agent.run("What's in the files?")

The agent sees synthetic history like:
    user: >>> submit("Please load the following files: config.json, settings.yaml")
    Please load the following files: config.json, settings.yaml

    assistant: with open('config.json') as f:
        print(f.read())
    with open('settings.yaml') as f:
        print(f.read())

    user: >>> with open('config.json') as f:
    ...     print(f.read())
    {"debug": true}

    >>> with open('settings.yaml') as f:
    ...     print(f.read())
    key: value

Behavior:
    - attach(name, content): Add or update - shows as file read
    - detach(name): Remove - shows as commented-out (redacted) read
    - Multiple attachments at the same point are batched into one exchange
    - If injecting before first user message, a synthetic request is added
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

    def _non_system_count(self) -> int:
        """Count non-system messages (for index alignment with API messages list)."""
        return sum(1 for m in self.conversation.messages if m.get("role") != "system")

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

        idx = self._non_system_count()
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
        idx = self._non_system_count()
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

    def _render_batch(self, reads: dict[str, str], redactions: list[str]) -> tuple[dict, dict]:
        """
        Render multiple attachments as a single synthetic REPL exchange.

        Args:
            reads: {name: content} for files to show as read
            redactions: [name, ...] for files to show as redacted
        """
        code_parts = []
        output_parts = []

        for name, content in reads.items():
            code_parts.append(f"with open({name!r}) as f:\n    print(f.read())")
            output_parts.append(f">>> with open({name!r}) as f:\n...     print(f.read())\n{content}")

        for name in redactions:
            code_parts.append(f"# with open({name!r}) as f:  # [redacted by system]\n#     print(f.read())")
            output_parts.append(f">>> # with open({name!r}) as f:  # [redacted by system]\n>>> #     print(f.read())")

        return (
            {"role": "assistant", "content": "\n".join(code_parts)},
            {"role": "user", "content": "\n\n".join(output_parts) + "\n"},
        )

    def _emit_attachment_batch(self, reads: dict, redactions: list, result: list):
        """Emit a batch of attachments, adding synthetic user request if needed."""
        if not reads and not redactions:
            return

        # Check if we need a synthetic user request to maintain alternation
        last_role = result[-1].get("role") if result else None
        if last_role in (None, "system") and reads:
            if len(reads) == 1:
                request_text = f"Please load {next(iter(reads))}"
            else:
                request_text = f"Please load the following files: {', '.join(reads)}"
            # Format as REPL output to maintain the illusion
            request_msg = f'>>> submit("{request_text}")\n{request_text}\n'
            result.append({"role": "user", "content": request_msg})

        assistant_msg, user_msg = self._render_batch(reads, redactions)
        result.append(assistant_msg)
        result.append(user_msg)

    def _inject_attachments(self, messages: list) -> list:
        """
        Transform messages to include synthetic REPL exchanges for attachments.

        Walks through messages and attachment state. When attachments change,
        injects message pairs that look like file reads in the REPL.

        If injecting would place an assistant message after a system message
        (or at the start), adds a synthetic user request first to maintain
        LLM provider compatibility.
        """
        if not self._attachment_state:
            return messages

        result = []
        # Track current state: {name: content} where content=None means detached
        active = {}

        # Separate system messages (keep them first, don't count in indices)
        system_messages = []
        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        # Add system messages first (they don't participate in attachment injection)
        result.extend(system_messages)

        # Determine max index we need to process (non-system messages + attachments)
        max_idx = len(non_system_messages)
        if self._attachment_state:
            max_idx = max(max_idx, max(self._attachment_state.keys()) + 1)

        for idx in range(max_idx):
            # Process attachment changes at this index
            changes = self._attachment_state.get(idx, {})
            reads = {}
            redactions = []

            for name, content in changes.items():
                prev_content = active.get(name)

                if content is None:
                    # Detaching
                    if name in active:
                        redactions.append(name)
                        del active[name]
                elif prev_content != content:
                    # New or updated attachment
                    if prev_content is not None:
                        # Content changed - redact old version
                        redactions.append(name)
                    reads[name] = content
                    active[name] = content

            self._emit_attachment_batch(reads, redactions, result)

            # Add the original message if it exists at this index
            if idx < len(non_system_messages):
                msg = non_system_messages[idx]
                # Merge consecutive user messages to maintain REPL illusion
                if (result and result[-1].get("role") == "user" and
                    msg.get("role") == "user"):
                    # Append to previous user message
                    prev = result[-1]["content"]
                    sep = "" if prev.endswith("\n") else "\n"
                    result[-1]["content"] = prev + sep + msg["content"]
                else:
                    result.append(msg)

        return result
