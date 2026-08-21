import sys
import contextlib
assert sys.version_info >= (3, 8), "Requires Python 3.8+"
import os
import json
import http.client
import socket
import urllib.parse
import threading
import time
import logging
import base64
import uuid
from collections import defaultdict

_NO_DEADLINE = object()


class DeadlineHTTPResponse(http.client.HTTPResponse):
    def __init__(self, sock, *, deadline, **kwargs):
        self._deadline = deadline
        self._deadline_socket = sock
        super().__init__(sock, **kwargs)

    def _apply_deadline(self):
        timeout = self._deadline()
        if timeout is not _NO_DEADLINE:
            self._deadline_socket.settimeout(timeout)

    def begin(self):
        self._apply_deadline()
        return super().begin()

    def read1(self, amt=-1):
        self._apply_deadline()
        return super().read1(amt)

    def read(self, amt=None):
        if amt is not None and amt < 0:
            amt = None
        chunks = []
        remaining = amt
        while remaining is None or remaining:
            size = 64 * 1024 if remaining is None else min(64 * 1024, remaining)
            chunk = self.read1(size)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        return b"".join(chunks)


class _DeadlineConnectionMixin:
    def __init__(self, *args, deadline=_NO_DEADLINE, **kwargs):
        self._deadline = (
            deadline
            if deadline is _NO_DEADLINE or deadline is None
            else time.monotonic() + deadline
        )
        if deadline is not _NO_DEADLINE:
            kwargs["timeout"] = self._remaining()
        super().__init__(*args, **kwargs)
        self.response_class = lambda sock, **response_kwargs: DeadlineHTTPResponse(
            sock, deadline=self._remaining, **response_kwargs
        )

    def _remaining(self):
        if self._deadline is _NO_DEADLINE or self._deadline is None:
            return self._deadline
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        return remaining

    def connect(self):
        timeout = self._remaining()
        if timeout is not _NO_DEADLINE:
            self.timeout = timeout
        return super().connect()

    def send(self, data):
        timeout = self._remaining()
        if timeout is not _NO_DEADLINE and self.sock is not None:
            self.sock.settimeout(timeout)
        return super().send(data)


class DeadlineHTTPConnection(_DeadlineConnectionMixin, http.client.HTTPConnection):
    pass


class DeadlineHTTPSConnection(_DeadlineConnectionMixin, http.client.HTTPSConnection):
    pass


from .utils import JSON_INDENT, UsageTracker
from .llm_registry import get_model_config
from .provider_admission import ProviderAdmission
from .conversation import Conversation
from .streaming import wrap_chat_completions_streaming_response

# Define TCP keepalive constants for cross-platform compatibility
try:
    TCP_KEEPIDLE = socket.TCP_KEEPIDLE
except AttributeError:
    TCP_KEEPIDLE = getattr(socket, "TCP_KEEPALIVE", None)  # macOS uses TCP_KEEPALIVE

# Message keys passed through to _call_completions and _call_messages
# in addition to the standard four: 'role', 'content', 'name', 'tool_call_id'
EXTRA_KEYS = {'images', 'audio'}

MEDIA_TYPES = {
    b'\xff\xd8\xff': "image/jpeg",
    b'\x89PN': "image/png",
}

IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg"}
GEMINI_AUDIO_MEDIA_TYPES = {
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/aiff",
    "audio/mp3",
    "audio/aac",
}
AUDIO_MEDIA_TYPES = GEMINI_AUDIO_MEDIA_TYPES | {"audio/mpeg"}
TRANSPORT_MEDIA_TYPES = {
    "completions": IMAGE_MEDIA_TYPES,
    "responses": IMAGE_MEDIA_TYPES,
    "messages": IMAGE_MEDIA_TYPES,
    "gemini": IMAGE_MEDIA_TYPES | GEMINI_AUDIO_MEDIA_TYPES,
}


def _detect_audio_type(data):
    """Detect audio MIME type from file magic bytes."""
    if data[:4] == b'RIFF': return "audio/wav"
    if data[:4] == b'fLaC': return "audio/flac"
    if data[:4] == b'OggS': return "audio/ogg"
    if data[:4] == b'FORM': return "audio/aiff"
    if data[:3] == b'ID3' or data[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
        return "audio/mp3"
    if data[:2] in (b'\xff\xf1', b'\xff\xf9'):
        return "audio/aac"
    raise ValueError(f"Unsupported audio format (magic: {data[:4].hex()})")

logger = logging.getLogger('agentlib')

class ValidationError(Exception): pass
class BadRequestError(Exception): pass
class MaxTokensError(Exception): pass
class ContextOverflowError(Exception): pass

CONTEXT_INPUT_BUFFER = 4_000
CONTEXT_OUTPUT_HEADROOM = 16_000
TOKEN_RATIO_EMA_ALPHA = 0.2



def _parse_completions_response(response_json):
    if 'choices' not in response_json:
        raise Exception(f"choices missing from response: {response_json}")
    choice = response_json['choices'][0]
    return choice.get('message', {}), choice.get('finish_reason'), response_json.get('usage')





def _openai_compatible_message_to_transport_blocks(message):
    blocks = []
    text = message.get('content')
    if isinstance(text, str) and text:
        blocks.append({'type': 'text', 'text': text})
    elif isinstance(text, list):
        for item in text:
            kind = item['type']
            if kind in ('text', 'input_text', 'output_text'):
                blocks.append({'type': 'text', 'text': item['text']})
            elif kind == 'image_url':
                image_url = item['image_url']
                blocks.append({
                    'type': 'attachment',
                    'media_type': item.get('media_type'),
                    'data_type': 'url',
                    'data': image_url['url'] if isinstance(image_url, dict) else image_url,
                })
            elif kind == 'input_file':
                blocks.append({
                    'type': 'attachment',
                    'media_type': item.get('media_type'),
                    'data_type': 'provider_id',
                    'data': item['file_id'],
                })
            else:
                raise NotImplementedError(f"Unknown legacy block in completions: {kind!r}")
    reasoning = (
        message.get('reasoning_content')
        or message.get('reasoning')
    )
    if reasoning:
        blocks.insert(0, {'type': 'reasoning', 'text': reasoning})
    for call in message.get('tool_calls') or []:
        function = call.get('function') or {}
        raw_args = function.get('arguments', {})
        args = (
            json.loads(raw_args)
            if isinstance(raw_args, str)
            else raw_args
        )
        tc_block = {
            'type': 'tool_call',
            'id': call.get('id'),
            'name': function.get('name'),
            'args': args,
        }
        if 'thoughtSignature' in call:
            tc_block['provider_metadata'] = {'thought_signature': call['thoughtSignature']}
        elif 'provider_metadata' in call:
            tc_block['provider_metadata'] = call['provider_metadata']
        blocks.append(tc_block)
    return blocks


def _iter_json_dicts(content):
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        if content[pos] != '{':
            next_brace = content.find('{', pos)
            if next_brace == -1:
                break
            pos = next_brace
        try:
            obj, end = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        if isinstance(obj, dict):
            yield obj, pos, end - 1
        pos = end


def _merge_tool_call_documents(content):
    merged = []
    for obj, _, _ in _iter_json_dicts(content):
        function_calls = obj.get("function_calls")
        if function_calls is None:
            continue
        if not isinstance(function_calls, list):
            raise ValidationError("function_calls must be a list.")
        merged.extend(function_calls)
    return merged


def _shimpp_merge_multiple_tool_call_documents(content):
    if content.count('"function_calls"') <= 1:
        return content
    merged = _merge_tool_call_documents(content)
    return json.dumps({"function_calls": merged}, indent=JSON_INDENT) if merged else content


def _shimpp_extract_tool_call_document(content):
    for obj, _, _ in _iter_json_dicts(content):
        function_calls = obj.get("function_calls")
        if function_calls is None:
            continue
        if not isinstance(function_calls, list):
            raise ValidationError("function_calls must be a list.")
        return json.dumps(obj, indent=JSON_INDENT)
    return content


def _shimpp_close_unterminated_tool_call_json(content):
    if '"function_calls"' not in content:
        return content
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        pass
    else:
        function_calls = obj.get("function_calls") if isinstance(obj, dict) else None
        if function_calls is not None and not isinstance(function_calls, list):
            raise ValidationError("function_calls must be a list.")
        return content
    stack = []
    in_string = False
    escape = False
    for ch in content:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    if not stack:
        return content
    candidate = content + ''.join('}' if ch == '{' else ']' for ch in reversed(stack))
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return content
    function_calls = obj.get("function_calls") if isinstance(obj, dict) else None
    if function_calls is None:
        return content
    if not isinstance(function_calls, list):
        raise ValidationError("function_calls must be a list.")
    return candidate


TOOL_CALL_RESPONSE_PREPROCESSORS = (
    _shimpp_merge_multiple_tool_call_documents,
    _shimpp_close_unterminated_tool_call_json,
    _shimpp_extract_tool_call_document,
)


def _preprocess_tool_call_response(content):
    for fn in TOOL_CALL_RESPONSE_PREPROCESSORS:
        content = fn(content)
    return content


def _normalize_json_containers(value, schema, root):
    if '$ref' in schema:
        target = root
        for part in schema['$ref'].removeprefix('#/').split('/'):
            target = target[part]
        schema = target

    nullable = False
    if 'anyOf' in schema:
        nullable = any(item.get('type') == 'null' for item in schema['anyOf'])
        variants = [item for item in schema['anyOf'] if item.get('type') != 'null']
        if len(variants) == 1:
            schema = variants[0]
            if '$ref' in schema:
                target = root
                for part in schema['$ref'].removeprefix('#/').split('/'):
                    target = target[part]
                schema = target

    expected = schema.get('type')
    if isinstance(value, str) and expected in ('object', 'array'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = object()
        if nullable and decoded is None:
            value = None
        elif (
            expected == 'object' and isinstance(decoded, dict)
            or expected == 'array' and isinstance(decoded, list)
        ):
            value = decoded

    if isinstance(value, dict):
        properties = schema.get('properties', {})
        return {
            key: _normalize_json_containers(item, properties.get(key, {}), root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        item_schema = schema.get('items', {})
        return [
            _normalize_json_containers(item, item_schema, root)
            for item in value
        ]
    return value


def _normalize_tool_arguments(arguments, tool):
    schema = tool.model_json_schema()
    return _normalize_json_containers(arguments, schema, schema)


def _extract_tool_calls_json(content):
    for obj, json_start_index, json_end_index in _iter_json_dicts(content):
        function_calls = obj.get("function_calls")
        if function_calls is None:
            continue
        if not isinstance(function_calls, list):
            raise ValidationError("function_calls must be a list.")
        return obj, json_start_index, json_end_index
    if '{' not in content:
        raise ValidationError("No JSON object found (missing '{').")
    if '}' not in content:
        raise ValidationError("Found '{' but no corresponding closing '}' found afterwards.")
    raise ValidationError('No JSON object containing "function_calls" found.')


def _gemini_resolve_schema_refs(schema, defs=None):
    if defs is None:
        defs = schema.get('$defs', {}) if isinstance(schema, dict) else {}
    if isinstance(schema, list):
        return [_gemini_resolve_schema_refs(item, defs) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if '$ref' in schema:
        ref = schema['$ref']
        if not ref.startswith('#/$defs/'):
            raise ValidationError(f"Unsupported schema ref for Gemini tools: {ref}")
        name = ref.split('/')[-1]
        target = defs.get(name)
        if target is None:
            raise ValidationError(f"Missing schema ref target for Gemini tools: {ref}")
        merged = dict(target)
        for k, v in schema.items():
            if k != '$ref':
                merged[k] = v
        return _gemini_resolve_schema_refs(merged, defs)
    return {
        k: _gemini_resolve_schema_refs(v, defs)
        for k, v in schema.items()
        if k != '$defs'
    }


def _gemini_transform_schema(schema):
    """
    Convert a Pydantic JSON schema into Gemini's function declaration schema subset.

    Gemini supports a narrower subset of OpenAPI/JSON Schema than Pydantic emits.
    In particular, Pydantic often emits unsupported keys such as:
    - additionalProperties (for dict[...] fields)
    - title / default
    - anyOf with {"type": "null"} for Optional fields
    - $defs / $ref for nested models
    """
    schema = _gemini_resolve_schema_refs(schema)

    def transform(node):
        if isinstance(node, list):
            return [transform(item) for item in node]
        if not isinstance(node, dict):
            return node

        if 'anyOf' in node:
            variants = node['anyOf']
            non_null = [v for v in variants if not (isinstance(v, dict) and v.get('type') == 'null')]
            if len(non_null) == 1:
                merged = dict(non_null[0])
                for k, v in node.items():
                    if k != 'anyOf':
                        merged[k] = v
                node = merged
            else:
                raise ValidationError(f"Unsupported union schema for Gemini tools: {node}")

        if 'oneOf' in node or 'allOf' in node:
            key = 'oneOf' if 'oneOf' in node else 'allOf'
            raise ValidationError(f"Unsupported composite schema '{key}' for Gemini tools: {node}")

        out = {}
        if 'description' in node:
            out['description'] = node['description']
        if 'enum' in node:
            out['enum'] = node['enum']
        if 'format' in node:
            out['format'] = node['format']

        node_type = node.get('type')
        if node_type in ('string', 'integer', 'number', 'boolean'):
            out['type'] = node_type
            return out

        if node_type == 'array':
            out['type'] = 'array'
            if 'items' in node:
                out['items'] = transform(node['items'])
            return out

        if node_type == 'object' or 'properties' in node or 'additionalProperties' in node:
            out['type'] = 'object'
            props = node.get('properties', {})
            if props:
                out['properties'] = {k: transform(v) for k, v in props.items()}
            required = [name for name in node.get('required', []) if name in props]
            if required:
                out['required'] = required
            return out

        return out or node

    return transform(schema)


def _gemini_schema_has_unsupported_fieldtypes(schema):
    """
    Return True when a schema uses constructs that Gemini function calling does
    not reliably support and should therefore use shim mode instead of native
    tool calling.

    Current unsupported cases:
    - dict/map-like objects emitted as additionalProperties
    - unresolved refs outside $defs
    - non-optional unions / oneOf / allOf composites
    - underspecified arrays/objects that do not give Gemini enough structure
    """
    def visit(node):
        if isinstance(node, list):
            return any(visit(item) for item in node)
        if not isinstance(node, dict):
            return False

        if '$ref' in node and not str(node['$ref']).startswith('#/$defs/'):
            return True
        if 'additionalProperties' in node:
            return True
        if 'oneOf' in node or 'allOf' in node:
            return True
        if 'anyOf' in node:
            variants = node['anyOf']
            non_null = [v for v in variants if not (isinstance(v, dict) and v.get('type') == 'null')]
            if len(non_null) != 1:
                return True
            return any(visit(v) for v in non_null)
        if node.get('type') == 'array':
            items = node.get('items')
            if not isinstance(items, dict) or not items:
                return True
            if items.get('type') == 'object' and not items.get('properties') and 'additionalProperties' not in items:
                return True
        if node.get('type') == 'object':
            if not node.get('properties') and 'additionalProperties' not in node:
                return True

        return any(visit(v) for v in node.values())

    return visit(schema)


def one_shot(model, messages=None, system=None, user=None, tools=None):
    """Make one model request without creating a conversation or agent."""
    if messages is not None:
        if system is not None or user is not None:
            raise ValueError(
                "messages is mutually exclusive with system and user"
            )
        return LLMClient(model).call(messages, tools=tools)

    if user is None:
        raise ValueError("one of messages or user is required")

    request = []
    if system is not None:
        request.append({
            "role": "system",
            "content": [{"type": "text", "text": system}],
        })
    request.append({
        "role": "user",
        "content": [{"type": "text", "text": user}],
    })
    response = LLMClient(model).call(request, tools=tools)
    if tools is not None:
        return response
    return '\n'.join(
        block["text"]
        for block in response.get("content", [])
        if block.get("type") == "text" and block.get("text")
    )


class LLMClient:
    usage_tracker = UsageTracker()

    def __init__(self, model_name, native=None):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.timeout = self.model_config.get('timeout')
        self.provider_admission = ProviderAdmission.from_model_config(
            model_name, self.model_config
        )
        self.native = self.model_config.get('tools') if native is None else native
        self.on_retry = None
        self._current_input_bytes = None


    def _input_bytes(self, messages, tools=None):
        messages = self._public_messages(messages)
        payload = {"messages": messages}
        if tools:
            payload["tools"] = tools
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=self._json_size_default).encode("utf-8"))

    @staticmethod
    def _public_messages(messages):
        return [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]

    @staticmethod
    def _json_size_default(value):
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _input_tokens_per_byte(self):
        return getattr(self.usage_tracker, "input_tokens_per_byte", {}).get(self.model_name)

    def _update_input_tokens_per_byte(self, input_bytes, usage):
        if not usage or not input_bytes:
            return
        ratio_usage = usage
        if transform := self.model_config.get('token_transform'):
            ratio_usage = transform(ratio_usage)
        prompt_tokens, cached_tokens = self.usage_tracker._prompt_and_cached_tokens(ratio_usage)
        input_tokens = prompt_tokens + cached_tokens
        if input_tokens <= 0:
            return
        ratios = getattr(self.usage_tracker, "input_tokens_per_byte", None)
        if ratios is None:
            ratios = {}
            self.usage_tracker.input_tokens_per_byte = ratios
        observed = input_tokens / input_bytes
        old = ratios.get(self.model_name)
        ratios[self.model_name] = observed if old is None else (
            old * (1 - TOKEN_RATIO_EMA_ALPHA) + observed * TOKEN_RATIO_EMA_ALPHA
        )

    def _estimate_input_tokens(self, input_bytes):
        ratio = self._input_tokens_per_byte()
        if ratio is None:
            return None
        return int(input_bytes * ratio) + CONTEXT_INPUT_BUFFER

    def _validate_context_budget(self, input_bytes):
        self._current_input_bytes = input_bytes
        estimated_input = self._estimate_input_tokens(input_bytes)
        if estimated_input is None:
            return
        max_input_tokens = self.model_config.get('max_input_tokens')
        if max_input_tokens is not None and estimated_input > max_input_tokens:
            raise ContextOverflowError(
                f"estimated input {estimated_input:,} tokens exceeds max_input_tokens "
                f"{max_input_tokens:,} for {self.model_name}"
            )
        context_window = self.model_config.get('context_window')
        if context_window is not None and estimated_input + CONTEXT_OUTPUT_HEADROOM > context_window:
            raise ContextOverflowError(
                f"estimated input {estimated_input:,} tokens + output headroom "
                f"{CONTEXT_OUTPUT_HEADROOM:,} exceeds context_window "
                f"{context_window:,} for {self.model_name}"
            )
    def validate_media_type(self, media_type):
        if not isinstance(media_type, str) or "/" not in media_type:
            raise BadRequestError("Invalid media attachment type")
        api_type = self.model_config["api_type"]
        if media_type not in TRANSPORT_MEDIA_TYPES.get(api_type, set()):
            if media_type.startswith("audio/"):
                if api_type == "completions":
                    raise BadRequestError("Audio input is not supported by OpenAI completions API")
                elif api_type == "responses":
                    raise BadRequestError("Audio input is not supported by OpenAI Responses API")
                elif api_type == "messages":
                    raise BadRequestError("Audio input is not supported by Anthropic Messages API")
            raise NotImplementedError(
                f"{api_type} transport does not support {media_type} attachments"
            )

    def _attachment_data(self, block):
        data_type = block.get('data_type')
        if data_type == 'bytes':
            return block['data']
        if data_type == 'filepath':
            with open(block['data'], 'rb') as f:
                return f.read()
        raise NotImplementedError(
            f"Unsupported attachment data type for reading: {data_type!r}"
        )

    def _binary_attachment(self, block):
        self.validate_media_type(block['media_type'])
        return block['media_type'], self._attachment_data(block)

    def _openai_attachment(self, block):
        data_type = block.get('data_type')
        if data_type == 'url':
            return {
                'type': 'image_url',
                'image_url': {'url': block['data']},
            }
        if data_type == 'provider_id':
            return {
                'type': 'input_file',
                'file_id': block['data'],
            }
        media_type, data = self._binary_attachment(block)
        if media_type in IMAGE_MEDIA_TYPES:
            encoded = base64.b64encode(data).decode('ascii')
            return {
                'type': 'image_url',
                'image_url': {'url': f"data:{media_type};base64,{encoded}"},
            }
        if media_type in AUDIO_MEDIA_TYPES:
            encoded = base64.b64encode(data).decode('ascii')
            audio_format = 'wav' if media_type == 'audio/wav' else 'mp3'
            return {
                'type': 'input_audio',
                'input_audio': {
                    'data': encoded,
                    'format': audio_format,
                },
            }
        raise NotImplementedError(f"Unsupported media type for completions: {media_type!r}")

    def _responses_attachment(self, block):
        data_type = block.get('data_type')
        if data_type == 'url':
            return {'type': 'input_image', 'image_url': block['data']}
        if data_type == 'provider_id':
            item = {'type': 'input_file', 'file_id': block['data']}
            if block.get('media_type'):
                item['media_type'] = block['media_type']
            return item
        media_type, data = self._binary_attachment(block)
        encoded = base64.b64encode(data).decode('ascii')
        return {
            'type': 'input_image',
            'image_url': f"data:{media_type};base64,{encoded}",
        }

    def _anthropic_attachment(self, block):
        data_type = block.get('data_type')
        if data_type in ('url', 'provider_id'):
            raise NotImplementedError(
                f"Anthropic Messages API does not support attachment data type: {data_type!r}"
            )
        media_type, data = self._binary_attachment(block)
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': media_type,
                'data': base64.b64encode(data).decode('ascii'),
            },
        }

    def _gemini_attachment(self, block):
        data_type = block.get('data_type')
        if data_type in ('url', 'provider_id'):
            raise NotImplementedError(
                f"Gemini generateContent does not support attachment data type: {data_type!r}"
            )
        media_type, data = self._binary_attachment(block)
        return {
            'inlineData': {
                'mimeType': media_type,
                'data': base64.b64encode(data).decode('ascii'),
            },
        }

    @staticmethod
    def _strip_response_media(message):
        return message

    def _completions_messages(self, messages):
        prepared = []
        for original in messages:
            message = dict(original)
            role = message['role']
            blocks = message.get('content', [])
            if role == 'tool':
                output = []
                for block in blocks:
                    kind = block['type']
                    if kind == 'text':
                        output.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {kind!r}"
                        )
                tool_message = {
                    'role': 'tool',
                    'tool_call_id': message['tool_call_id'],
                    'content': '\n'.join(output),
                }
                if 'name' in message:
                    tool_message['name'] = message['name']
                prepared.append(tool_message)
                continue

            content = []
            tool_calls = []
            cache_breakpoint = (
                bool(message.get('_prompt_cache_breakpoint'))
                and self.model_config.get('explicit_prompt_cache')
            )
            for block in blocks:
                kind = block['type']
                if kind == 'text':
                    item = {'type': 'text', 'text': block['text']}
                    if cache_breakpoint:
                        item['prompt_cache_breakpoint'] = {'mode': 'explicit'}
                        cache_breakpoint = False
                    content.append(item)
                elif kind == 'attachment':
                    content.append(self._openai_attachment(block))
                elif kind == 'reasoning':
                    continue
                elif kind == 'commentary':
                    content.append({'type': 'text', 'text': block['text']})
                elif kind == 'tool_call':
                    raw_args = block['args']
                    tool_calls.append({
                        'id': block['id'],
                        'type': 'function',
                        'function': {
                            'name': block['name'],
                            'arguments': (
                                raw_args
                                if isinstance(raw_args, str)
                                else json.dumps(raw_args)
                            ),
                        },
                    })
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            out = {'role': role}
            if content:
                out['content'] = content
            else:
                out['content'] = ''
            if tool_calls:
                out['tool_calls'] = tool_calls
            if cache_breakpoint and content:
                content[-1] = {
                    **content[-1],
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                }
            prepared.append(out)
        return prepared

    def _call_completions(self, messages, tools):
        """
        Call OpenAI Completions API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        transport_messages = list(messages)
        context_window = self.model_config.get('context_window')
        extra_config = dict(self.model_config.get('config', {}))
        current_max_tokens = extra_config.get('max_tokens')
        max_tokens_retry = 0

        while True:
            req = {
                "model": self.model_config['model'],
                "messages": self._completions_messages(transport_messages),
                **extra_config,
            }
            if tools:
                req.update({
                    "tools": tools,
                    "tool_choice": "required",
                })
            if self.model_config['port'] == 443:
                conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
                conn.connect()
                sock = conn.sock
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if TCP_KEEPIDLE is not None:
                    sock.setsockopt(
                        socket.IPPROTO_TCP, TCP_KEEPIDLE, 60
                    )  # 60 sec idle before keepalive
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)    # 10 sec between probes
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)       # 3 probes before giving up
            else:
                conn = DeadlineHTTPConnection(self.model_config['host'], self.model_config['port'], timeout=self.timeout, deadline=self.timeout)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.model_config['api_key']}",
            }
            body = json.dumps(req)
            request_path = self.model_config.get('request_path', self.model_config['path'])
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("----------- TO LLM -----------")
                    logger.debug(f"POST {request_path} {headers}")
                    logger.debug(body)
                conn.request("POST", request_path, body, headers)
                response = conn.getresponse()
                content_type = ""
                if getattr(response, "headers", None):
                    content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type.lower():
                    response = wrap_chat_completions_streaming_response(response)
                response_data = response.read().decode()
                if logger.isEnabledFor(logging.INFO):
                    logger.info("---------- FROM LLM ----------")
                    logger.info(response_data)
                if response.status == 429:
                    print(response)
                    logger.warning("Throttled. Waiting 20s")
                    time.sleep(20)
                    raise Exception("Throttled")
                if response.status == 400:
                    logger.debug(req)
                    raise BadRequestError(response_data.strip())
                elif response.status != 200:
                    raise Exception(f"API Error {response.status}: {response_data}")

                response_json = json.loads(response_data)
                parser = self.model_config.get('response_parser') or _parse_completions_response
                provider_message, stop_reason, usage = parser(response_json)
                message = {
                    'role': 'assistant',
                    'content': _openai_compatible_message_to_transport_blocks(provider_message),
                    'provider_metadata': {'stop_reason': stop_reason},
                }
                if usage:
                    self.usage_tracker.log(self.model_name, usage)
                    self._update_input_tokens_per_byte(self._current_input_bytes, usage)

                # Truncated response: feed it back and retry with doubled max_tokens.
                # Keeps doubling until prompt + output would exceed context_window.
                # Retry messages stay local — they never reach the Conversation history.
                if stop_reason in ('max_tokens', 'length', 'MAX_TOKENS') and context_window and current_max_tokens and usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    next_max_tokens = current_max_tokens * 2
                    if prompt_tokens + next_max_tokens <= context_window:
                        max_tokens_retry += 1
                        if self.on_retry:
                            self.on_retry("max_tokens", max_tokens_retry)
                        transport_messages.append(message)
                        transport_messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'text',
                                'text': (
                                    'Incomplete response detected. '
                                    'Resubmit your response.'
                                ),
                            }],
                        })
                        current_max_tokens = next_max_tokens
                        extra_config['max_tokens'] = current_max_tokens
                        logger.warning(f"stop_reason={stop_reason}, doubling max_tokens to {current_max_tokens}")
                        continue

                return message
            finally:
                conn.close()

    def _responses_request(self, messages, tools):
        config = dict(self.model_config.get('config', {}))
        if 'reasoning_effort' in config:
            config['reasoning'] = {'effort': config.pop('reasoning_effort')}
        if 'max_tokens' in config:
            config['max_output_tokens'] = config.pop('max_tokens')
        input_items = []
        has_cache_breakpoint = False
        for message in messages:
            role = message['role']
            blocks = message.get('content', [])
            if role == 'tool':
                output = []
                for block in blocks:
                    kind = block['type']
                    if kind == 'text':
                        output.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {kind!r}"
                        )
                input_items.append({
                    'type': 'function_call_output',
                    'call_id': message['tool_call_id'],
                    'output': '\n'.join(output),
                })
                continue

            message_items = []
            content = []

            def flush_content(phase=None):
                nonlocal content
                if content:
                    item = {'role': role, 'content': content}
                    if phase is not None:
                        item['phase'] = phase
                    message_items.append(item)
                    content = []

            for block in blocks:
                kind = block['type']
                if kind in ('text', 'commentary'):
                    phase = 'commentary' if kind == 'commentary' else None
                    if content and phase is not None:
                        flush_content()
                    content.append({
                        'type': (
                            'output_text'
                            if role == 'assistant'
                            else 'input_text'
                        ),
                        'text': block['text'],
                    })
                    if phase is not None:
                        flush_content(phase)
                elif kind == 'attachment':
                    content.append(self._responses_attachment(block))
                elif kind == 'reasoning':
                    flush_content()
                    metadata = block.get('provider_metadata') or {}
                    if 'encrypted_content' in metadata:
                        message_items.append({
                            'type': 'reasoning',
                            'encrypted_content': metadata['encrypted_content'],
                            'summary': [],
                        })
                elif kind == 'tool_call':
                    flush_content()
                    raw_args = block['args']
                    message_items.append({
                        'type': 'function_call',
                        'call_id': block['id'],
                        'name': block['name'],
                        'arguments': (
                            raw_args
                            if isinstance(raw_args, str)
                            else json.dumps(raw_args)
                        ),
                    })
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            flush_content()
            if (
                message.get('_prompt_cache_breakpoint')
                and self.model_config.get('explicit_prompt_cache')
            ):
                message_items[-1]['content'][-1] = {
                    **message_items[-1]['content'][-1],
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                }
                has_cache_breakpoint = True
            input_items.extend(message_items)
        response_tools = [
            {'type': 'function', **tool.get('function', tool)}
            for tool in tools or []
        ]
        req = {'model': self.model_config['model'], 'input': input_items, **config}
        if has_cache_breakpoint:
            req.setdefault('prompt_cache_options', {'mode': 'explicit'})
        if response_tools:
            req['tools'] = response_tools
        return req

    def _parse_responses_result(self, response_json):
        output = response_json.get('output')
        if not output or not isinstance(output, list):
            raise Exception(f"output missing from response: {response_json}")
        blocks = []
        for item in output:
            kind = item['type']
            if kind == 'message':
                phase = item.get('phase')
                block_type = 'text' if phase in (None, 'final') else 'commentary'
                if phase not in (None, 'final', 'commentary'):
                    sys.stderr.write(
                        f"Warning: unrecognized Responses API message phase: {phase!r}\n"
                    )
                for content in item.get('content', []):
                    content_kind = content['type']
                    if content_kind in ('output_text', 'text'):
                        blocks.append({
                            'type': block_type,
                            'text': content['text'],
                        })
                    elif content_kind == 'input_file':
                        blocks.append(_attachment(
                            content.get('media_type'),
                            'provider_id',
                            content['file_id'],
                        ))
                    elif content_kind == 'input_image':
                        blocks.append(_attachment(
                            content.get('media_type') or 'image/jpeg',
                            'url',
                            content['image_url'],
                        ))
                    else:
                        raise NotImplementedError(
                            f"Unknown Responses content type: {content_kind!r}"
                        )
            elif kind == 'reasoning':
                reasoning_block = {
                    'type': 'reasoning',
                    'text': '\n'.join(
                        part['text']
                        for part in item.get('summary', [])
                        if part.get('text')
                    ),
                }
                if 'encrypted_content' in item:
                    reasoning_block['provider_metadata'] = {
                        'encrypted_content': item['encrypted_content'],
                    }
                blocks.append(reasoning_block)
            elif kind == 'output_text':
                blocks.append({'type': 'text', 'text': item['text']})
            elif kind == 'function_call':
                raw_args = item.get('arguments', '')
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str) and raw_args
                    else raw_args
                )
                blocks.append({
                    'type': 'tool_call',
                    'id': item.get('call_id') or item.get('id'),
                    'name': item['name'],
                    'args': args,
                })
            else:
                raise NotImplementedError(
                    f"Unknown Responses output type: {kind!r}"
                )

        if not blocks and isinstance(response_json.get('output_text'), str):
            blocks.append({'type': 'text', 'text': response_json['output_text']})

        stop_reason = (response_json.get('incomplete_details') or {}).get('reason')
        if stop_reason is None:
            has_tool_calls = any(b['type'] == 'tool_call' for b in blocks)
            stop_reason = (
                'tool_calls' if has_tool_calls
                else 'stop' if response_json.get('status') == 'completed'
                else response_json.get('status')
            )

        message = {
            'role': 'assistant',
            'content': blocks,
            'provider_metadata': {'stop_reason': stop_reason},
        }
        return message

    def _call_responses(self, messages, tools):
        req = self._responses_request(messages, tools)
        if self.model_config['port'] == 443:
            conn = DeadlineHTTPSConnection(
                self.model_config['host'],
                timeout=self.timeout,
                deadline=self.timeout,
            )
            conn.connect()
        else:
            conn = DeadlineHTTPConnection(
                self.model_config['host'],
                self.model_config['port'],
                timeout=self.timeout,
                deadline=self.timeout,
            )
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.model_config['api_key']}",
        }
        try:
            request_path = self.model_config.get(
                'request_path', self.model_config['path']
            )
            body = json.dumps(req)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {request_path} {headers}")
                logger.debug(body)
            conn.request('POST', request_path, body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 400:
                logger.debug(req)
                raise BadRequestError(response_data.strip())
            if response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            response_json = json.loads(response_data)
            if usage := response_json.get('usage'):
                self.usage_tracker.log(self.model_name, usage)
                self._update_input_tokens_per_byte(self._current_input_bytes, usage)
            return self._parse_responses_result(response_json)
        finally:
            conn.close()

    def _call_messages(self, messages, tools):
        """
        Call Anthropic Messages API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        system_parts = []
        _messages = []
        for message in messages:
            role = message['role']
            blocks = message.get('content', [])
            if role == 'system':
                for block in blocks:
                    if block['type'] == 'text':
                        system_parts.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Anthropic system message does not support content type: {block['type']!r}"
                        )
                continue
            if role == 'tool':
                tool_content = []
                for block in blocks:
                    kind = block['type']
                    if kind == 'text':
                        tool_content.append({'type': 'text', 'text': block['text']})
                    elif kind == 'attachment':
                        tool_content.append(self._anthropic_attachment(block))
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {kind!r}"
                        )
                _messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'tool_result',
                        'tool_use_id': message['tool_call_id'],
                        'content': tool_content,
                    }],
                })
                continue
            content = []
            for block in blocks:
                kind = block['type']
                if kind in ('text', 'commentary'):
                    content.append({'type': 'text', 'text': block['text']})
                elif kind == 'attachment':
                    content.append(self._anthropic_attachment(block))
                elif kind == 'tool_call':
                    content.append({
                        'type': 'tool_use',
                        'id': block['id'],
                        'name': block['name'],
                        'input': block['args'],
                    })
                elif kind == 'reasoning':
                    item = {'type': 'thinking', 'thinking': block.get('text', '')}
                    metadata = block.get('provider_metadata') or {}
                    if 'signature' in metadata:
                        item['signature'] = metadata['signature']
                    elif 'signature' in block:
                        item['signature'] = block['signature']
                    content.append(item)
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            _messages.append({'role': role, 'content': content})

        req = {
            "model": self.model_config['model'],
            "messages": _messages,
            "max_tokens": self.model_config.get('config', {}).get('max_tokens', 4096),
            **{k: v for k, v in self.model_config.get('config', {}).items() if k != 'max_tokens'}
        }
        if system_parts:
            req["system"] = '\n\n'.join(system_parts)
        if tools:
            req.update({
                "tools": [ {
                    "name": t['function']['name'],
                    "description": t['function']['description'],
                    "input_schema": t['function']['parameters'],
                } for t in tools],
                "tool_choice": {"type": "any"},
            })
        conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.model_config['api_key'],
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps(req)
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {self.model_config['path']} {headers}")
                logger.debug(body)
            conn.request("POST", self.model_config['path'], body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 429:
                logger.warning("Throttled. Waiting 20s")
                time.sleep(20)
                raise Exception("Throttled")
            if response.status == 400:
                logger.debug(req)
                raise BadRequestError(response_data.strip())
            elif response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            response_json = json.loads(response_data)
            if usage := response_json.get('usage'):
                self.usage_tracker.log(self.model_name, usage)
                self._update_input_tokens_per_byte(self._current_input_bytes, usage)
            blocks = []
            for content_block in response_json.get('content', []):
                kind = content_block['type']
                if kind == 'text':
                    blocks.append({'type': 'text', 'text': content_block['text']})
                elif kind == 'thinking':
                    item = {'type': 'reasoning', 'text': content_block.get('thinking', '')}
                    if 'signature' in content_block:
                        item['provider_metadata'] = {'signature': content_block['signature']}
                    blocks.append(item)
                elif kind == 'tool_use':
                    blocks.append({
                        'type': 'tool_call',
                        'id': content_block['id'],
                        'name': content_block['name'],
                        'args': content_block['input'],
                    })
                else:
                    raise NotImplementedError(
                        f"Unknown Anthropic content block type: {kind!r}"
                    )
            message = {
                'role': 'assistant',
                'content': blocks,
                'provider_metadata': {'stop_reason': response_json.get('stop_reason')},
            }
            return message
        finally:
            conn.close()

    def _call_gemini(self, messages, tools):
        """
        Call Gemini native generateContent API.

        Args:
            messages: List of projected message dicts.
            tools: Optional tool specifications.
        """
        contents = []
        system_parts = []
        for m in messages:
            role = m['role']
            blocks = m.get('content', [])
            if role == 'system':
                for block in blocks:
                    if block['type'] == 'text':
                        system_parts.append({"text": block['text']})
                    else:
                        raise NotImplementedError(
                            f"Gemini system instruction does not support content type: {block['type']!r}"
                        )
                continue
            if role == 'tool':
                output = []
                for block in blocks:
                    if block['type'] == 'text':
                        output.append(block['text'])
                    else:
                        raise NotImplementedError(
                            f"Unknown tool result content type: {block['type']!r}"
                        )
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {
                        "name": m['name'],
                        "response": {"result": '\n'.join(output)}
                    }}]
                })
                continue
            parts = []
            for block in blocks:
                kind = block['type']
                if kind in ('text', 'commentary'):
                    parts.append({"text": block['text']})
                elif kind == 'attachment':
                    parts.append(self._gemini_attachment(block))
                elif kind == 'tool_call':
                    part = {
                        "functionCall": {
                            "name": block['name'],
                            "args": block['args'],
                        }
                    }
                    metadata = block.get('provider_metadata') or {}
                    if 'thought_signature' in metadata:
                        part['thoughtSignature'] = metadata['thought_signature']
                    elif 'thoughtSignature' in metadata:
                        part['thoughtSignature'] = metadata['thoughtSignature']
                    elif 'thoughtSignature' in block:
                        part['thoughtSignature'] = block['thoughtSignature']
                    elif 'thought_signature' in block:
                        part['thoughtSignature'] = block['thought_signature']
                    parts.append(part)
                elif kind == 'reasoning':
                    part = {'text': block.get('text', ''), 'thought': True}
                    metadata = block.get('provider_metadata') or {}
                    if 'thought_signature' in metadata:
                        part['thoughtSignature'] = metadata['thought_signature']
                    elif 'thoughtSignature' in metadata:
                        part['thoughtSignature'] = metadata['thoughtSignature']
                    elif 'thoughtSignature' in block:
                        part['thoughtSignature'] = block['thoughtSignature']
                    elif 'thought_signature' in block:
                        part['thoughtSignature'] = block['thought_signature']
                    parts.append(part)
                else:
                    raise NotImplementedError(
                        f"Unknown transport content type: {kind!r}"
                    )
            contents.append({
                "role": "model" if role == 'assistant' else "user",
                "parts": parts,
            })
        # Merge consecutive same-role messages (required by Gemini API)
        merged = []
        for entry in contents:
            if merged and merged[-1]['role'] == entry['role']:
                merged[-1]['parts'].extend(entry['parts'])
            else:
                merged.append(entry)
        # Build request
        model_name = self.model_config['model']
        path = f"{self.model_config['path']}/models/{model_name}:generateContent"
        req = {"contents": merged}
        if system_parts:
            req["systemInstruction"] = {"parts": system_parts}
        # Map config keys to generationConfig
        generation_config = {}
        thinking_config = {}
        for k, v in self.model_config.get('config', {}).items():
            if k == 'max_tokens':
                generation_config['maxOutputTokens'] = v
            elif k in ('thinkingBudget', 'thinkingLevel'):
                thinking_config[k] = v
            else:
                generation_config[k] = v
        if thinking_config:
            generation_config['thinkingConfig'] = thinking_config
        if generation_config:
            req["generationConfig"] = generation_config
        if tools:
            req["tools"] = [{"functionDeclarations": [{
                "name": t['function']['name'],
                "description": t['function']['description'],
                "parameters": _gemini_transform_schema(t['function']['parameters']),
            } for t in tools]}]
            req["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
        conn = DeadlineHTTPSConnection(self.model_config['host'], timeout=self.timeout, deadline=self.timeout)
        conn.connect()
        sock = conn.sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if TCP_KEEPIDLE is not None:
            sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.model_config['api_key'],
        }
        body = json.dumps(req)
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("----------- TO LLM -----------")
                logger.debug(f"POST {path} {headers}")
                logger.debug(body)
            conn.request("POST", path, body, headers)
            response = conn.getresponse()
            response_data = response.read().decode()
            if logger.isEnabledFor(logging.INFO):
                logger.info("---------- FROM LLM ----------")
                logger.info(response_data)
            if response.status == 429:
                logger.warning("Throttled. Waiting 20s")
                time.sleep(20)
                raise Exception("Throttled")
            if response.status == 400:
                logger.debug(req)
                raise BadRequestError(response_data.strip())
            elif response.status != 200:
                raise Exception(f"API Error {response.status}: {response_data}")
            response_json = json.loads(response_data)
            if usage := response_json.get('usageMetadata'):
                self.usage_tracker.log(self.model_name, usage)
                self._update_input_tokens_per_byte(self._current_input_bytes, usage)
            if not response_json.get('candidates'):
                raise Exception(f"candidates missing from response: {response_json}")
            candidate = response_json['candidates'][0]
            blocks = []
            for part in candidate.get('content', {}).get('parts', []):
                if 'text' in part:
                    if part.get('thought'):
                        item = {'type': 'reasoning', 'text': part['text']}
                        if 'thoughtSignature' in part:
                            item['provider_metadata'] = {'thought_signature': part['thoughtSignature']}
                        blocks.append(item)
                    else:
                        blocks.append({'type': 'text', 'text': part['text']})
                elif 'functionCall' in part:
                    fc = part['functionCall']
                    call = {
                        'type': 'tool_call',
                        'id': fc.get('id') or f"gemini_{fc['name']}",
                        'name': fc['name'],
                        'args': fc.get('args', {}),
                    }
                    if 'thoughtSignature' in part:
                        call['provider_metadata'] = {'thought_signature': part['thoughtSignature']}
                    blocks.append(call)
                elif 'inlineData' in part:
                    data = part['inlineData']
                    blocks.append(_attachment(
                        data.get('mimeType'),
                        'bytes',
                        base64.b64decode(data['data']),
                    ))
                else:
                    raise NotImplementedError(
                        f"Unknown Gemini part type: {list(part.keys())!r}"
                    )
            message = {
                'role': 'assistant',
                'content': blocks,
                'provider_metadata': {'stop_reason': candidate.get('finishReason')},
            }
            return message
        finally:
            conn.close()

    def prepare_message(self, message):
        raw_content = message.get('content', [])
        if isinstance(raw_content, str):
            raw_content = [raw_content]

        content = []
        tool_calls = []
        for block in raw_content:
            if isinstance(block, str):
                block = {'type': 'text', 'text': block}
            elif not isinstance(block, dict):
                raise TypeError(
                    f"Message content blocks must be dicts or strings, got "
                    f"{type(block).__name__}"
                )
            if block.get('type') == 'tool_call':
                tool_calls.append({
                    'name': block['name'],
                    'arguments': block['args'],
                })
            else:
                content.append(block)

        if tool_calls:
            content.append({
                'type': 'text',
                'text': json.dumps(
                    {'function_calls': tool_calls},
                    indent=JSON_INDENT,
                ),
            })

        if message.get('role') == 'tool':
            text = '\n'.join(
                block.get('text', '')
                for block in content
                if block.get('type') == 'text'
            )
            return {
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': f"{message.get('name', 'tool')}: {text}",
                }],
            }

        return {
            **message,
            'content': content,
        }

    @staticmethod
    def _strip_orphaned_tool_use(messages):
        """Remove canonical tool-call blocks with no matching tool result."""
        tool_result_ids = {
            message['tool_call_id']
            for message in messages
            if (
                message.get('role') == 'tool'
                and 'tool_call_id' in message
            )
        }
        out = []
        for message in messages:
            if message.get('role') != 'assistant':
                out.append(message)
                continue
            content = message.get('content', [])
            kept = [
                block
                for block in content
                if (
                    block.get('type') != 'tool_call'
                    or block.get('id', '') in tool_result_ids
                )
            ]
            if kept:
                out.append({**message, 'content': kept})
        return out

    def _call(self, messages, tools=None):
        if not self.native:
            messages = [ self.prepare_message(msg) for msg in messages ]
        # Drop orphaned tool_use blocks (from interrupted tool-call loops)
        transport_messages = self._strip_orphaned_tool_use(messages)
        self._validate_context_budget(self._input_bytes(transport_messages, tools))
        api_type = self.model_config['api_type']
        callers = {
            "completions": self._call_completions,
            "responses": self._call_responses,
            "messages": self._call_messages,
            "gemini": self._call_gemini,
        }
        try:
            caller = callers[api_type]
        except KeyError:
            raise NotImplementedError(api_type)
        return self._strip_response_media(caller(transport_messages, tools))

    @staticmethod
    def _sleep_backoff(attempt, base=15):
        """
        Exponential back-off helper. Sleeps for `base * 2**attempt` seconds.
        """
        time.sleep(base * (2 ** attempt))

    def call(self, messages, tools=None, retry=3, attempt=0):
        if tools is not None:
            if self.native:
                if (
                    self.model_config.get('api_type') == 'gemini'
                    and any(
                        message.get('role') == 'assistant'
                        and any(
                            block.get('type') == 'tool_call'
                            and not (
                                block.get('provider_metadata') or {}
                            ).get('thought_signature')
                            for block in message.get('content', [])
                        )
                        for message in messages
                    )
                ):
                    return self.tool_call_shim(messages, tools, retry=retry)
                return self.tool_call_native(messages, tools, retry=retry)
            return self.tool_call_shim(messages, tools, retry=retry)
        try:
            with (
                self.provider_admission.admitted()
                if self.provider_admission is not None
                else contextlib.nullcontext()
            ):
                return self._call(messages)
        except ContextOverflowError:
            raise
        except Exception as e:
            err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
            logger.error(f"call {type(e).__name__}: {err}", exc_info=True)
            if retry:
                self._sleep_backoff(attempt)
                return self.call(messages, retry=retry-1, attempt=attempt+1)
            raise

    def tool_call_native(self, messages, tools, retry=5):
        if self.model_config['api_type'] == "gemini":
            for tool in tools.values():
                schema = tool.model_json_schema()
                if _gemini_schema_has_unsupported_fieldtypes(schema):
                    logger.info("Gemini native tool calling fallback to shim mode due to unsupported schema field types")
                    return self.tool_call_shim(messages, tools, retry=retry)

        provider_tools = []
        for name, tool in tools.items():
            schema = tool.model_json_schema()
            schema.pop('title', None)
            provider_tools.append({
                'type': 'function',
                'function': {
                    'description': schema.pop('description', ''),
                    'name': name,
                    'parameters': schema,
                },
            })

        feedback = []
        for attempt in range(retry + 1):
            resp_msg = {}
            try:
                with (
                    self.provider_admission.admitted()
                    if self.provider_admission is not None
                    else contextlib.nullcontext()
                ):
                    resp_msg = self._call(messages + feedback, provider_tools)
                if transform := self.model_config.get('response_transform'):
                    resp_msg = transform(resp_msg, tools)

                metadata = resp_msg.get('provider_metadata') or {}
                stop = metadata.get('stop_reason')
                text = '\n'.join(
                    block.get('text', '')
                    for block in resp_msg.get('content', [])
                    if block.get('type') in ('text', 'commentary')
                )
                if stop in ('max_tokens', 'length', 'MAX_TOKENS'):
                    raise MaxTokensError(
                        f"stop_reason={stop}, content={text[:500]}"
                    )

                tool_calls = [
                    block
                    for block in resp_msg.get('content', [])
                    if block.get('type') == 'tool_call'
                ]
                if not tool_calls:
                    raise ValidationError(
                        f"tool_calls missing (stop_reason={stop}): "
                        f"{text[:1000]}{'...' if len(text) > 1000 else ''}"
                    )

                for tool_call in tool_calls:
                    name = tool_call['name']
                    if name not in tools:
                        raise ValidationError(f"Unknown tool '{name}'")
                    arguments = tool_call.get('args')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as e:
                            raise ValidationError(
                                f"Failed to decode arguments JSON for tool "
                                f"'{name}': {e}"
                            )
                    if not isinstance(arguments, dict):
                        raise ValidationError(
                            f"Arguments for '{name}' are not a dict"
                        )
                    tool = tools[name]
                    arguments = _normalize_tool_arguments(arguments, tool)
                    tool_call['args'] = arguments
                    try:
                        tool.model_validate(arguments)
                    except Exception as e:
                        raise ValidationError(
                            f"Invalid arguments for tool '{name}': {e}"
                        )

                return resp_msg

            except ValidationError as e:
                if attempt >= retry:
                    raise
                logger.info(
                    f"ValidationError: {e}, retry {attempt + 1}/{retry}"
                )
                if resp_msg.get('content'):
                    feedback.append(resp_msg)
                feedback.append({
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': (
                            f"ERROR: {e}. Your previous tool call did not "
                            "match the tool schema. Reply again with a valid "
                            "tool call only."
                        ),
                    }],
                })
            except (BadRequestError, MaxTokensError, ContextOverflowError):
                raise
            except Exception as e:
                err = (
                    str(e)
                    if len(str(e)) < 1000
                    else str(e)[:1000] + '...'
                ).replace("\n", " ")
                logger.error(
                    f"tool_call_native {type(e).__name__}: {err}",
                    exc_info=True,
                )
                if attempt >= retry:
                    raise
                self._sleep_backoff(attempt)

    def tool_call_shim(
        self, messages, tools, retry=3, attempt=0, _feedback=None
    ):
        provider_tools = []
        for name, tool in tools.items():
            schema = tool.model_json_schema()
            provider_tools.append({
                'type': 'function',
                'function': {
                    'description': schema.pop('description', ''),
                    'name': name,
                    'parameters': schema,
                },
            })
        instructions = (
            "### SYSTEM NOTICE ###\n"
            "Available functions:\n"
            f"{json.dumps(provider_tools, indent=JSON_INDENT)}\n\n"
            'You MUST respond ONLY with a JSON object containing a key '
            '"function_calls" which is an ARRAY of one or more function calls '
            'needed to fulfill the request. Each element in the array should '
            'be a JSON object with "name" and "arguments" keys. If multiple '
            'calls are needed, include multiple objects in the array.\n\n'
            'Example Response Format:{\n'
            '  "function_calls": [\n'
            '    {\n'
            '      "name": "function_name_1",\n'
            '      "arguments": { "arg1": "value1" }\n'
            '    }\n'
            '  ]\n'
            '}\n'
        )
        if JSON_INDENT is None:
            instructions = instructions.replace('\n', ' ')
            while '  ' in instructions:
                instructions = instructions.replace('  ', ' ')

        request_messages = [
            {**message, 'content': list(message.get('content', []))}
            for message in messages
        ]
        instruction_block = {'type': 'text', 'text': instructions}
        if request_messages[-1]['role'] == 'user':
            request_messages[-1]['content'].append(instruction_block)
        else:
            request_messages.append({
                'role': 'user',
                'content': [instruction_block],
            })
        if _feedback:
            request_messages.extend(_feedback)

        try:
            with (
                self.provider_admission.admitted()
                if self.provider_admission is not None
                else contextlib.nullcontext()
            ):
                resp_msg = self._call(request_messages)
        except ContextOverflowError:
            raise
        except Exception as e:
            err = (
                str(e)
                if len(str(e)) < 1000
                else str(e)[:1000] + '...'
            ).replace("\n", " ")
            logger.error(f"tool_call_shim {type(e).__name__}: {err}")
            if retry:
                self._sleep_backoff(attempt)
                return self.tool_call_shim(
                    messages, tools, retry - 1, attempt + 1, _feedback
                )
            raise

        try:
            content = '\n'.join(
                block.get('text', '')
                for block in resp_msg.get('content', [])
                if block.get('type') in ('text', 'commentary', 'reasoning')
            )
            content = _preprocess_tool_call_response(content)
            document, json_start_index, json_end_index = (
                _extract_tool_calls_json(content)
            )
            function_calls = document['function_calls']
            if not function_calls:
                raise ValidationError("Function calls are required.")
            blocks = []
            remaining_text = (
                content[:json_start_index] + content[json_end_index + 1:]
            ).strip('`').removeprefix('json').strip()
            if remaining_text:
                blocks.append({'type': 'text', 'text': remaining_text})
            for call in function_calls:
                name = call['name']
                if name not in tools:
                    raise ValidationError(f"Unknown tool '{name}'")
                arguments = call['arguments']
                if not isinstance(arguments, dict):
                    raise ValidationError(
                        f"Arguments for '{name}' are not a dict"
                    )
                tool = tools[name]
                arguments = _normalize_tool_arguments(arguments, tool)
                try:
                    tool.model_validate(arguments)
                except Exception as e:
                    raise ValidationError(
                        f"Invalid arguments for tool '{name}': {e}"
                    )
                blocks.append({
                    'type': 'tool_call',
                    'id': f"call_{uuid.uuid4().hex}",
                    'name': name,
                    'args': arguments,
                })
            return {'role': 'assistant', 'content': blocks}
        except (ValidationError, KeyError) as e:
            if not retry:
                raise
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"ValidationError: {e}, retry")
            feedback = list(_feedback or [])
            if resp_msg.get('content'):
                feedback.append(resp_msg)
            feedback.append({
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': (
                        f'ERROR: {e}. You MUST respond ONLY with a JSON '
                        'object containing "function_calls".'
                    ),
                }],
            })
            return self.tool_call_shim(
                messages, tools, retry - 1, attempt, feedback
            )





    def conversation(self, system_prompt):
        return Conversation(self, system_prompt)
