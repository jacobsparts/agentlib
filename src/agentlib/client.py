import sys
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
from collections import defaultdict

from .utils import throttle, JSON_INDENT, UsageTracker
from .llm_registry import get_model_config
from .conversation import Conversation

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


class LLMClient:
    usage_tracker = UsageTracker()

    def __init__(self, model_name, native=None):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.timeout = self.model_config.get('timeout', 300)
        self.concurrency_lock = threading.BoundedSemaphore(self.model_config.get('concurrency',10))
        self.native = self.model_config.get('tools') if native is None else native

    def _call_completions(self, messages, tools):
        """
        Call OpenAI Completions API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Messages may include 'images' key with list of raw bytes (PNG/JPEG).
            tools: Optional tool specifications.
        """
        # OpenAI Completions API-compatible format
        for m in messages:
            if m.pop('audio', None):
                raise BadRequestError("Audio input is not supported by OpenAI completions API")
            if images := m.pop('images', None):
                m['content'] = [
                    *([{"type": "text", "text": m['content']}] if m['content'] else []),
                    *[{"type": "image_url", "image_url": {
                        "url": f"data:{MEDIA_TYPES[img[:3]]};base64,{base64.b64encode(img).decode()}"
                    }} for img in images]
                ]
        req = {
            "model": self.model_config['model'],
            "messages": messages,
            **self.model_config.get('config', {})
        }
        if tools:
            req.update({
                "tools": tools,
                "tool_choice": "required",
            })
        if self.model_config['port'] == 443:
            conn = http.client.HTTPSConnection(self.model_config['host'], timeout=self.timeout)
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
            conn = http.client.HTTPConnection(self.model_config['host'], self.model_config['port'], timeout=self.timeout)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.model_config['api_key']}",
        }
        body = json.dumps(req)
        try:
            throttle(self.model_config['host'], self.model_config.get('tpm', 5))
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
            if usage := response_json.get('usage'):
                self.usage_tracker.log(self.model_name, usage)
            if not 'choices' in response_json:
                raise Exception(f"choices missing from response: {response_json}")
            choice = response_json['choices'][0]
            message = choice['message']
            message['_stop_reason'] = choice.get('finish_reason')
            # Normalize: some providers return tool_calls: null instead of omitting it
            if 'tool_calls' in message and message['tool_calls'] is None:
                del message['tool_calls']
            return message
        finally:
            conn.close()

    def _call_messages(self, messages, tools):
        """
        Call Anthropic Messages API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Messages may include 'images' key with list of raw bytes (PNG/JPEG).
            tools: Optional tool specifications.
        """
        # Anthropic Messages API-compatible format
        for m in messages:
            if m.pop('audio', None):
                raise BadRequestError("Audio input is not supported by Anthropic Messages API")
            if images := m.pop('images', None):
                m['content'] = [
                    *([{"type": "text", "text": m['content']}] if m['content'] else []),
                    *[{"type": "image", "source": {
                        "type": "base64",
                        "media_type": MEDIA_TYPES[img[:3]],
                        "data": base64.b64encode(img).decode()
                    }} for img in images]
                ]
        system_message = None
        _messages = []
        for msg in messages:
            if msg['role'] == 'system':
                if system_message is None:
                    system_message = msg['content']
                else:
                    _messages.append({**msg, 'role': 'user'})
            elif msg['role'] == 'tool':
                _messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'tool_result',
                        'tool_use_id': msg['tool_call_id'],
                        'content': msg['content']
                    }]
                })
            elif msg['role'] == 'assistant' and 'tool_calls' in msg:
                content = []
                if text := msg['content']:
                    content.append({"type": "text", "text": text})
                for row in msg['tool_calls']:
                    tc = row['function']
                    content.append({
                        "type": "tool_use",
                        "id": row['id'],
                        "name": tc['name'],
                        "input": json.loads(tc['arguments'])
                    })
                if content:
                    _messages.append({'role': 'assistant', 'content': content})
            else:
                _messages.append(msg)
        req = {
            "model": self.model_config['model'],
            "messages": _messages,
            "max_tokens": self.model_config.get('config', {}).get('max_tokens', 4096),
            **{k: v for k, v in self.model_config.get('config', {}).items() if k != 'max_tokens'}
        }
        if system_message:
            req["system"] = system_message
        if tools:
            req.update({
                "tools": [ {
                    "name": t['function']['name'],
                    "description": t['function']['description'],
                    "input_schema": t['function']['parameters'],
                } for t in tools],
                "tool_choice": {"type": "any"},
            })
        conn = http.client.HTTPSConnection(self.model_config['host'], timeout=self.timeout)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.model_config['api_key'],
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps(req)
        try:
            throttle(self.model_config['host'], self.model_config.get('tpm', 5))
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
            content = ""
            tool_calls = []
            for content_block in response_json.get('content', []):
                if content_block['type'] == 'text':
                    content += content_block['text']
                elif content_block['type'] == 'tool_use':
                    tool_calls.append({
                        'id': content_block['id'],
                        'function': {
                            'name': content_block['name'],
                            'arguments': json.dumps(content_block['input'])
                        }
                    })
            message = {
                'role': 'assistant',
                'content': content
            }
            message['_stop_reason'] = response_json.get('stop_reason')
            if tool_calls:
                message['tool_calls'] = tool_calls
            return message
        finally:
            conn.close()

    def _call_gemini(self, messages, tools):
        """
        Call Gemini native generateContent API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Messages may include 'images' key with list of raw bytes (PNG/JPEG).
                      Messages may include 'audio' key with list of raw bytes
                      (WAV/MP3/FLAC/OGG/AIFF/AAC).
            tools: Optional tool specifications.
        """
        contents = []
        system_parts = []
        for m in messages:
            role = m['role']
            if role == 'system':
                system_parts.append({"text": m['content']})
                continue
            if role == 'tool':
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {
                        "name": m['name'],
                        "response": {"result": m['content']}
                    }}]
                })
                continue
            if role == 'assistant':
                parts = []
                if m.get('content'):
                    parts.append({"text": m['content']})
                if 'tool_calls' in m:
                    for tc in m['tool_calls']:
                        part = {"functionCall": {
                            "name": tc['function']['name'],
                            "args": json.loads(tc['function']['arguments'])
                        }}
                        if sig := tc.get('thoughtSignature'):
                            part['thoughtSignature'] = sig
                        parts.append(part)
                contents.append({"role": "model", "parts": parts})
                continue
            # role == 'user'
            parts = []
            if m.get('content'):
                parts.append({"text": m['content']})
            if images := m.pop('images', None):
                for img in images:
                    parts.append({"inlineData": {
                        "mimeType": MEDIA_TYPES[img[:3]],
                        "data": base64.b64encode(img).decode()
                    }})
            if audio := m.pop('audio', None):
                for aud in audio:
                    parts.append({"inlineData": {
                        "mimeType": _detect_audio_type(aud),
                        "data": base64.b64encode(aud).decode()
                    }})
            contents.append({"role": "user", "parts": parts})
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
        conn = http.client.HTTPSConnection(self.model_config['host'], timeout=self.timeout)
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
            throttle(self.model_config['host'], self.model_config.get('tpm', 5))
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
            if not response_json.get('candidates'):
                raise Exception(f"candidates missing from response: {response_json}")
            candidate = response_json['candidates'][0]
            content = ""
            tool_calls = []
            for part in candidate.get('content', {}).get('parts', []):
                if 'text' in part:
                    content += part['text']
                elif 'functionCall' in part:
                    fc = part['functionCall']
                    tc = {
                        'id': f"gemini_{fc['name']}",
                        'function': {
                            'name': fc['name'],
                            'arguments': json.dumps(fc['args'])
                        }
                    }
                    if sig := part.get('thoughtSignature'):
                        tc['thoughtSignature'] = sig
                    tool_calls.append(tc)
            message = {'role': 'assistant', 'content': content}
            message['_stop_reason'] = candidate.get('finishReason')
            if tool_calls:
                message['tool_calls'] = tool_calls
            return message
        finally:
            conn.close()

    def prepare_message(self, m):
        # Tool call emulation
        if 'tool_calls' in m:
            tool_calls_str = json.dumps({ "function_calls": [ {
                "name": tc['function']['name'],
                "arguments": json.loads(tc['function']['arguments']),
            } for tc in m['tool_calls'] ] }, indent=JSON_INDENT)
            return {
                'role': 'assistant',
                'content': f"{m['content'] or ''}\n{tool_calls_str}".strip(),
                **{k: v for k, v in m.items() if k in EXTRA_KEYS}
            }
        elif m['role'] == 'tool':
            return {
                'role': 'user',
                'content': f"{m['name']}: {m['content']}",
                **{k: v for k, v in m.items() if k in EXTRA_KEYS}
            }
        else:
            return m

    @staticmethod
    def _strip_orphaned_tool_use(messages):
        """Remove tool_use blocks that have no matching tool_result, and drop
        any assistant messages that become empty as a result.  This prevents
        API errors when a prior tool-call loop was interrupted mid-execution."""
        tool_result_ids = {m['tool_call_id'] for m in messages
                          if m.get('role') == 'tool' and 'tool_call_id' in m}
        out = []
        for msg in messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                kept = [tc for tc in msg['tool_calls']
                        if tc.get('id', '') in tool_result_ids]
                if kept or msg.get('content'):
                    msg = {**msg, 'tool_calls': kept} if kept else \
                          {k: v for k, v in msg.items() if k != 'tool_calls'}
                    out.append(msg)
            else:
                out.append(msg)
        return out

    def _call(self, messages, tools=None):
        if not self.native:
            messages = [ self.prepare_message(msg) for msg in messages ]
        # Strip internal metadata keys (underscore-prefixed) before sending to API
        messages = [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]
        # Drop orphaned tool_use blocks (from interrupted tool-call loops)
        messages = self._strip_orphaned_tool_use(messages)
        if self.model_config['api_type'] == "completions":
            return self._call_completions(messages, tools)
        elif self.model_config['api_type'] == "messages":
            return self._call_messages(messages, tools)
        elif self.model_config['api_type'] == "gemini":
            return self._call_gemini(messages, tools)
        else:
            raise NotImplementedError(self.model_config['api_type'])

    @staticmethod
    def _sleep_backoff(attempt, base=15):
        """
        Exponential back-off helper. Sleeps for `base * 2**attempt` seconds.
        """
        time.sleep(base * (2 ** attempt))

    def text_call(self, messages, retry=3, attempt=0):
        try:
            with self.concurrency_lock:
                return self._call(messages)
        except Exception as e:
            err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
            logger.error(f"text_call {type(e).__name__}: {err} (line {sys.exc_info()[2].tb_lineno})")
            if retry:
                self._sleep_backoff(attempt)
                return self.text_call(messages, retry-1, attempt+1)
            raise

    def tool_call_native(self, messages, tools, retry=5):
        if self.model_config['api_type'] == "gemini":
            for tool in tools.values():
                schema = tool.model_json_schema()
                if _gemini_schema_has_unsupported_fieldtypes(schema):
                    logger.info("Gemini native tool calling fallback to shim mode due to unsupported schema field types")
                    return self.tool_call_shim(messages, tools, retry=retry)

        _tools = []
        for name, tool in tools.items():
            schema = tool.model_json_schema()
            schema.pop('title', None)
            _tools.append({
                'type': 'function',
                'function': {
                    'description': schema.pop('description'),
                    'name': name,
                    'parameters': schema
                }
            })
        
        for attempt in range(retry + 1):
            try:
                with self.concurrency_lock:
                    resp_msg = self._call(messages, _tools)
                if transform := self.model_config.get('response_transform'):
                    resp_msg = transform(resp_msg, tools)
                stop = resp_msg.get('_stop_reason')
                if stop in ('max_tokens', 'length', 'MAX_TOKENS'):
                    content = resp_msg.get('content', '')
                    raise MaxTokensError(f"stop_reason={stop}, content={content[:500]}")
                if not resp_msg.get("tool_calls"):
                    content = resp_msg.get('content', '')
                    err = f"tool_calls missing (stop_reason={stop}): {content[:1000]}{'...' if len(content) > 1000 else ''}"
                    raise ValidationError(err)
                
                # Validate tool calls against schemas
                for tool_call in resp_msg.get("tool_calls", []):
                    name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Failed to decode arguments JSON for tool '{name}': {e}")
                    
                    if not name in tools:
                        raise ValidationError(f"Unknown tool '{name}'")
                    
                    tool = tools[name]
                    if not isinstance(arguments, dict):
                        raise ValidationError(f"Arguments for '{name}' are not a dict")
                    
                    try:
                        tool.model_validate(arguments)
                    except Exception as ve:
                        error_msg = f"Invalid arguments for tool '{name}': {ve}"
                        if attempt < retry:
                            if logger.isEnabledFor(logging.INFO):
                                logger.info(f"ValidationError: {ve}, retry {attempt+1}/{retry}")
                            raise ValidationError(error_msg)
                        else:
                            raise ValidationError(error_msg)
                
                return resp_msg
                
            except ValidationError as e:
                if attempt < retry:
                    logger.info(f"ValidationError: {e}, retry {attempt+1}/{retry}")
                    continue
                raise
                
            except (BadRequestError, MaxTokensError):
                raise
            except Exception as e:
                err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
                logger.error(f"tool_call_native {type(e).__name__}: {err}")
                if attempt < retry:
                    self._sleep_backoff(attempt)
                    continue
                raise

    def tool_call_shim(self, messages, tools, retry=3, attempt=0, _feedback=None):
        _tools = []
        for name, tool in tools.items():
            schema = tool.model_json_schema()
            _tools.append({
                'type': 'function',
                'function': {
                    'description': schema.pop('description'),
                    'name': name,
                    'parameters': schema
                }
            })
        instructions = (
            "### SYSTEM NOTICE ###\n"
            "Available functions:\n"
            f"{json.dumps(_tools, indent=JSON_INDENT)}\n\n"
            'You MUST respond ONLY with a JSON object containing a key "function_calls" which'
            ' is an ARRAY of one or more function calls needed to fulfill the request. Each element in'
            ' the array should be a JSON object with "name" and "arguments" keys. If multiple'
            ' calls are needed, include multiple objects in the array.\n\n'
            'Example Response Format:'
            '{\n'
            '  "function_calls": [\n'
            '    {\n'
            '      "name": "function_name_1",\n'
            '      "arguments": { "arg1": "value1", ... }\n'
            '    },\n'
            '    {\n'
            '      "name": "function_name_2",\n'
            '      "arguments": { "arg1": "value1", ... }\n'
            '    }\n'
            '    // ... more calls if needed\n'
            '  ]\n'
            '}"\n'
        )
        if JSON_INDENT is None:
            instructions = instructions.replace('\n',' ')
            while '  ' in instructions:
                instructions = instructions.replace('  ',' ')
        _messages = messages.copy()
        if _messages[-1]['role'] == "user":
            _messages[-1]['content']+= f"\n\n{instructions}"
        else:
            _messages.append({"role": "user", "content": instructions})
        if _feedback:
            _messages.extend(_feedback)
        try:
            with self.concurrency_lock:
                resp_msg = self._call(_messages)
        except Exception as e:
            err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
            logger.error(f"tool_call_shim {type(e).__name__}: {err}")
            if retry:
                self._sleep_backoff(attempt)
                return self.tool_call_shim(messages, tools, retry-1, attempt+1)
            raise
        try:
            content = resp_msg.get('content') or resp_msg.get('reasoning_content') or ''
            content = _preprocess_tool_call_response(content)
            tool_calls, json_start_index, json_end_index = _extract_tool_calls_json(content)
            if not (function_calls := tool_calls["function_calls"]):
                raise ValidationError(f"Function calls are required.")
            for call in function_calls:
                name = call["name"]
                arguments = call["arguments"]
                tool = tools[name]
                if not isinstance(arguments, dict):
                    logger.debug(json.dumps(tool_calls, indent=4))
                    raise ValidationError(f"Arguments for '{name}' are not a dict")
                try:
                    tool.model_validate(arguments)
                except Exception as ve:
                    raise ValidationError(f"Invalid arguments for tool '{name}': {ve}")
            resp_msg = {
                'role': 'assistant',
                'tool_calls': [
                    {'function': {'name': row['name'], 'arguments': json.dumps(row['arguments'])}}
                    for row in tool_calls['function_calls']
                ],
                'content': (content[:json_start_index] + content[json_end_index+1:]).strip('`').removeprefix('json').strip()
            }
            return resp_msg
        except (ValidationError,KeyError) as e:
            if retry:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"ValidationError: {e}, retry")
                feedback = list(_feedback or [])
                failed_content = resp_msg.get('content') or resp_msg.get('reasoning_content') or ''
                if failed_content:
                    feedback.append({"role": "assistant", "content": failed_content})
                feedback.append({
                    "role": "user",
                    "content": f"ERROR: {e}. You MUST respond ONLY with a JSON object containing \"function_calls\"."
                })
                return self.tool_call_shim(messages, tools, retry-1, _feedback=feedback)
            raise


    def call(self, messages, tools=None):
        if tools is None:
            return self.text_call(messages)
        elif self.native:
            return self.tool_call_native(messages, tools)
        else:
            return self.tool_call_shim(messages, tools)


    def conversation(self, system_prompt):
        return Conversation(self, system_prompt)
