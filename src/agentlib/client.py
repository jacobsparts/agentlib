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
EXTRA_KEYS = {'images'}

MEDIA_TYPES = {
    b'\xff\xd8\xff': "image/jpeg",
    b'\x89PN': "image/png",
}

logger = logging.getLogger('agentlib')

class ValidationError(Exception): pass
class BadRequestError(Exception): pass


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
                "parameters": t['function']['parameters'],
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

    def _call(self, messages, tools=None):
        if not self.native:
            messages = [ self.prepare_message(msg) for msg in messages ]
        # Strip internal metadata keys (underscore-prefixed) before sending to API
        messages = [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]
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
                if not resp_msg.get("tool_calls"):
                    content = resp_msg.get('content', '')
                    err = f"tool_calls missing from response: {content[:1000]}{'...' if len(content) > 1000 else ''}"
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
                
            except Exception as e:
                err = (str(e) if len(str(e)) < 1000 else str(e)[:1000]+'...').replace("\n"," ")
                logger.error(f"tool_call_native {type(e).__name__}: {err}")
                if attempt < retry:
                    self._sleep_backoff(attempt)
                    continue
                raise

    def tool_call_shim(self, messages, tools, retry=3, attempt=0):
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
            content = resp_msg['content']
            json_start_index = content.find('{')
            if json_start_index == -1:
                raise ValidationError("No JSON object found (missing '{').")
            json_end_index = content.rfind('}')
            if json_end_index == -1:
                raise ValidationError("Found '{' but no corresponding closing '}' found afterwards.")
            try:
                tool_calls = json.loads(content[json_start_index:json_end_index+1])
            except json.JSONDecodeError as e:
                raise ValidationError(f"Failed to decode JSON: {e}")
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
                    error_msg = f"Invalid arguments for tool '{name}': {ve}"
                    # Add validation error message to the messages list
                    messages.append({
                        "role": "system", 
                        "content": f"VALIDATION ERROR: {error_msg}"
                    })
                    raise ValidationError(error_msg)
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
                return self.tool_call_shim(messages, tools, retry-1)
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
