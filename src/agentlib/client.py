import sys
assert sys.version_info >= (3, 8), "Requires Python 3.8+"
import os
import json
import http.client
import urllib.parse
import threading
import time
import logging
from collections import defaultdict

from .utils import throttle, JSON_INDENT, UsageTracker
from .llm_registry import get_model_config

logger = logging.getLogger('agentlib')

class ValidationError(Exception): pass
class BadRequestError(Exception): pass


class LLMClient:
    usage_tracker = UsageTracker()
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.timeout = self.model_config.get('timeout', 300)
        self.concurrency_lock = threading.BoundedSemaphore(self.model_config.get('concurrency',10))

    def _call_completions(self, messages, tools):
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
                logger.warning("Throttled. Waiting 20s")
                with throttle(self.model_config['host']):
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
            return response_json['choices'][0]['message']
        finally:
            conn.close()

    def prepare_message(self, m):
        # Tool call emulation
        if 'tool_calls' in m:
            tool_calls_str = json.dumps({ "function_calls": [ {
                "name": tc['function']['name'],
                "arguments": json.loads(tc['function']['arguments']),
            } for tc in m['tool_calls'] ] }, indent=JSON_INDENT)
            return {'role': 'assistant', 'content': f"{m['content']}\n{tool_calls_str}".strip()}
        elif m['role'] == 'tool':
            return {'role': 'user', 'content': f"{m['name']}: {m['content']}"}
        else:
            return m

    def _call(self, messages, tools=None):
        if not self.model_config.get('tools'):
            messages = [ self.prepare_message(msg) for msg in messages ]
        if self.model_config['api_type'] == "completions":
            return self._call_completions(messages, tools)
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
            logger.error(f"text_call {type(e).__name__}: {err}")
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
        elif self.model_config.get('tools'):
            return self.tool_call_native(messages, tools)
        else:
            return self.tool_call_shim(messages, tools)


    def conversation(self, system_prompt):
        from .conversation import Conversation
        return Conversation(self, system_prompt)
