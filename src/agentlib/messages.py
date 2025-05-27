import json
import http.client
import time
import logging

logger = logging.getLogger('agentlib')

def _call_messages(self, messages, tools):
    # Extract system message if present
    system_message = None
    filtered_messages = []
    
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        else:
            filtered_messages.append(msg)
    
    req = {
        "model": self.model_config['model'],
        "messages": filtered_messages,
        "max_tokens": self.model_config.get('config', {}).get('max_tokens', 4096),
        **{k: v for k, v in self.model_config.get('config', {}).items() if k != 'max_tokens'}
    }
    
    if system_message:
        req["system"] = system_message
        
    if tools:
        req.update({
            "tools": tools,
            "tool_choice": {"type": "any"},
        })
        
    if self.model_config['port'] == 443:
        conn = http.client.HTTPSConnection(self.model_config['host'], timeout=self.timeout)
    else:
        conn = http.client.HTTPConnection(self.model_config['host'], self.model_config['port'], timeout=self.timeout)
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.model_config['api_key']}",
        "anthropic-version": "2023-06-01",
    }
    
    body = json.dumps(req)
    
    try:
        from .utils import throttle
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
            from .client import BadRequestError
            raise BadRequestError(response_data.strip())
        elif response.status != 200:
            raise Exception(f"API Error {response.status}: {response_data}")

        response_json = json.loads(response_data)
        if usage := response_json.get('usage'):
            self.usage_tracker.log(self.model_name, usage)
            
        # Convert Anthropic response format to OpenAI-compatible format
        content = ""
        tool_calls = []
        
        for content_block in response_json.get('content', []):
            if content_block['type'] == 'text':
                content += content_block['text']
            elif content_block['type'] == 'tool_use':
                tool_calls.append({
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
