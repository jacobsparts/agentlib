import threading
import time
import logging
from collections import defaultdict

logger = logging.getLogger('agentlib')

JSON_INDENT = 2

throttle_lock = defaultdict(threading.BoundedSemaphore)
throttle_last = defaultdict(float)
def throttle(name='default', tps=5):
    with throttle_lock[name]:
        now = time.monotonic()
        next_time = throttle_last[name] + 1 / tps
        if sleep := max(next_time - now, 0):
            time.sleep(sleep)
        throttle_last[name] = time.monotonic()


class UsageTracker:
    lock = threading.BoundedSemaphore()
    def __init__(self):
        self.model_usage = defaultdict(lambda: {
            'prompt_tokens': 0,
            'cached_tokens': 0,
            'completion_tokens': 0,
            'reasoning_tokens': 0,
            'cost': 0.0
        })

    def log(self, model_name, usage):
        with self.lock:
            from .llm_registry import get_model_config
            model_config = get_model_config(model_name)
            cached_tokens = (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
            prompt_tokens = usage.get('prompt_tokens', 0) - cached_tokens
            reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
            completion_tokens = usage.get('completion_tokens', 0)
            if 'gemini' in model_name:
                reasoning_tokens = max(usage.get('total_tokens', 0) - (prompt_tokens + completion_tokens), 0)
            elif not 'grok' in model_name:
                completion_tokens -= reasoning_tokens
            if 'total_tokens' in usage:
                if not prompt_tokens + cached_tokens + completion_tokens + reasoning_tokens == usage['total_tokens']:
                    logger.warning("⚠️ Tokens don't add up", usage)
            self.model_usage[model_name]['prompt_tokens'] += prompt_tokens
            self.model_usage[model_name]['cached_tokens'] += cached_tokens
            self.model_usage[model_name]['completion_tokens'] += completion_tokens
            self.model_usage[model_name]['reasoning_tokens'] += reasoning_tokens 
            input_cost = prompt_tokens * (model_config.get('input_cost',0) / 1000000.0)
            cached_cost = cached_tokens * ((model_config.get('cached_cost') or input_cost) / 1000000.0)
            output_cost = completion_tokens * (model_config.get('output_cost',0) / 1000000.0)
            reasoning_cost = reasoning_tokens * ((model_config.get('reasoning_cost', model_config.get('output_cost')) or output_cost) / 1000000.0)
            if model_name.startswith('gemini-2.5-pro') and prompt_tokens > 200000:
                input_cost *= 2
                output_cost *= 1.5
                reasoning_cost *= 1.5
            self.model_usage[model_name]['cost'] += input_cost + output_cost + reasoning_cost

    def print_stats(self):
        for model_name, usage in self.model_usage.items():
            parts = [ part for part in [
                f"In={usage['prompt_tokens']}" if usage['prompt_tokens'] else None,
                f"Cached={usage['cached_tokens']}" if usage['cached_tokens'] else None,
                f"Out={usage['completion_tokens']}" if usage['completion_tokens'] else None,
                f"Rsn={usage['reasoning_tokens']}" if usage['reasoning_tokens'] else None,
                f"Cost=${usage['cost']:.3f}" if usage['cost'] else None
            ] if part ]
            if parts:
                print(f"{model_name}: {', '.join(parts)}")
    
    def __del__(self):
        self.print_stats()
