import atexit
import os
import threading
import time
import logging
from collections import defaultdict
from .llm_registry import get_model_config

logger = logging.getLogger('agentlib')

JSON_INDENT = None

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
        self.history = []
        if not os.getenv("AGENTLIB_SUPPRESS_USAGE_ATEXIT"):
            atexit.register(self.print_stats)

    def log(self, model_name, usage):
        with self.lock:
            self.history.append((model_name, usage))

    def _coalesce_paths(self, obj, paths):
        for path in paths:
            if isinstance(path, str):
                path = (path,)
            value = obj
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None:
                return value
        return 0

    def _normalize(self, model_name, usage):
        model_config = get_model_config(model_name)
        if transform := model_config.get('token_transform'):
            usage = transform(usage)
        # Anthropic uses input_tokens/output_tokens; OpenAI uses prompt_tokens/completion_tokens
        cached_tokens = self._coalesce_paths(usage, [
            'native_tokens_cached',
            ('prompt_tokens_details', 'cached_tokens'),
        ])
        cached_tokens += usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
        prompt_tokens = self._coalesce_paths(usage, [
            'native_tokens_prompt',
            'prompt_tokens',
            'input_tokens',
        ])
        prompt_tokens -= cached_tokens
        if prompt_tokens < 0:
            logger.warning(f"⚠️ Negative prompt token count: {usage}")
            return {'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0, 'cost': 0.0}
        reasoning_tokens = self._coalesce_paths(usage, [
            'native_tokens_reasoning',
            ('completion_tokens_details', 'reasoning_tokens'),
        ])
        completion_tokens = self._coalesce_paths(usage, [
            'native_tokens_completion',
            'completion_tokens',
            'output_tokens',
        ])
        if total := usage.get('total_tokens'):
            if reasoning_tokens > 0:
                completion_tokens = total - (prompt_tokens + cached_tokens + reasoning_tokens)
            else:
                reasoning_tokens = total - (prompt_tokens + cached_tokens + completion_tokens)
            if not prompt_tokens + cached_tokens + completion_tokens + reasoning_tokens == total or not completion_tokens >= 0 or not reasoning_tokens >= 0:
                tokens = {
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "completion_tokens": completion_tokens,
                }
                logger.warning(f"⚠️ Tokens don't add up: {usage} -> {tokens}")
        input_rate = model_config.get('input_cost') or 0
        cached_rate = model_config.get('cached_cost') if model_config.get('cached_cost') is not None else input_rate
        output_rate = model_config.get('output_cost') or 0
        reasoning_rate = model_config.get('reasoning_cost') if model_config.get('reasoning_cost') is not None else output_rate
        input_cost = prompt_tokens * (input_rate / 1000000.0)
        cached_cost = cached_tokens * (cached_rate / 1000000.0)
        output_cost = completion_tokens * (output_rate / 1000000.0)
        reasoning_cost = reasoning_tokens * (reasoning_rate / 1000000.0)
        if cost_transform := model_config.get('cost_transform'):
            input_cost, cached_cost, output_cost, reasoning_cost = cost_transform(
                prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens,
                input_cost, cached_cost, output_cost, reasoning_cost,
            )
        return {
            'prompt_tokens': prompt_tokens,
            'cached_tokens': cached_tokens,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'cost': input_cost + cached_cost + output_cost + reasoning_cost,
        }

    @property
    def model_usage(self):
        totals = defaultdict(lambda: {
            'prompt_tokens': 0, 'cached_tokens': 0,
            'completion_tokens': 0, 'reasoning_tokens': 0, 'cost': 0.0
        })
        for model_name, usage in self.history:
            normalized = self._normalize(model_name, usage)
            for k in normalized:
                totals[model_name][k] += normalized[k]
        return totals

    def print_stats(self):
        for model_name, usage in self.model_usage.items():
            parts = [ part for part in [
                f"In={usage['prompt_tokens']}" if usage['prompt_tokens'] else None,
                f"Cached={usage['cached_tokens']}" if usage['cached_tokens'] else None,
                f"Rsn={usage['reasoning_tokens']}" if usage['reasoning_tokens'] else None,
                f"Out={usage['completion_tokens']}" if usage['completion_tokens'] else None,
                f"Cost=${usage['cost']:.3f}" if usage['cost'] else None
            ] if part ]
            if parts:
                print(f"{model_name}: {', '.join(parts)}")

