"""
Example of custom provider and model registration with inline key
"""
import os
import agentlib

os.environ['CLIPROXYAPI_API_KEY'] = 'sk-YOUR-API-KEY'
agentlib.register_provider(
    "cliproxyapi",
    host="localhost",
    port=8317,
    path="/v1/chat/completions",
    tpm=1/3,
    concurrency=1,
    timeout=None,
    tools=True,
)
agentlib.register_model("cliproxyapi","opus",
    model="claude-opus-4-5-20251101",
    config={
        "include_reasoning": "true",
        "reasoning_effort": "high",
        "stream": False
    },
    input_cost=0.0,
    cached_cost=0.0,
    output_cost=0.0,
    reasoning_cost=0.0,
)
