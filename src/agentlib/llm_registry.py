import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ProviderConfig:
    provider: str
    host: str
    path: str
    port: int = 443
    tpm: int = 60
    concurrency: int = 5
    timeout: int = 120
    tools: bool = False
    api_type: str = "completions"

@dataclass
class ModelConfig:
    model: str
    provider: ProviderConfig
    config: dict = field(default_factory=dict)
    input_cost: float = None
    output_cost: float = None
    cached_cost: float = None
    reasoning_cost: float = None
    timeout: int = None
    tools: bool = None

class EndpointRegistry:
    def __init__(self):
        self._models = {}
        self._providers = {}

    def register_provider(self, name, **kwargs):
        kwargs['provider'] = name
        self._providers[name] = ProviderConfig(**kwargs)

    def register_model(self, provider, alias, **kwargs):
        if not (prov_obj := self._providers.get(provider)):
            raise ValueError(f"unknown provider: {provider}")
        kwargs.setdefault('model', alias)
        self._models[f"{provider}/{alias}"] = ModelConfig(provider=prov_obj, **kwargs)

    def get_model_config(self, name):
        _model = dict(self._models[name].__dict__)
        _provider = _model.pop('provider').__dict__
        keys = _model.keys() | _provider.keys()
        model_config = { k: v if (v := _model.get(k)) is not None else _provider.get(k) for k in keys }
        env_var = f"{model_config['provider'].upper()}_API_KEY"
        if not (api_key := os.getenv(env_var)):
            raise Exception(f"{env_var} is not set.")
        return {**model_config, 'api_key': api_key}

registry = EndpointRegistry()
register_provider = registry.register_provider
register_model = registry.register_model
get_model_config = registry.get_model_config

# --- OpenAI ---
register_provider("openai",
    host="api.openai.com",
    path="/v1/chat/completions",
    tpm=100,
    concurrency=30,
    timeout=300,
    tools=True,
    api_type="completions",
)
register_model("openai","o4-mini",
    model="o4-mini",
    config={"reasoning_effort": "high", "service_tier": "flex"},
    input_cost=0.55,
    output_cost=2.2,
    cached_cost=0.275,
    timeout=900,
)
register_model("openai","o4-mini-medium",
    config={"reasoning_effort": "medium", "service_tier": "flex"},
)
register_model("openai","gpt-4.1",
    model="gpt-4.1",
    input_cost=2.0,
    output_cost=8.0,
)
register_model("openai","gpt-4.1-mini",
    model="gpt-4.1-mini",
    input_cost=0.4,
    output_cost=1.6,
)
register_model("openai","gpt-4.1-nano",
    model="gpt-4.1-nano",
    input_cost=0.1,
    output_cost=0.4,
)

# --- Anthropic ---
register_provider("anthropic",
    host="api.anthropic.com",
    path="/v1/messages",
    tpm=100,
    concurrency=30,
    timeout=300,
    tools=True,
    api_type="messages"
)
register_model("anthropic","claude-sonnet-4",
    model="claude-sonnet-4-20250514",
    input_cost=3.00,
    cached_cost=0.3,
    output_cost=15.0,
)

# --- Google ---
register_provider("google",
    host="generativelanguage.googleapis.com",
    path="/v1beta/openai/chat/completions",
    tpm=5,
    concurrency=3,
    timeout=None,
    tools=True,
    api_type="completions",
)
register_model("google","gemini-2.5-flash",
    model="gemini-2.5-flash-preview-05-20",
    config={"reasoning_effort": "high"},
    input_cost=0.15,
    cached_cost=0.025,
    output_cost=0.6,
    reasoning_cost=3.5,
)
register_model("google","gemini-2.0-flash",
    model="gemini-2.0-flash",
    input_cost=0.1,
    cached_cost=0.025,
    output_cost=0.4,
)

# --- X.AI ---
register_provider("xai",
    host="api.x.ai",
    path="/v1/chat/completions",
    tpm=50,
    concurrency=50,
    timeout=300,
    tools=True,
    api_type="completions",
)
register_model("xai","grok-3",
    model="grok-3",
    input_cost=3.0,
    output_cost=15.0,
)
register_model("xai","grok-3-mini",
    model="grok-3-mini",
    config={"reasoning_effort": "high"},
    input_cost=0.30,
    output_cost=0.50,
)

# --- OpenRouter ---
register_provider("openrouter",
    host="openrouter.ai",
    path="/api/v1/chat/completions",
    timeout=300,
    tools=None,
    api_type="completions",
)
lambda_config = {'provider': {'order': ['Lambda'], 'allow_fallbacks': False}}
novita_config = {'provider': {'order': ['Novita'], 'allow_fallbacks': False}}
register_model("openrouter","deepseek-r1",
    model="deepseek/deepseek-r1",
    config=lambda_config,
    input_cost=0.54,
    output_cost=2.18,
    tools=True,
)
register_model("openrouter","deepseek-v3-0324",
    model="deepseek/deepseek-chat-v3-0324",
    config=lambda_config,
    input_cost=0.34,
    output_cost=0.88,
    tools=True,
)
register_model("openrouter","qwen3-235b-a22b",
    model="qwen/qwen3-235b-a22b",
    config=novita_config,
    input_cost=0.1,
    output_cost=0.1,
    tools=False,
)
