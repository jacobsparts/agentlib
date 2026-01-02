import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

class ModelNotFoundError(Exception):
    """Raised when an unknown model is requested."""
    pass

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
        self._aliases = {}

    def register_provider(self, name, **kwargs):
        kwargs['provider'] = name
        self._providers[name] = ProviderConfig(**kwargs)

    def register_model(self, provider, alias, aliases=None, **kwargs):
        if not (prov_obj := self._providers.get(provider)):
            raise ValueError(f"unknown provider: {provider}")
        kwargs.setdefault('model', alias)
        full_name = f"{provider}/{alias}"
        self._models[full_name] = ModelConfig(provider=prov_obj, **kwargs)
        if aliases:
            for a in (aliases if isinstance(aliases, list) else [aliases]):
                self._aliases[a] = full_name

    def resolve_model_name(self, name):
        """Resolve an alias or short name to the full model name (provider/model).
        Returns the input unchanged if not an alias."""
        return self._aliases.get(name, name)

    def get_model_config(self, name):
        # Resolve alias if it exists
        resolved_name = self._aliases.get(name, name)
        
        # Check if model exists
        if resolved_name not in self._models:
            # Provide helpful error message
            available = list(self._models.keys()) + list(self._aliases.keys())
            raise ModelNotFoundError(
                f"Unknown model '{name}'. Available models and aliases:\n" +
                "\n".join(f"  - {m}" for m in sorted(available))
            )
        
        _model = dict(self._models[resolved_name].__dict__)
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
resolve_model_name = registry.resolve_model_name

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
register_model("openai","gpt-5.2",
    model="gpt-5.2",
    input_cost=1.75,
    cached_cost=0.175,
    output_cost=14.0,
)
register_model("openai","gpt-5.1",
    model="gpt-5.1",
    input_cost=1.25,
    cached_cost=0.125,
    output_cost=10.0,
    config={"reasoning_effort": "none"},
)
register_model("openai","gpt-5-mini",
    model="gpt-5-mini",
    aliases="mini",
    input_cost=0.25,
    cached_cost=0.025,
    output_cost=2.0,
    config={"reasoning_effort": "high"},
)
register_model("openai","gpt-5-mini-flex",
    model="gpt-5-mini",
    input_cost=0.125,
    cached_cost=0.013,
    output_cost=1.0,
    config={"reasoning_effort": "high", "service_tier": "flex"},
    timeout=1200,
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
register_model("anthropic","claude-haiku-4-5",
    model="claude-haiku-4-5",
    aliases="haiku",
    input_cost=1.00,
    cached_cost=0.1,
    output_cost=5.0,
)
register_model("anthropic","claude-sonnet-4-5",
    model="claude-sonnet-4-5",
    aliases="sonnet",
    input_cost=3.00,
    cached_cost=0.3,
    output_cost=15.0,
)
register_model("anthropic","claude-opus-4-5",
    model="claude-opus-4-5",
    aliases="opus",
    input_cost=5.00,
    cached_cost=0.5,
    output_cost=25.0,
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
register_model("google","gemini-3-pro",
    model="gemini-3-pro-preview",
    aliases="pro",
    config={"reasoning_effort": "high"},
    input_cost=2.00,
    cached_cost=0.2,
    output_cost=12.00,
    reasoning_cost=12.00,
)
register_model("google","gemini-3-flash-preview",
    model="gemini-3-flash-preview",
    aliases="flash",
    config={"reasoning_effort": "high"},
    input_cost=0.5,
    cached_cost=0.05,
    output_cost=3.0,
    reasoning_cost=3.0,
)
register_model("google","gemini-2.5-pro",
    model="gemini-2.5-pro",
    config={"reasoning_effort": "high"},
    input_cost=1.25,
    cached_cost=0.125,
    output_cost=10.00,
    reasoning_cost=10.00,
)
register_model("google","gemini-2.5-flash",
    model="gemini-2.5-flash",
    config={"reasoning_effort": "high"},
    input_cost=0.3,
    cached_cost=0.03,
    output_cost=2.5,
    reasoning_cost=2.5,
)

# --- X.AI ---
register_provider("xai",
    host="api.x.ai",
    path="/v1/chat/completions",
    tpm=50,
    concurrency=50,
    timeout=300,
    tools=False,
    api_type="completions",
)
register_model("xai","grok-4-1",
    model="grok-4-1-fast-reasoning",
    input_cost=0.2,
    cached_cost=0.05,
    output_cost=0.5,
)
register_model("xai","grok-code-fast-1",
    model="grok-code-fast-1",
    aliases="grok-code",
    input_cost=0.2,
    cached_cost=0.02,
    output_cost=1.5,
)

# --- OpenRouter ---
register_provider("openrouter",
    host="openrouter.ai",
    path="/api/v1/chat/completions",
    timeout=300,
    tools=None,
    api_type="completions",
)
zai_config = {'provider': {'order': ['z-ai'], 'allow_fallbacks': False}, 'temperature': 0}
register_model("openrouter","glm-4.7",
    model="z-ai/glm-4.7",
    aliases="glm",
    config=zai_config,
    input_cost=0.4,
    output_cost=1.5,
    tools=False,
)

# --- User Configuration ---
from agentlib.config import get_config_spec, get_user_config

def load_user_config():
    """Load user's custom model configurations from ~/.agentlib/config.py"""
    # Get spec and module without executing
    spec, user_config = get_config_spec()
    
    if spec is None and user_config is None:
        # Config doesn't exist
        return
    
    if spec is None:
        # Already loaded by someone else
        return
    
    # Inject registry functions before execution
    user_config.register_provider = register_provider
    user_config.register_model = register_model
    user_config.registry = registry
    
    # Now execute with injected functions
    get_user_config()

load_user_config()
