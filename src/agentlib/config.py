"""User configuration loading for agentlib."""

import importlib.util
import sys
from pathlib import Path

_CONFIG_NOT_FOUND = object()
_user_config = None
_config_spec = None

def get_config_spec():
    global _config_spec, _user_config
    
    if _user_config is not None:
        return None, _user_config
    
    if _config_spec is not None:
        return _config_spec
    
    config_path = Path.home() / ".agentlib" / "config.py"
    if not config_path.exists():
        _user_config = _CONFIG_NOT_FOUND
        return None, None
    
    try:
        spec = importlib.util.spec_from_file_location("agentlib_user_config", config_path)
        if spec and spec.loader:
            user_config = importlib.util.module_from_spec(spec)
            _config_spec = (spec, user_config)
            return spec, user_config
    except Exception as e:
        print(f"Warning: Failed to create config spec from {config_path}: {e}", file=sys.stderr)
    
    return None, None

def get_user_config():
    global _user_config
    
    if _user_config is not None:
        return None if _user_config is _CONFIG_NOT_FOUND else _user_config
    
    spec, module = get_config_spec()
    if spec is None:
        return None if module is _CONFIG_NOT_FOUND else module
    
    try:
        spec.loader.exec_module(module)
        _user_config = module
        return module
    except Exception as e:
        config_path = Path.home() / ".agentlib" / "config.py"
        print(f"Warning: Failed to load user config from {config_path}: {e}", file=sys.stderr)
        _user_config = _CONFIG_NOT_FOUND
        return None
