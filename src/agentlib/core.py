import sys
assert sys.version_info >= (3, 8), "Requires Python 3.8+"
import logging

logger = logging.getLogger('agentlib')
handler = logging.StreamHandler()
logger.addHandler(handler)

import pydantic
if pydantic.__version__ < '2':
    from . import pydantic_patch

from .utils import JSON_INDENT
from .client import LLMClient, ValidationError, BadRequestError
from .conversation import Conversation
from .agent import BaseAgent
