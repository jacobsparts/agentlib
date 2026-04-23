
from .basic import TASKS as BASIC_TASKS
from .behavior import TASKS as BEHAVIOR_TASKS

TASKS = BASIC_TASKS + BEHAVIOR_TASKS

__all__ = ["TASKS", "BASIC_TASKS", "BEHAVIOR_TASKS"]
