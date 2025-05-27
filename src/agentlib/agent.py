import json
import inspect
import types
import textwrap
import logging
from typing import Literal

import pydantic
from pydantic import BaseModel, Field, create_model

from .client import LLMClient

logger = logging.getLogger('agentlib')

class AgentMeta(type):
    def __new__(mcls, name, bases, clsdict):
        local_tools = {}
        local_specs = {}

        for attr_name, attr_value in clsdict.items():
            if callable(attr_value) and hasattr(attr_value, '_tool_name'):
                tool_name = getattr(attr_value, '_tool_name')
                tool_spec = getattr(attr_value, '_tool_spec')
                local_tools[tool_name] = attr_value
                local_specs[tool_name] = tool_spec
                delattr(attr_value, '_tool_name')
                delattr(attr_value, '_tool_spec')

        cls = super().__new__(mcls, name, bases, clsdict)

        final_tool_registry = {}
        for base in reversed(cls.__mro__[1:]):
            if base_registry := getattr(base, '_toolimpl', None):
                final_tool_registry.update(base_registry)
        final_tool_registry.update(local_tools)
        cls._toolimpl = final_tool_registry

        final_spec_registry = {}
        for base in reversed(cls.__mro__[1:]):
            if base_specs := getattr(base, '_toolspec', None):
                final_spec_registry.update(base_specs)
        final_spec_registry.update(local_specs)
        cls._toolspec = final_spec_registry

        return cls


class BaseAgent(metaclass=AgentMeta):
    
    class TurnLimitError(Exception): pass

    def tool(_input=None, model=None): # decorator
        if has_decorator_parameters := model is not None:
            toolspec = _input = model
        else:
            toolspec = None
        def decorator(fn):
            nonlocal toolspec
            if fn.__doc__ is None:
                raise ValueError(f"Missing docstring: {fn.__name__}")
            fn._tool_name = toolname = fn.__name__
            def regen_toolspec(self, fn=fn, toolname=toolname, toolspec=toolspec):
                model_name = ''.join(word.title() for word in toolname.split('_'))
                if toolspec is None: # function signature based schema
                    def parameter_field(p):
                        _type = p.annotation
                        if type(p.default) is str:
                            if type(_type) is types.FunctionType:
                                _type = _type(self)
                            if type(_type) in (tuple, list):
                                if len(_type):
                                    _type = Literal[tuple(_type)]
                                else:
                                    _type = None
                            return _type, Field(..., description=p.default)
                        else:
                            return _type, p.default
                    fields = {}
                    for p in list(inspect.signature(fn).parameters.values())[1:]:
                        field_def = parameter_field(p)
                        if field_def is not None:
                            fields[p.name] = field_def
                else: # decorator argument schema or schema callback
                    if type(toolspec) is types.FunctionType:
                        toolspec = toolspec(self)
                    fields = {
                        name: (field.annotation, field)
                        for name, field in toolspec.model_fields.items()
                    }
                toolspec = create_model(model_name, **fields)
                toolspec.__doc__ = textwrap.dedent(fn.__doc__).strip()
                return toolspec
            fn._tool_spec = regen_toolspec
            return fn
        return decorator(_input) if not has_decorator_parameters else decorator

    @property
    def toolspecs(self):
        result = {}
        for k,v in self.__class__._toolspec.items():
            if type(v) is types.FunctionType:
                if spec := v(self):
                    result[k] = spec
            else:
                result[k] = v
        return result

    def toolcall(self, toolname, function_args):
        if func := self.__class__._toolimpl.get(toolname):
            return func(self, **function_args)
        raise KeyError(f"No tool '{toolname}' registered for class {self.__class__.__name__}")

    @property
    def llm_client(self):
        try:
            return self._llm_client
        except AttributeError:
            assert hasattr(self, 'model'), "model must be defined"
            native = self.native if hasattr(self, 'native') else None
            self._llm_client = LLMClient(self.model, native=native)
            return self._llm_client

    @property
    def conversation(self):
        try:
            return self._conversation
        except AttributeError:
            assert hasattr(self, 'system'), "system must be defined"
            self._conversation = self.llm_client.conversation(self.system)
            return self._conversation

    def llm(self):
        return self.conversation.llm(self.toolspecs)

    def text(self):
        return self.conversation.llm()['content']

    def usermsg(self, *args, **kwargs):
        return self.conversation.usermsg(*args, **kwargs)

    def chat(self, msg):
        self.usermsg(msg)
        return self.text()

    def toolmsg(self, *args, **kwargs):
        return self.conversation.toolmsg(*args, **kwargs)

    def run_loop(self, max_turns):
        self.complete = False
        for i in range(max_turns):
            resp_msg = self.llm()
            for tool_call in resp_msg["tool_calls"]:
                if tool_call['function']['name'] == "panic":
                    resp_msg["tool_calls"] = [tool_call]
                    break
            for tool_call in resp_msg["tool_calls"]:
                function_name = tool_call['function']['name']
                function_args = json.loads(tool_call['function']['arguments'])
                tool_response = self.toolcall(function_name, function_args)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"{function_name}: {tool_response}")
                if not tool_response and not self.complete:
                    raise ValueError(f"invalid return value from {function_name} and not complete")
                self.toolmsg(tool_response or "Success", name=function_name, tool_call_id=tool_call.get('id',''))
                if self.complete:
                    return tool_response
            if self.complete:
                break
        else:
            raise self.TurnLimitError(f"Turn limit of {max_turns} exceeded")

    def run(self, msg, max_turns=10):
        self.usermsg(msg)
        return self.run_loop(max_turns=max_turns)
