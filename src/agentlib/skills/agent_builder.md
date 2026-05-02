# Build AgentLib agents from requirements and production examples

Use this skill when building or modifying agents with `agentlib`. Prefer code-first, production-shaped patterns over generic agent tutorials.

## Build workflow

1. Define the job boundary.
   - Is this a one-shot function, an interactive assistant, a scheduled automation, or a subsystem in a larger Python pipeline?
   - What data is authoritative? What should be deterministic Python instead of LLM judgment?
   - What result shape must callers receive?
2. Choose the agent paradigm.
3. Put orchestration in normal Python when possible.
4. Give the model narrow, validated tools.
5. Keep prompts concrete: role, data, workflow, required tool calls, output shape, examples.
6. Validate at tool boundaries and again after agent output.
7. Run a small real or fixture-backed test and inspect the transcript.

## Paradigm selection

| Need | Use | Pattern |
|---|---|---|
| Structured business decision | `BaseAgent` | Tool-calling loop, validated submit tool |
| Interactive terminal app | `CLIAgent` | `CLIMixin + BaseAgent` with final response tool |
| Data Q&A with computation | `PythonToolResponseMixin, CLIAgent` | Agent writes code in `python_execute_response` |
| Code-heavy autonomous workflow | `REPLAgent` | Model response is Python source |
| Large workflow with stages | Normal Python pipeline + specialist `BaseAgent`s | Python owns state, retries, validation |
| BaseAgent needs occasional deep research | BaseAgent tool that calls a `REPLAgent` | Hybrid pattern |
| Cron/scheduled automation | Plain script + `BaseAgent` | Deterministic loop around agent decisions |

Rule of thumb:
- Use `BaseAgent` when tool contracts and output validation matter.
- Use `REPLAgent` when computation/exploration dominates and Python state is useful.
- Use a hybrid when most work is structured but occasional research benefits from a REPL.
- Keep deterministic validation, persistence, retries, and external side effects outside the LLM when they can be normal Python.


## Model selection and configuration

Select the appropriate model automatically from this set when building agents:

| Model | Use |
|---|---|
| `xai/grok-4-1` | Default for simple tasks. Fast, cheap, good general-purpose model, especially data retrieval tasks. |
| `sonnet` | Everyday reasoning model. More intelligent than `xai/grok-4-1`, slower, and more opinionated. |
| `opus` | High-intelligence model for non-deterministic goal-oriented workflows, data analysis, and communication-heavy tasks. Prefers many turns. |
| `gpt-5.5-medium` | High-intelligence model, concise and direct. Best REPL performance and a good orchestrator. |

Default choices:
- Simple retrieval, extraction, formatting, routing: `xai/grok-4-1`.
- Normal business reasoning or mixed judgment: `sonnet`.
- Complex data analysis or open-ended goal pursuit: ask before using `opus`.
- REPL-heavy agents or orchestration-heavy workflows: ask before using `gpt-5.5-medium`.

Approval rule:
- For interactive, user-driven agents, choose the best fit from the table.
- For unattended automation, scheduled jobs, or side-effecting production workflows, use only `xai/grok-4-1` or `sonnet` unless the user explicitly approves `opus` or `gpt-5.5-medium`.
- If requirements do not clearly justify one of the four models, ask the user.

```python
class Agent(BaseAgent):
    model = "xai/grok-4-1"
```

For deployable projects:
- Put model selection in a config helper if several agents share tiers.
- Allow command-line or environment overrides for evaluation.
- Keep model choice at class level unless runtime switching is a requirement.
- Require explicit approval before configuring unattended automation to use `opus` or `gpt-5.5-medium`.

```python
class Agent(BaseAgent):
    model = get_model("listing")
```


## Minimal BaseAgent

```python
from agentlib import BaseAgent

class HashAgent(BaseAgent):
    model = "xai/grok-4-1"
    system = "You are a hashing assistant. Use sha256 to answer."

    @BaseAgent.tool
    def sha256(self, text: str = "Text to hash"):
        """Return the SHA-256 hex digest of the input text."""
        import hashlib
        self.respond(hashlib.sha256(text.encode()).hexdigest())

agent = HashAgent()
print(agent.run("hash hello world"))
```

`run()` requires tool use. A tool calls `self.respond(value)` to end the loop and return `value` to the Python caller. Use `chat()` only for direct text completion without tool enforcement.

## BaseAgent with state and a custom run method

```python
from agentlib import BaseAgent

class ReportAgent(BaseAgent):
    model = "sonnet"
    system = """You write concise reports from provided records.
Call submit_report when done."""

    def __init__(self, records: list[dict]):
        self.records = records

    @BaseAgent.tool
    def submit_report(self, summary: str = "Final report"):
        """Submit the final report."""
        self.respond(summary)

    def run(self, question: str, max_turns: int = 10):
        self.usermsg(f"Question: {question}\n\nRecords:\n{self.records}")
        return self.run_loop(max_turns=max_turns)
```

Use custom `run()` when you need to:
- load context before the first LLM call
- store input parameters on `self`
- attach files or structured data
- call `run_loop()` on an existing conversation for follow-up retry feedback

## Tool design

Tools should be small contracts, not general-purpose escape hatches.

```python
from typing import Literal
from agentlib import BaseAgent

class DecisionAgent(BaseAgent):
    model = "sonnet"
    system = "Choose an action, then call decide."

    @BaseAgent.tool
    def think(self, notes: str = "Working notes"):
        """Record reasoning and continue."""
        return "OK"

    @BaseAgent.tool
    def decide(
        self,
        action: Literal["APPROVE", "REJECT", "ESCALATE"] = "Decision",
        reason: str = "Reason for the decision",
    ):
        """Submit the final decision."""
        self.respond({"action": action, "reason": reason})
```

Patterns:
- Use string defaults as parameter descriptions.
- Use `Literal[...]` or list annotations for enums.
- Return actionable error strings for fixable problems; raise only for real failures.
- Make final tools call `self.respond(...)`.
- Use an explicit `think()` tool when the model benefits from a scratchpad turn.

## Pydantic submit schemas

Use Pydantic for structured final outputs and validation-heavy tools.

```python
from pydantic import BaseModel, Field
from agentlib import BaseAgent

class ReviewDecision(BaseModel):
    should_mark_read: bool = Field(..., description="Whether to mark the item read")
    reasoning: str = Field(..., description="Decision rationale")
    confidence: str = Field(..., description="high, medium, or low")

class Reviewer(BaseAgent):
    model = "sonnet"
    system = "Review the item, then call submit_decision."

    @BaseAgent.tool(model=ReviewDecision)
    def submit_decision(self, **decision):
        """Submit the structured review decision."""
        self.respond(decision)
```

Use this for cron/batch agents where the surrounding script needs reliable fields.

## Dynamic schemas

Use a dynamic schema when valid choices depend on instance state.

```python
from pydantic import Field, create_model
from agentlib import BaseAgent

class Picker(BaseAgent):
    model = "xai/grok-4-1"
    system = "Pick one option and call choose."

    def __init__(self, options):
        self.options = options

    def choose_model(self):
        return create_model(
            "ChooseModel",
            choice=(str, Field(..., description=f"One of: {self.options}")),
            reason=(str, Field(..., description="Why this choice")),
        )

    @BaseAgent.tool(model=choose_model)
    def choose(self, **payload):
        """Submit selected option."""
        if payload["choice"] not in self.options:
            return f"ERROR: choice must be one of {self.options}"
        self.respond(payload)
```

## Validation-by-tool feedback

Do not rely on prompts alone for hard constraints. Validate inside tools and return exact correction guidance.

```python
@BaseAgent.tool
def decide(self, items: list = "List of item decisions"):
    """Submit decisions for all items."""
    expected_ids = {x["id"] for x in self.items}
    got_ids = {x.get("id") for x in items}
    missing = expected_ids - got_ids
    if missing:
        return f"ERROR: missing decisions for IDs: {sorted(missing)}. Resubmit the full list."
    self.respond(items)
```

This pattern is common in production agents:
- prompt gives canonical examples
- tool rejects common wrong shapes
- agent gets another turn to repair
- deterministic Python validates again after completion

## REPLAgent basics

A `REPLAgent` makes the LLM write Python directly. The assistant response is executed as source code.

```python
from agentlib import REPLAgent

class AnalysisAgent(REPLAgent):
    model = "xai/grok-4-1"
    system = """You are a data analyst.
Write Python only. Use emit(answer, release=True) for the final answer."""

    @REPLAgent.tool
    def load_records(self):
        """Return records to analyze."""
        return [{"name": "A", "sales": 10}, {"name": "B", "sales": 20}]

agent = AnalysisAgent()
print(agent.run("Which record has the most sales?"))
```

The model should produce code like:

```python
records = load_records()
best = max(records, key=lambda r: r["sales"])
emit(f"{best['name']} has the most sales: {best['sales']}", release=True)
```

REPLAgent completion:
- `emit(value, release=True)` returns `value` from `agent.run()`.
- `emit(value)` is a progress update; the agent continues.
- Tools can still call `self.respond(value)`.

## REPLAgent tools: proxied vs injected

Default tools are proxied: the subprocess calls back to the host process. They can access `self`.

```python
@REPLAgent.tool
def query_db(self, sql: str = "SQL query"):
    """Query the host database."""
    return self.db.fetch_all(sql)
```

Injected tools run directly inside the subprocess. They are faster and visible as source, but cannot use `self`.

```python
@REPLAgent.tool(inject=True)
def read_text(self, path: str = "File path"):
    """Read a local text file."""
    from pathlib import Path
    return Path(path).read_text()
```

Use proxied tools for host state, credentials, live DB connections, and side effects. Use injected tools for pure local helpers.

## REPLAgent startup and grounding

Use `repl_startup` to preload imports, constants, clients, or data.

```python
class ResearchREPL(REPLAgent):
    model = "sonnet"
    system = """You answer questions about the preloaded `data` object.
Use emit(answer, release=True)."""

    def __init__(self, sku, data_snapshot=None):
        self.sku = sku
        self.data_snapshot = data_snapshot

    def repl_startup(self):
        return [
            "from myapp import sku_data",
            f"SKU = {self.sku!r}",
            "data = sku_data.fetch_all(SKU)",
            "assert data is not None",
        ]
```

For complex objects, add a synthetic initial exploration turn so the model sees the API and shape before real questions. Production pattern:
- preload `data`
- append assistant code that prints key attributes
- append user output showing those attributes
- optionally append generated API docs or summaries
- then answer focused research questions

Use this when a BaseAgent delegates to a REPL research subagent.

## Hybrid: BaseAgent with REPL research tool

This is often the best balance: structured outer agent, REPL only for hard analysis.

```python
from agentlib import BaseAgent, REPLAgent

class ResearchREPL(REPLAgent):
    model = "sonnet"
    system = """Answer specific data questions using Python.
If the answer is already in summaries, say so. Use emit(answer, release=True)."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def repl_startup(self):
        return [
            "from myapp.data import load_dataset",
            f"DATASET_ID = {self.dataset_id!r}",
            "data = load_dataset(DATASET_ID)",
        ]

class PricingAgent(BaseAgent):
    model = "sonnet"
    system = """Make the pricing decision from the summaries.
Use research only for specific questions not answered by the summaries."""

    def __init__(self, sku_data):
        self.sku_data = sku_data
        self._research_repl = None

    def _get_research_repl(self):
        if self._research_repl is None:
            self._research_repl = ResearchREPL(self.sku_data.id)
        return self._research_repl

    @BaseAgent.tool
    def research(self, question: str = "Specific question about raw data"):
        """Investigate raw data in a Python REPL."""
        repl = self._get_research_repl()
        repl.usermsg(question)
        return repl.run_loop(max_turns=25)

    @BaseAgent.tool
    def decide(self, price: float = "Target price", reason: str = "Reasoning"):
        """Submit final decision."""
        self.respond({"price": price, "reason": reason})

    def _cleanup(self):
        repl = getattr(self, "_research_repl", None)
        if repl is not None:
            repl.close()
        super()._cleanup()
```

Refinements from production:
- Make `research()` push back on vague questions.
- Tell the REPL to remind the caller when the answer was already in provided summaries.
- Include caller identity in the research prompt when several specialist agents share one REPL.
- Scope the REPL tightly so it does not answer workflow/mechanics questions outside its data domain.

## PythonToolResponseMixin for data Q&A

Use this when a tool-calling agent should execute Python as a tool and return the printed output directly.

```python
from agentlib import PythonToolResponseMixin
from agentlib.cli import CLIAgent

class FinanceAgent(PythonToolResponseMixin, CLIAgent):
    model = "xai/grok-4-1"
    system = """You answer questions about simplefin.sqlite3.

Call sql_query(query) in Python to query the database.
Use python_execute_response to format and return the final answer."""

    welcome_message = "Finance Assistant"
    cli_prompt = "finance> "
    max_turns = 30
    repl_startup = ["import sqlite3", "from datetime import datetime"]

    @PythonToolResponseMixin.repl
    def sql_query(self, query):
        """Execute SQL against the finance database and return list[dict]."""
        conn = sqlite3.connect("simplefin.sqlite3")
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

if __name__ == "__main__":
    FinanceAgent.main()
```

Use when:
- the model benefits from SQL/Python scratch work
- final answer is mostly computed output
- you want `BaseAgent`-style control rather than full REPLAgent autonomy

## CLI agents

`CLIAgent` is `CLIMixin + BaseAgent`.

```python
from agentlib.cli import CLIAgent

class Assistant(CLIAgent):
    model = "sonnet"
    system = "You are helpful. Call submit_response for final answers."
    welcome_message = "[bold]Assistant[/bold]"
    cli_prompt = "assistant> "
    history_db = "~/.assistant_history.db"
    max_turns = 20

    @CLIAgent.tool
    def submit_response(self, response: str = "Markdown response to user"):
        """Send final response."""
        self.respond(response)

if __name__ == "__main__":
    Assistant.main()
```

Mixin order: list mixins before `BaseAgent`, `CLIAgent`, or `REPLAgent`.

```python
class DataAssistant(PythonToolResponseMixin, CLIAgent):
    ...
```

## Attachments

Use attachments for large named context that may be updated or invalidated.

```python
from agentlib import AttachmentMixin, BaseAgent

class DocAgent(AttachmentMixin, BaseAgent):
    model = "sonnet"
    system = "Answer using attached documents."

    @BaseAgent.tool
    def done(self, answer: str = "Answer"):
        """Submit answer."""
        self.respond(answer)

agent = DocAgent()
agent.attach("schema.sql", open("schema.sql").read())
print(agent.run("Summarize the schema"))
```

For `REPLAgent`, use `REPLAttachmentMixin` so attachments render like line-numbered reads.

## Multi-agent pipeline pattern

For complex production workflows, make a normal Python state machine and call specialist agents at stages.

```python
from dataclasses import dataclass, field

@dataclass
class PipelineResult:
    item_id: str
    action: str
    decisions: list[dict] = field(default_factory=list)
    reasoning_log: list[dict] = field(default_factory=list)
    error: str | None = None

class Pipeline:
    def __init__(self, item_id: str, debug: bool = False):
        self.item_id = item_id
        self.debug = debug
        self.context = {}
        self.reasoning_log = []
        self.traces = {}

    def _log(self, stage, message, data=None):
        self.reasoning_log.append({"stage": stage, "message": message, "data": data})

    def _capture_trace(self, stage, agent):
        self.traces[stage] = list(agent.conversation.messages)

    def run(self):
        try:
            data = self.load_data()
            self.context["data"] = data

            gate = self.gate(data)
            if gate["action"] != "PROCEED":
                return PipelineResult(self.item_id, gate["action"], reasoning_log=self.reasoning_log)

            decision = self.decide(data)
            validated = self.validate(decision)
            return self.submit(validated)
        except Exception as e:
            self._log("error", str(e))
            return PipelineResult(self.item_id, "ERROR", reasoning_log=self.reasoning_log, error=str(e))
```

Production lessons:
- Fetch data once in Python, pass summaries to agents.
- Keep all durable state in the pipeline, not hidden in LLM text.
- Capture conversation traces for diagnosis.
- Perform deterministic validation after agent outputs.
- Retry by sending validation feedback to the same agent session when continuity matters.
- Use fallback behavior deliberately; do not silently auto-approve on agent failure.

## Shared base class for specialist agents

```python
from agentlib import BaseAgent

class WorkflowAgent(BaseAgent):
    model = "sonnet"

    @BaseAgent.tool
    def think(self, notes: str = "Working notes"):
        """Record reasoning and continue."""
        return "OK"

    def format_context(self, context: dict) -> str:
        import json
        return json.dumps(context, indent=2, default=str)
```

Then subclass per role:

```python
class BaselineAgent(WorkflowAgent):
    system = "You set baseline values. Call decide."

class ListingAgent(WorkflowAgent):
    system = "You make per-listing decisions. Call analyze before decide."
```

Shared base classes are good for:
- common model selection
- common tools (`think`, `research`, policy helpers)
- notebook/history formatting
- cleanup of shared subagents

## Cron/batch automation pattern

Use a script to select work, call an agent per item, then apply side effects.

```python
class ItemReviewer(BaseAgent):
    model = "sonnet"
    system = open("review.prompt").read()

    def __init__(self, manager):
        self.manager = manager

    @BaseAgent.tool
    def get_item_info(self, item_id: int = "Item ID"):
        """Retrieve full item details."""
        return self.manager.get_item_details(item_id)

    @BaseAgent.tool(model=ReviewDecision)
    def submit_decision(self, **decision):
        """Submit final decision."""
        self.respond(decision)

    def review_item(self, item_id: int):
        return self.run(
            f"Review item {item_id}. First call get_item_info, then submit_decision.",
            max_turns=10,
        )
```

Script responsibilities:
- parse flags (`--dry-run`, `--ticket`, `--interactive`)
- fetch candidate IDs deterministically
- limit scope if needed
- print summaries
- apply DB updates/notifications only after agent decision
- close DB connections in `close()` / context manager
- report success/failure through monitoring

Security notes:
- Do not put real API keys in committed code.
- Parameterize SQL.
- Keep destructive operations outside broad LLM tools when possible.

## Prompt design

Strong production prompts include:
- Role and exact scope.
- Data available.
- Required workflow steps.
- Tool call examples with valid shapes.
- Hard constraints.
- Common wrong shapes to avoid.
- Deadline/turn expectations when needed.
- Instruction to prefer provided summaries before expensive research.

Example fragment:

```text
## WORKFLOW

You must call analyze() before decide(). decide() rejects if analyze() has not been called.

### analyze()
Call analyze(listings=[...], macro_context="...")

Rules:
- `position` must be exactly one of: at_baseline, below_baseline, above_baseline
- Do not append explanations to enum values.
- If evidence is thin, say so explicitly.

### decide()
Call decide(decisions=[...], reasoning="...")
Use exactly these keys: listing_id, action, proposed_price, deviation_reason.
```

Do not ask the LLM to remember critical constraints only in prose; enforce them in tool code too.

## Context strategy

Prefer progressive disclosure:
1. Put high-signal summaries in the prompt.
2. Provide tools for drill-down.
3. Penalize vague drill-down calls.
4. Keep raw dumps out of initial context unless small.

Good tool guidance:

```text
All data needed for common decisions is in the summaries. Use research(question)
only for raw transaction details, edge cases, or novel cross-cutting analyses.
Ask a specific question; do not request open-ended analysis.
```

## Deterministic validation and retry

After an agent returns, validate in Python.

```python
result = agent.run()
errors = validate_result(result)

if errors:
    agent.usermsg(
        "Validation feedback on your prior output. "
        "Revise and resubmit the full payload.\n\n" + "\n".join(errors)
    )
    result = agent.run_loop(max_turns=10)
```

Use the same agent session for retries if prior reasoning and context matter. Use a new agent if you want a clean re-evaluation.

## Hooks and mixin extension

Agentlib mixins cooperate through these hooks:
- `_ensure_setup()` for lazy resources
- `_build_system_prompt()` for prompt additions
- `_get_dynamic_toolspecs()` for dynamic tools
- `_handle_toolcall(toolname, function_args)` for dispatch
- `_cleanup()` for resource cleanup

Always call `super()` when overriding.

```python
class MyMixin:
    def _ensure_setup(self):
        super()._ensure_setup()
        if getattr(self, "_mine_ready", False):
            return
        self._mine_ready = True

    def _build_system_prompt(self):
        return super()._build_system_prompt() + "\nExtra instructions."

    def _cleanup(self):
        try:
            resource = getattr(self, "_resource", None)
            if resource:
                resource.close()
        finally:
            super()._cleanup()
```


## File editing agents

For BaseAgent file editors, use `FilePatchMixin`.

```python
from agentlib import BaseAgent, FilePatchMixin

class Editor(FilePatchMixin, BaseAgent):
    model = "sonnet"
    system = "Edit files using apply_patch, then call done."
    patch_preview = None

    @BaseAgent.tool
    def done(self, summary: str = "Summary"):
        """Finish."""
        self.respond(summary)
```

## Testing and verification

Minimum checks before calling an agent done:
- Instantiate the agent.
- Inspect `agent.toolspecs` for expected tools and schemas.
- Run a smoke prompt with a cheap model or fixture data.
- Verify final Python return type.
- Check transcript when a tool schema is rejected.
- Test validation errors by intentionally sending bad tool payloads where feasible.
- Use context managers for agents with subprocesses, shell, REPL, or DB resources.

Example:

```python
with MyAgent(fake_data) as agent:
    print(agent.toolspecs.keys())
    result = agent.run("Process fixture item 123", max_turns=10)
    assert isinstance(result, dict)
    assert "action" in result
```

## Anti-patterns

Avoid:
- Putting the whole workflow inside one giant prompt.
- Letting the LLM own durable state that Python should own.
- Broad tools like `execute_sql(sql)` without read/write limits when only reads are needed.
- One tool that accepts arbitrary JSON when a Pydantic schema would fit.
- Prompts that say "must" without tool validation.
- Auto-approving on agent failure.
- Recreating expensive agents/subprocesses on every tool call when stateful reuse is intended.
- REPLAgent for simple structured decisions.
- BaseAgent for open-ended code/data exploration where Python state matters.
- Hidden side effects in tools named like read-only helpers.
- Missing `close()` / context manager around agents with resources.

## Agent type checklist

BaseAgent:
- [ ] `model` set
- [ ] `system` non-empty
- [ ] final tool calls `self.respond(...)`
- [ ] tool parameters have descriptions
- [ ] validation returns actionable errors
- [ ] custom `run()` calls `usermsg()` then `run_loop()`

REPLAgent:
- [ ] system says raw Python only and `emit(..., release=True)` for final
- [ ] `repl_startup` loads required state
- [ ] proxied vs injected tools chosen intentionally
- [ ] max turns high enough for exploration
- [ ] output is verified, not just printed

CLIAgent:
- [ ] `welcome_message`, `cli_prompt`, `history_db`
- [ ] final response tool exists or response mixin is used
- [ ] interrupts and turn limits are acceptable
- [ ] `main()` guarded by `if __name__ == "__main__"`

Production:
- [ ] deterministic data loading outside agent
- [ ] trace/log capture
- [ ] dry-run/debug mode
- [ ] deterministic validation
- [ ] safe side-effect boundaries
- [ ] resource cleanup
