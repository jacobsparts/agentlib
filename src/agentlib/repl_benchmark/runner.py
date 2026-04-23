
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable, Optional

from .core import (
    BenchmarkCategoryScore,
    BenchmarkRunResult,
    BenchmarkTask,
    BenchmarkTaskContext,
    BenchmarkTaskResult,
    InstrumentedREPLBenchmarkMixin,
    finalize_scores,
)
from .discovery import discover_tasks


class REPLBenchmarkRunner:
    def __init__(
        self,
        agent_cls,
        *,
        model: Optional[str] = None,
        tasks: Optional[Iterable[BenchmarkTask]] = None,
        task_modules: Optional[Iterable[str]] = None,
        task_paths: Optional[Iterable[str]] = None,
        include_builtin: bool = True,
        agent_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.agent_cls = agent_cls
        self.model = model
        self.agent_kwargs = agent_kwargs or {}
        self.tasks = list(tasks) if tasks is not None else discover_tasks(
            task_modules,
            task_paths,
            include_builtin=include_builtin,
        )

    def build_agent_class(self):
        return type(
            f"Benchmarked{self.agent_cls.__name__}",
            (InstrumentedREPLBenchmarkMixin, self.agent_cls),
            {},
        )

    def _usage_delta(self, before, after, model_name):
        history = after[len(before):]
        totals = {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "requests": len(history),
            "model": model_name,
        }
        tracker = getattr(self._agent.llm_client, "usage_tracker", None)
        for model_name_item, usage in history:
            if tracker is None:
                continue
            normalized = tracker._normalize(model_name_item, usage)
            for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens", "cost"):
                totals[key] += normalized.get(key, 0)
        return totals

    def run(self) -> BenchmarkRunResult:
        if not self.tasks:
            raise ValueError("No benchmark tasks found")

        AgentClass = self.build_agent_class()
        task_results: list[BenchmarkTaskResult] = []

        with AgentClass(**self.agent_kwargs) as agent:
            self._agent = agent
            if self.model is not None:
                agent.model = self.model
                if hasattr(agent, "_llm_client"):
                    delattr(agent, "_llm_client")
            model_name = agent.model
            tracker = agent.llm_client.usage_tracker
            usage_before_all = list(tracker.history)

            for task in self.tasks:
                if task.setup:
                    task.setup(agent)

                history_before = list(tracker.history)
                if hasattr(agent, "_benchmark_reset_metrics"):
                    agent._benchmark_reset_metrics()
                else:
                    agent._benchmark_metrics = {}
                agent.complete = False
                agent._final_result = None

                error = None
                result = None
                started_at = time.time()
                agent._benchmark_metrics["started_at"] = started_at
                try:
                    agent.usermsg(task.prompt)
                    result = agent.run_loop(max_turns=task.max_turns, max_syntax_retries=task.max_syntax_retries)
                except BaseException as exc:
                    error = exc
                finished_at = time.time()
                agent._benchmark_metrics["finished_at"] = finished_at

                ctx = BenchmarkTaskContext(
                    task=task,
                    agent=agent,
                    metrics=dict(agent._benchmark_metrics),
                    result=result,
                    error=error,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                passed, violations, scores = task.checker(ctx)
                total_score, total_possible = finalize_scores(scores)
                usage = self._usage_delta(history_before, tracker.history, model_name)
                metrics = dict(agent._benchmark_metrics)
                metrics["duration_seconds"] = ctx.duration_seconds
                metrics["usage"] = usage

                task_results.append(BenchmarkTaskResult(
                    task_id=task.id,
                    passed=bool(passed),
                    result=result,
                    error=None if error is None else f"{type(error).__name__}: {error}",
                    score_by_category=scores,
                    total_score=total_score,
                    total_possible=total_possible,
                    metrics=metrics,
                    violations=violations,
                ))

                if task.teardown:
                    task.teardown(agent)

            usage = self._usage_delta(usage_before_all, tracker.history, model_name)

        totals_by_category = defaultdict(lambda: BenchmarkCategoryScore(earned=0.0, possible=0.0, details=[]))
        for result in task_results:
            for name, score in result.score_by_category.items():
                totals = totals_by_category[name]
                totals.earned += score.earned
                totals.possible += score.possible
        total_score = sum(score.earned for score in totals_by_category.values())
        total_possible = sum(score.possible for score in totals_by_category.values())

        return BenchmarkRunResult(
            model=model_name,
            task_results=task_results,
            totals_by_category=dict(totals_by_category),
            total_score=total_score,
            total_possible=total_possible,
            usage=usage,
        )
