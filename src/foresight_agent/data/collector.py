"""Execute a curated local-tool catalog and export unlabeled trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from foresight_agent.agent.executor import AgentExecutor
from foresight_agent.agent.planner import Planner
from foresight_agent.agent.state import AgentState, TaskStep
from foresight_agent.evaluation.policies import NoRecoveryController
from foresight_agent.tools import CalculatorTool, LocalTextReadTool, LocalTextSearchTool, ToolRegistry

from .external import ExternalTrajectory, write_normalized_jsonl


@dataclass(frozen=True)
class CollectionTask:
    task_id: str
    category: str
    goal: str
    steps: list[dict[str, Any]]


def load_catalog(path: str | Path) -> tuple[dict[str, str], list[CollectionTask]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = {name: str(payload[name]) for name in ("dataset_id", "version", "source")}
    tasks = []
    seen = set()
    for index, raw in enumerate(payload.get("tasks", []), 1):
        task = CollectionTask(str(raw["task_id"]), str(raw["category"]), str(raw["goal"]), raw["steps"])
        if task.task_id in seen or not task.steps:
            raise ValueError(f"Catalog task {index} is duplicate or empty")
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError("Catalog contains no tasks")
    return metadata, tasks


def collect(catalog_path: str | Path, output_dir: str | Path,
            tool_root: str | Path = "data/collection") -> dict[str, Path]:
    metadata, tasks = load_catalog(catalog_path)
    registry = ToolRegistry([CalculatorTool(), LocalTextReadTool(tool_root), LocalTextSearchTool(tool_root)])
    trajectories: list[ExternalTrajectory] = []
    for task in tasks:
        plan_steps = []
        for index, specification in enumerate(task.steps, 1):
            tool = registry.get(str(specification["tool"]))
            arguments = dict(specification.get("arguments", {}))

            def execute(state: AgentState, selected=tool, supplied=arguments):
                result = selected.run(supplied)
                return {"progress_score": (state.current_step + 1) / len(state.plan), **result}

            plan_steps.append(TaskStep(f"{specification['tool']}_{index}", execute))
        result = AgentExecutor(Planner(plan_steps), controller=NoRecoveryController()).run(task.goal)
        for record in result.state.trajectory:
            trajectories.append(ExternalTrajectory(
                dataset_id=metadata["dataset_id"], task_id=task.task_id, run_id="run-01",
                step=record.step, goal=record.goal, action=record.action, observation=record.observation,
                success=record.success, confidence=record.confidence, progress_score=record.progress_score,
                repeated_action=record.repeated_action, contradiction=record.contradiction,
                retry_count=record.retry_count, step_position=record.step_position,
                progress_delta=record.progress_delta, previous_risk=record.previous_risk,
                actual_failure=None, failure_type=None, annotator_id=None,
                source=f"{metadata['source']}:{task.category}",
            ))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"trajectories": output / "raw_trajectories.jsonl", "manifest": output / "manifest.json"}
    write_normalized_jsonl(trajectories, paths["trajectories"])
    manifest = {**metadata, "collected_at": datetime.now(timezone.utc).isoformat(),
                "catalog": str(catalog_path), "tasks": len(tasks), "rows": len(trajectories),
                "categories": sorted({task.category for task in tasks}),
                "labels": "unlabeled", "network_used": False}
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Collected {len(trajectories)} steps from {len(tasks)} tasks: {paths['trajectories']}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("experiments/tasks/v08_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/collection/v08"))
    parser.add_argument("--tool-root", type=Path, default=Path("data/collection"))
    args = parser.parse_args()
    collect(args.catalog, args.output_dir, args.tool_root)


if __name__ == "__main__":
    main()
