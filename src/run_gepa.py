"""
run_gepa.py — GEPA prompt optimization for the shopping concierge system prompt.

Optimizes prompts/baseline.md using a composite score across three evaluators:
  - task_completeness       (weight 1/3): did the answer actually complete the task?
  - critical_agents_called  (weight 1/3): were the necessary agents called?
  - sequence_respected      (weight 1/3): were prerequisite agents called before actions?
                                          (None for non-sequential tasks — excluded from mean)

Datasets: data/eval/train.jsonl (train), data/eval/val.jsonl (val)

Output:
    gepa/composite/prompt_{ts}.txt
    gepa/composite/stats_{ts}.json

Usage:
    uv run python src/run_gepa.py
"""

import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DEMO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Persistent background event loop.
# task_completeness calls asyncio.run(), which on Python 3.13 + anyio
# triggers "Event loop is closed" when TLS transport cleanup fires after
# the loop has already closed. Fix: patch asyncio.run to submit to a
# persistent loop that never closes. run_coroutine_threadsafe is
# thread-safe so all ThreadPoolExecutor workers can submit concurrently.
_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True).start()


def _async_run(coro, *, debug=False):
    return asyncio.run_coroutine_threadsafe(coro, _bg_loop).result()


asyncio.run = _async_run

import gepa.optimize_anything as oa
from gepa.optimize_anything import GEPAConfig, EngineConfig, optimize_anything

from agents.shopping_assistant import get_user_context, ALL_SUBAGENTS
from eval.evaluator import task_completeness, critical_agents_called, sequence_respected

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

import uuid

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROMPTS_DIR = DEMO_DIR / "prompts"
SEED_PROMPT = (PROMPTS_DIR / "baseline.md").read_text().strip()
TRAIN_JSONL = DEMO_DIR / "data" / "eval" / "train.jsonl"
VAL_JSONL = DEMO_DIR / "data" / "eval" / "val.jsonl"

MAX_WORKERS = 8  # parallel examples per iteration

# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    iteration: int
    composite_score: float
    tc_mean: float
    cac_mean: float
    sr_mean: float
    n_examples: int
    agent_calls: int
    duration_s: float


@dataclass
class RunStats:
    start_time: str = ""
    end_time: str = ""
    total_duration_s: float = 0.0
    total_iterations: int = 0
    total_agent_calls: int = 0
    baseline_train_score: float = 0.0
    baseline_val_score: float = 0.0
    optimized_val_score: float = 0.0
    val_delta: float = 0.0
    best_iteration: int = -1
    best_train_score: float = 0.0
    train_size: int = 0
    val_size: int = 0
    max_metric_calls: int = 0
    agent_model: str = ""
    judge_model: str = ""
    iterations: list = field(default_factory=list)


_stats = RunStats()
_agent_call_count = 0


def _count_agent_call():
    global _agent_call_count
    _agent_call_count += 1

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text().splitlines():
        if line.strip():
            examples.append(json.loads(line))
    return examples

# ---------------------------------------------------------------------------
# Agent factory — creates a fresh agent with a given system prompt
# ---------------------------------------------------------------------------

def _make_agent(system_prompt: str):
    return create_deep_agent(
        model="claude-haiku-4-5-20251001",
        system_prompt=system_prompt,
        subagents=ALL_SUBAGENTS,
        checkpointer=MemorySaver(),
    )

# ---------------------------------------------------------------------------
# Run one example through the agent and score it
# Called in a thread — asyncio.run() in task_completeness is safe here
# ---------------------------------------------------------------------------

def _run_and_score(agent, example: dict) -> dict:
    _count_agent_call()

    inputs = example["inputs"]
    query = inputs["query"]
    user_email = inputs["user_email"]
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    user_context = get_user_context(user_email)
    messages = [
        {"role": "system", "content": user_context},
        {"role": "user", "content": query},
    ]

    agents_called: list[str] = []
    final_output: str = ""

    for chunk in agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="debug",
        subgraphs=True,
    ):
        if isinstance(chunk, tuple):
            if chunk[0] != ():
                continue
            _, event = chunk
        else:
            event = chunk

        if isinstance(event, dict):
            event_type = event.get("type", "")
            payload = event.get("payload", {})

            if event_type == "task" and payload.get("name") == "tools":
                input_data = payload.get("input", {})
                if input_data.get("__type") == "tool_call_with_context":
                    tc = input_data.get("tool_call", {})
                    if tc.get("name") == "task":
                        agent_name = tc.get("args", {}).get("subagent_type")
                        if agent_name and agent_name not in agents_called:
                            agents_called.append(agent_name)

            if event_type == "task_result" and payload.get("name") == "model":
                result_data = payload.get("result", {})
                for msg in result_data.get("messages", []):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        final_output = msg.content

    run_out = {
        "agents_called": agents_called,
        "final_output": final_output,
        "query": query,
        "user_email": user_email,
    }

    # Wrap for evaluator interface
    run_obj = type("Run", (), {"outputs": run_out})()
    ex_obj = type("Example", (), {
        "inputs": example["inputs"],
        "outputs": example["outputs"],
    })()

    tc_result = task_completeness(run_obj, ex_obj)
    cac_result = critical_agents_called(run_obj, ex_obj)
    sr_result = sequence_respected(run_obj, ex_obj)

    # Composite per example: mean of non-None scores
    metric_scores = [
        s for s in [tc_result["score"], cac_result["score"], sr_result["score"]]
        if s is not None
    ]
    example_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0

    return {
        "example_score": example_score,
        "tc": tc_result["score"],
        "cac": cac_result["score"],
        "sr": sr_result["score"],
        "tc_comment": tc_result["comment"],
        "cac_comment": cac_result["comment"],
        "sr_comment": sr_result["comment"],
        "agents_called": agents_called,
        "query": query,
        "task_type": example["outputs"].get("task_type", "?"),
    }

# ---------------------------------------------------------------------------
# Score a full dataset with a candidate prompt
# ---------------------------------------------------------------------------

def score_dataset(system_prompt: str, examples: list[dict]) -> tuple[float, list[dict]]:
    agent = _make_agent(system_prompt)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_and_score, agent, ex): i for i, ex in enumerate(examples)}
        results = [None] * len(examples)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    composite = sum(r["example_score"] for r in results) / len(results)
    return composite, results

# ---------------------------------------------------------------------------
# GEPA evaluator
# ---------------------------------------------------------------------------

def make_evaluator(trainset: list[dict], out_dir: Path, ts: str):
    iteration_counter = [0]
    best_score = [0.0]
    best_candidate = [None]

    def evaluate(candidate: str) -> float:
        global _agent_call_count

        iteration_counter[0] += 1
        iteration = iteration_counter[0]
        t0 = time.perf_counter()
        calls_before = _agent_call_count

        score, results = score_dataset(candidate, trainset)

        duration = time.perf_counter() - t0
        agent_calls = _agent_call_count - calls_before

        # Per-metric means (excluding None)
        tc_scores = [r["tc"] for r in results if r["tc"] is not None]
        cac_scores = [r["cac"] for r in results if r["cac"] is not None]
        sr_scores = [r["sr"] for r in results if r["sr"] is not None]
        tc_mean = sum(tc_scores) / len(tc_scores) if tc_scores else 0.0
        cac_mean = sum(cac_scores) / len(cac_scores) if cac_scores else 0.0
        sr_mean = sum(sr_scores) / len(sr_scores) if sr_scores else 0.0

        _stats.iterations.append(asdict(IterationStats(
            iteration=iteration,
            composite_score=score,
            tc_mean=round(tc_mean, 3),
            cac_mean=round(cac_mean, 3),
            sr_mean=round(sr_mean, 3),
            n_examples=len(results),
            agent_calls=agent_calls,
            duration_s=round(duration, 2),
        )))

        if score > best_score[0]:
            best_score[0] = score
            best_candidate[0] = candidate
            _stats.best_iteration = iteration
            _stats.best_train_score = score
            # Save checkpoint immediately so results survive an early kill
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"prompt_{ts}.txt").write_text(candidate)
            (out_dir / f"stats_{ts}.json").write_text(
                json.dumps(asdict(_stats), indent=2)
            )
            print(f"  *** new best {score:.3f} — checkpoint saved ***")

        # Log failures as ASI for GEPA's reflection LLM
        for r in results:
            if r["cac"] == 0.0:
                oa.log(
                    f"MISSING AGENTS — query: \"{r['query'][:100]}\" | "
                    f"task_type: {r['task_type']} | "
                    f"agents_called: {r['agents_called']} | "
                    f"{r['cac_comment'][:200]}"
                )
            elif r["tc"] is not None and r["tc"] < 0.5:
                oa.log(
                    f"INCOMPLETE TASK — query: \"{r['query'][:100]}\" | "
                    f"task_type: {r['task_type']} | "
                    f"agents_called: {r['agents_called']} | "
                    f"{r['tc_comment'][:200]}"
                )
            elif r["sr"] is not None and r["sr"] < 0.5:
                oa.log(
                    f"WRONG SEQUENCE — query: \"{r['query'][:100]}\" | "
                    f"agents_called: {r['agents_called']} | "
                    f"{r['sr_comment'][:200]}"
                )

        print(
            f"  iter {iteration:>3} | score {score:.3f} | "
            f"tc {tc_mean:.3f} | cac {cac_mean:.3f} | sr {sr_mean:.3f} | "
            f"agent calls {agent_calls} | {duration:.1f}s"
        )

        return score

    return evaluate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime

    MAX_METRIC_CALLS = 30

    trainset = load_jsonl(TRAIN_JSONL)
    valset = load_jsonl(VAL_JSONL)

    _stats.train_size = len(trainset)
    _stats.val_size = len(valset)
    _stats.max_metric_calls = MAX_METRIC_CALLS
    _stats.agent_model = "claude-haiku-4-5-20251001"
    _stats.judge_model = "claude-haiku-4-5-20251001"
    _stats.start_time = datetime.datetime.now().isoformat()
    ts = _stats.start_time[:19].replace(":", "-")
    out_dir = DEMO_DIR / "gepa" / "composite"

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    print("\nScoring baseline on trainset...")
    baseline_train, _ = score_dataset(SEED_PROMPT, trainset)
    _stats.baseline_train_score = baseline_train
    print(f"  Baseline train composite: {baseline_train:.3f}")

    print("\nScoring baseline on valset...")
    baseline_val, _ = score_dataset(SEED_PROMPT, valset)
    _stats.baseline_val_score = baseline_val
    print(f"  Baseline val composite:   {baseline_val:.3f}")

    print(f"\nRunning GEPA ({MAX_METRIC_CALLS} max evaluator calls)...")
    t_start = time.perf_counter()

    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=make_evaluator(trainset, out_dir, ts),
        objective=(
            "Optimize a system prompt for a shopping concierge Deep Agent that routes "
            "customer queries to specialized subagents. The agent must call the right "
            "subagents (product catalog, web research, return policy, etc.) to gather "
            "all information needed to fully answer the customer. For sequential tasks "
            "(e.g. check eligibility before taking action), the prerequisite agent must "
            "be called first. Improve task completeness, critical agent coverage, and "
            "correct sequencing."
        ),
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=MAX_METRIC_CALLS)),
    )

    total_duration = time.perf_counter() - t_start
    best_prompt = result.best_candidate

    print("\nValidating on held-out valset...")
    optimized_val, _ = score_dataset(best_prompt, valset)

    _stats.end_time = datetime.datetime.now().isoformat()
    _stats.total_duration_s = round(total_duration, 2)
    _stats.total_iterations = len(_stats.iterations)
    _stats.total_agent_calls = _agent_call_count
    _stats.optimized_val_score = optimized_val
    _stats.val_delta = round(optimized_val - baseline_val, 4)

    print("\n" + "="*60)
    print("GEPA RUN SUMMARY")
    print("="*60)
    print(f"  Total wall time:          {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  GEPA iterations:          {_stats.total_iterations}")
    print(f"  Total agent calls:        {_stats.total_agent_calls}")
    print(f"  Avg agent calls/iter:     {_stats.total_agent_calls / max(_stats.total_iterations, 1):.1f}")
    print(f"  Avg time/iteration:       {total_duration / max(_stats.total_iterations, 1):.1f}s")
    print(f"  Best iteration:           #{_stats.best_iteration}")
    print(f"  Baseline val composite:   {baseline_val:.3f}")
    print(f"  Optimized val composite:  {optimized_val:.3f}")
    print(f"  Val delta:                {_stats.val_delta:+.4f}")
    print("="*60)

    prompt_path = out_dir / f"prompt_{ts}.txt"
    stats_path = out_dir / f"stats_{ts}.json"
    prompt_path.write_text(best_prompt)
    stats_path.write_text(json.dumps(asdict(_stats), indent=2))
    print(f"\nSaved → {prompt_path}")
    print(f"Saved → {stats_path}")
