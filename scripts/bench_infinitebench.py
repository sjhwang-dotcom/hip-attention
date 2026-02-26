"""InfiniteBench evaluation for HiP attention with CovarianceRouter.

Dataset: xinrongzhang2022/InfiniteBench (HuggingFace)
Tasks are stored as splits: passkey, kv_retrieval, number_string, etc.
Each sample has fields: id, context, input, answer (list[str]), options (list[str]).

Usage:
    # Baseline HiP attention (assumes SGLang server already running)
    python scripts/bench_infinitebench.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tasks passkey,kv_retrieval \
        --max-samples 20 \
        --server-url http://localhost:30000

    # With CovarianceRouter (experimental)
    python scripts/bench_infinitebench.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tasks passkey \
        --max-samples 20 \
        --use-covariance-router \
        --router-rank 4 \
        --router-budget 128

    # Using OpenAI-compatible chat endpoint
    python scripts/bench_infinitebench.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tasks passkey \
        --max-samples 10 \
        --api-style chat
"""

import argparse
import json
import re
import time
from pathlib import Path


def load_infinitebench(task: str, max_samples: int = 20) -> list:
    """Load InfiniteBench task from HuggingFace.

    Tasks are stored as splits in the dataset, not configs.
    The answer field is a Sequence[str] (list of acceptable answers).
    """
    from datasets import load_dataset

    ds = load_dataset("xinrongzhang2022/InfiniteBench", split=task)

    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        # answer is a list of strings; take the first as primary
        answer_list = row["answer"]
        primary_answer = answer_list[0] if answer_list else ""
        samples.append({
            "input": row["input"],
            "context": row["context"],
            "answer": primary_answer,
            "all_answers": answer_list,
            "task": task,
            "id": i,
        })

    return samples


def format_prompt(sample: dict) -> str:
    """Format InfiniteBench sample into a prompt.

    All three target tasks (passkey, kv_retrieval, number_string) use the
    same pattern: context followed by the input question/instruction.
    """
    context = sample["context"]
    input_text = sample["input"]
    return f"{context}\n\n{input_text}"


def evaluate_answer(task: str, prediction: str, answer: str, all_answers: list) -> bool:
    """Check if prediction matches any acceptable answer.

    For retrieval tasks, we check if the ground-truth answer appears anywhere
    in the model's prediction (case-insensitive). For passkey, we also extract
    numbers from the prediction and compare directly.
    """
    pred = prediction.strip().lower()
    ans = answer.strip().lower()

    if task == "passkey":
        # Passkey is a 5-digit number; extract all digit sequences from prediction
        numbers = re.findall(r"\d+", pred)
        return any(n == ans for n in numbers)
    elif task == "kv_retrieval":
        # Check if the target value appears in the prediction
        return any(a.strip().lower() in pred for a in all_answers)
    elif task == "number_string":
        return any(a.strip().lower() in pred for a in all_answers)
    else:
        # Generic substring match
        return any(a.strip().lower() in pred for a in all_answers)


def run_generate_api(
    samples: list,
    server_url: str,
    max_new_tokens: int,
) -> list:
    """Send prompts via SGLang /generate endpoint (non-OpenAI)."""
    import requests

    results = []
    for sample in samples:
        prompt = format_prompt(sample)
        est_tokens = int(len(prompt.split()) * 1.3)

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{server_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0,
                    },
                },
                timeout=600,
            )
            resp.raise_for_status()
            output = resp.json()["text"]
        except Exception as e:
            output = f"ERROR: {e}"

        latency = time.perf_counter() - t0
        correct = evaluate_answer(
            sample["task"], output, sample["answer"], sample["all_answers"]
        )

        results.append({
            "id": sample["id"],
            "task": sample["task"],
            "correct": correct,
            "latency_s": round(latency, 2),
            "est_input_tokens": est_tokens,
            "prediction": output[:200],
            "answer": sample["answer"],
        })

        status = "\033[92mPASS\033[0m" if correct else "\033[91mFAIL\033[0m"
        print(f"  [{status}] #{sample['id']} latency={latency:.1f}s tokens~{est_tokens}")

    return results


def run_chat_api(
    samples: list,
    server_url: str,
    model: str,
    max_new_tokens: int,
) -> list:
    """Send prompts via OpenAI-compatible /v1/chat/completions endpoint."""
    import requests

    results = []
    for sample in samples:
        prompt = format_prompt(sample)
        est_tokens = int(len(prompt.split()) * 1.3)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": 0,
                    "stream": False,
                },
                headers={"Authorization": "Bearer sk-placeholder"},
                timeout=600,
            )
            resp.raise_for_status()
            output = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            output = f"ERROR: {e}"

        latency = time.perf_counter() - t0
        correct = evaluate_answer(
            sample["task"], output, sample["answer"], sample["all_answers"]
        )

        results.append({
            "id": sample["id"],
            "task": sample["task"],
            "correct": correct,
            "latency_s": round(latency, 2),
            "est_input_tokens": est_tokens,
            "prediction": output[:200],
            "answer": sample["answer"],
        })

        status = "\033[92mPASS\033[0m" if correct else "\033[91mFAIL\033[0m"
        print(f"  [{status}] #{sample['id']} latency={latency:.1f}s tokens~{est_tokens}")

    return results


def print_summary(results: list, label: str = "") -> None:
    """Print accuracy and latency summary, grouped by task."""
    if not results:
        print("No results.")
        return

    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    print(f"\n{'=' * 64}")
    print(f"  {label or 'Results'}")
    print(f"{'=' * 64}")
    for task, task_results in by_task.items():
        n_correct = sum(1 for r in task_results if r["correct"])
        n_total = len(task_results)
        avg_latency = sum(r["latency_s"] for r in task_results) / n_total
        pct = 100 * n_correct / n_total
        print(f"  {task:20s}: {n_correct}/{n_total} ({pct:5.1f}%)  avg_latency={avg_latency:.1f}s")

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    avg_lat = sum(r["latency_s"] for r in results) / total
    pct = 100 * total_correct / total
    print(f"  {'TOTAL':20s}: {total_correct}/{total} ({pct:5.1f}%)  avg_latency={avg_lat:.1f}s")
    print(f"{'=' * 64}\n")


def main():
    parser = argparse.ArgumentParser(
        description="InfiniteBench evaluation for HiP attention"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--tasks", default="passkey",
        help="Comma-separated task names (passkey, kv_retrieval, number_string, ...)"
    )
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--api-style", choices=["generate", "chat"], default="generate",
        help="API endpoint style: 'generate' for /generate, 'chat' for /v1/chat/completions"
    )

    # CovarianceRouter flags (passed to server via env vars or config)
    parser.add_argument("--use-covariance-router", action="store_true",
                        help="Enable CovarianceRouter (experimental). "
                             "Requires server launched with appropriate config.")
    parser.add_argument("--router-rank", type=int, default=4,
                        help="CovarianceRouter rank (number of principal directions)")
    parser.add_argument("--router-budget", type=int, default=128,
                        help="CovarianceRouter block budget")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.use_covariance_router:
        print(f"CovarianceRouter enabled: rank={args.router_rank}, budget={args.router_budget}")
        print("  NOTE: The server must be launched with CovarianceRouter config.")
        print("  See hip_attn.v1_2.topk.k_importance.CovarianceRouter")

    all_results = []
    for task in tasks:
        print(f"\n--- Loading InfiniteBench task: {task}")
        samples = load_infinitebench(task, args.max_samples)
        print(f"    Loaded {len(samples)} samples")

        # Show context length stats
        lengths = [len(s["context"].split()) for s in samples]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        print(f"    Context length: avg={avg_len:.0f} words, max={max_len} words")

        print(f"    Running against {args.server_url} ({args.api_style} API)...")
        if args.api_style == "generate":
            results = run_generate_api(
                samples,
                server_url=args.server_url,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            results = run_chat_api(
                samples,
                server_url=args.server_url,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
            )
        all_results.extend(results)

    label = "CovarianceRouter" if args.use_covariance_router else "HiP Baseline"
    label += f" ({args.model.split('/')[-1]})"
    print_summary(all_results, label=label)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model,
                "tasks": tasks,
                "covariance_router": args.use_covariance_router,
                "router_rank": args.router_rank,
                "router_budget": args.router_budget,
                "results": all_results,
            }, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
