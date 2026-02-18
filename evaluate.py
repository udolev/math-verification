"""Evaluate a model on all math datasets using vLLM.

Usage:
    uv run python evaluate.py --model Qwen/Qwen3-1.7B --seed 42 --output results.json
    uv run python evaluate.py --model Qwen/Qwen3-1.7B --seed 42 --greedy --output results.json
    uv run python evaluate.py --model Qwen/Qwen3-1.7B --seed 42 --no-thinking --output results.json
    uv run python evaluate.py --model Qwen/Qwen3-1.7B --seed 42 --tp 4 --gpus 0,1,2,3 --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

from datasets import load_dataset
from vllm import LLM, SamplingParams

from verify import extract_gold_answer, verify_answer

# as recommended by qwen in the model card: https://huggingface.co/Qwen/Qwen3-1.7B
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

# to reproduce: https://www.arxiv.org/abs/2601.19847
REPRODUCTION_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Thinking process:\nPlease provide a step-by-step thinking process and put your final answer in \\boxed{{}}."
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    question_col: str
    answer_col: str
    subsample: int | None
    max_tokens: int


DATASETS = [
    DatasetConfig("gsm8k", "openai/gsm8k", "main", "test", "question", "answer", None, 32768),
    DatasetConfig("math-500", "HuggingFaceH4/MATH-500", None, "test", "problem", "answer", None, 32768),
    DatasetConfig("aime_2024", "HuggingFaceH4/aime_2024", None, "train", "problem", "answer", None, 38912),
    DatasetConfig("aime_2025", "MathArena/aime_2025", None, "train", "problem", "answer", None, 38912),
    DatasetConfig("aime_2026", "MathArena/aime_2026", None, "train", "problem", "answer", None, 38912),
    DatasetConfig("dapo", "ftajwar/deduplicated_dapo_dataset", None, "train", "prompt", "answer", 100, 32768),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on math datasets")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results file")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of recommended sampling params")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    return parser.parse_args()


def load_dataset_examples(config: DatasetConfig, seed: int) -> tuple[list[str], list[str]]:
    ds = load_dataset(config.hf_path, name=config.hf_config, split=config.split)
    if config.subsample is not None:
        ds = ds.shuffle(seed=seed).select(range(min(config.subsample, len(ds))))
    questions = [row[config.question_col] for row in ds]
    gold_raws = [str(row[config.answer_col]) for row in ds]
    return questions, gold_raws


def build_messages(questions: list[str]) -> list[list[dict]]:
    return [
        [{"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}]
        for q in questions
    ]


def run_inference(
    llm: LLM,
    messages: list[list[dict]],
    sampling_params: SamplingParams,
    chat_template_kwargs: dict | None = None,
) -> list[str]:
    kwargs = {}
    if chat_template_kwargs:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    outputs = llm.chat(messages, sampling_params, **kwargs)
    return [o.outputs[0].text for o in outputs]


def evaluate_dataset(
    llm: LLM,
    base_params: dict,
    config: DatasetConfig,
    seed: int,
    chat_template_kwargs: dict | None = None,
) -> dict:
    questions, gold_raws = load_dataset_examples(config, seed)
    n = len(questions)
    if n == 0:
        return {"name": config.name, "n": 0, "has_boxed": 0, "strict": 0, "flex": 0,
                "has_boxed_pct": 0.0, "strict_at_1": 0.0, "flex_at_1": 0.0, "examples": []}

    sampling_params = SamplingParams(**base_params, max_tokens=config.max_tokens)
    messages = build_messages(questions)
    model_outputs = run_inference(llm, messages, sampling_params, chat_template_kwargs)

    has_boxed = 0
    strict_correct = 0
    flex_correct = 0
    examples = []
    for question, model_output, gold_raw in zip(questions, model_outputs, gold_raws):
        gold_answer = extract_gold_answer(gold_raw, config.name)
        flex_result = verify_answer(model_output, gold_answer, config.name)
        strict_result = verify_answer(model_output, gold_answer, config.name, strict=True)
        if flex_result["has_boxed"]:
            has_boxed += 1
        if strict_result["correct"]:
            strict_correct += 1
        if flex_result["correct"]:
            flex_correct += 1
        examples.append({
            "question": question,
            "gold_answer": gold_answer,
            "model_output": model_output,
            "extracted": flex_result["extracted"],
            "has_boxed": flex_result["has_boxed"],
            "strict_correct": strict_result["correct"],
            "flex_correct": flex_result["correct"],
        })

    return {
        "name": config.name,
        "n": n,
        "has_boxed": has_boxed,
        "strict": strict_correct,
        "flex": flex_correct,
        "has_boxed_pct": has_boxed / n * 100,
        "strict_at_1": strict_correct / n * 100,
        "flex_at_1": flex_correct / n * 100,
        "examples": examples,
    }


def format_table(model: str, seed: int, results: list[dict], pending: list[str] | None = None) -> str:
    lines = [
        f"Model: {model}",
        f"Seed: {seed}",
        "",
        f"{'Dataset':<14}| {'N':>5} | {'Boxed%':>6} | {'Strict':>6} | {'Flex':>6} | {'Strict@1':>8} | {'Flex@1':>6}",
        f"{'-' * 14}|{'-' * 7}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 8}",
    ]
    for r in results:
        lines.append(
            f"{r['name']:<14}"
            f"| {r['n']:>5} "
            f"| {r['has_boxed_pct']:>5.1f}% "
            f"| {r['strict']:>6} "
            f"| {r['flex']:>6} "
            f"| {r['strict_at_1']:>7.1f}% "
            f"| {r['flex_at_1']:>5.1f}%"
        )
    if pending:
        for name in pending:
            lines.append(f"{name:<14}|       ...pending")
    return "\n".join(lines)


def print_table(model: str, seed: int, results: list[dict], pending: list[str] | None = None) -> None:
    table = format_table(model, seed, results, pending)
    # Clear screen and print updated table
    print("\033[2J\033[H" + table, flush=True)


def save_results(model: str, seed: int, results: list[dict], output_path: str) -> None:
    output = {
        "model": model,
        "seed": seed,
        "datasets": {},
    }
    for r in results:
        output["datasets"][r["name"]] = {
            "n": r["n"],
            "has_boxed": r["has_boxed"],
            "has_boxed_pct": r["has_boxed_pct"],
            "strict": r["strict"],
            "flex": r["flex"],
            "strict_at_1": r["strict_at_1"],
            "flex_at_1": r["flex_at_1"],
            "examples": r["examples"],
        }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    llm = LLM(model=args.model, seed=args.seed, tensor_parallel_size=args.tp)

    if args.greedy:
        base_params = dict(temperature=0, top_p=1.0, seed=args.seed)
    elif args.no_thinking:
        base_params = dict(temperature=0.7, top_p=0.8, top_k=20, seed=args.seed)
    else:
        base_params = dict(temperature=0.6, top_p=0.95, top_k=20, seed=args.seed)

    chat_template_kwargs = {"enable_thinking": False} if args.no_thinking else None

    results = []
    pending = [c.name for c in DATASETS]
    print_table(args.model, args.seed, results, pending)

    for config in DATASETS:
        pending.remove(config.name)
        try:
            result = evaluate_dataset(llm, base_params, config, args.seed, chat_template_kwargs)
            results.append(result)
        except Exception as e:
            print(f"WARNING: Skipping {config.name}: {e}", file=sys.stderr)
        print_table(args.model, args.seed, results, pending)
        if args.output:
            save_results(args.model, args.seed, results, args.output)

    # Print final table without ANSI codes for log file readability
    print("\n\n=== Final Results ===")
    print(format_table(args.model, args.seed, results), flush=True)


if __name__ == "__main__":
    main()
