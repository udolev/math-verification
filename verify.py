"""Verification module for LLM math evaluation outputs."""

from __future__ import annotations

from typing import Any

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse,
    verify,
)

PRED_STRICT_CONFIGS = [LatexExtractionConfig(boxed_match_priority=0)]
PRED_FLEX_CONFIGS = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
GOLD_LATEX_CONFIGS = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
GOLD_EXPR_CONFIGS = [ExprExtractionConfig()]

INTEGER_DATASETS = {"gsm8k", "aime_2024", "aime_2025", "aime_2026", "dapo"}


def extract_gold_answer(gold_raw: str, dataset: str) -> str:
    dataset = dataset.lower()

    if dataset == "gsm8k":
        return gold_raw.split("####")[-1].strip().replace(",", "")

    if dataset in INTEGER_DATASETS:
        return str(int(gold_raw))

    return gold_raw


def verify_answer(
    model_output: str, gold_answer: str, dataset: str, *, strict: bool = False
) -> dict[str, Any]:
    dataset = dataset.lower()
    response_text = model_output.split("</think>")[-1]
    has_boxed = r"\boxed{" in response_text

    pred_configs = PRED_STRICT_CONFIGS if strict else PRED_FLEX_CONFIGS
    try:
        parsed_pred = parse(response_text, extraction_config=pred_configs)
    except Exception:
        return {"correct": False, "has_boxed": has_boxed, "extracted": None}

    if not parsed_pred:
        return {"correct": False, "has_boxed": has_boxed, "extracted": None}

    if dataset in INTEGER_DATASETS:
        gold_configs = GOLD_EXPR_CONFIGS
        gold_text = gold_answer
    else:
        gold_configs = GOLD_LATEX_CONFIGS
        gold_text = rf"\boxed{{{gold_answer}}}"

    try:
        parsed_gold = parse(gold_text, extraction_config=gold_configs)
    except Exception:
        return {"correct": False, "has_boxed": has_boxed, "extracted": str(parsed_pred[0])}

    if not parsed_gold:
        return {"correct": False, "has_boxed": has_boxed, "extracted": str(parsed_pred[0])}

    try:
        correct = verify(parsed_gold, parsed_pred)
    except Exception:
        correct = False

    return {
        "correct": correct,
        "has_boxed": has_boxed,
        "extracted": str(parsed_pred[0]),
    }
