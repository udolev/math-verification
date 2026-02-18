# math-verification

Evaluate LLMs on math benchmarks (GSM8K, MATH-500, AIME 2024/2025/2026, DAPO) using vLLM inference and [`math-verify`](https://pypi.org/project/math-verify/) for answer extraction and verification.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Evaluate a model

```bash
uv run python evaluate.py --model <hf-model-path> --seed 42
```

**Options:**

| Flag | Description |
|------|-------------|
| `--greedy` | Greedy decoding instead of default sampling |
| `--no-thinking` | Disable thinking mode (for hybrid reasoning models like Qwen3) |
| `--tp N` | Tensor parallel size |
| `--gpus 0,1,...` | Comma-separated GPU IDs |
| `--output results.json` | Save full results to JSON |

### Use the verifier directly

```python
from verify import extract_gold_answer, verify_answer

gold = extract_gold_answer("#### 1,234", "gsm8k")
result = verify_answer(model_output, gold, "gsm8k")
# {"correct": bool, "has_boxed": bool, "extracted": str | None}
```
