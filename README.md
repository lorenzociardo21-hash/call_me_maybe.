*This project has been created as part of the 42 curriculum by lciardo.*

# call-me-maybe

## Description

**call-me-maybe** is a function calling tool that bridges the gap between natural language and structured, machine-executable output. Given a prompt like *"What is the sum of 40 and 2?"*, the system does not answer the question — instead it identifies the right function to call and extracts its arguments with the correct types:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The core challenge is reliability: small language models (0.6B parameters) succeed at producing valid JSON only ~30% of the time when prompted naively. This project solves that with **constrained decoding** — a technique that guides token generation step-by-step to guarantee 100% structurally valid, schema-compliant output.

The model used is **Qwen/Qwen3-0.6B**, accessed through the provided `llm_sdk` package.

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository, then:
uv sync
```

> Make sure the `llm_sdk/` directory is placed at the same level as `src/`.

### Running the program

```bash
# Default paths (data/input/ → data/output/)
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target | Command | Description |
|---|---|---|
| `install` | `make install` | Install dependencies via `uv sync` |
| `run` | `make run` | Run the program with default paths |
| `debug` | `make debug` | Run with Python's `pdb` debugger |
| `clean` | `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `lint` | `make lint` | Run `flake8` + `mypy` |
| `lint-strict` | `make lint-strict` | Run `flake8` + `mypy --strict` |

### Input files (place in `data/input/`)

**`functions_definition.json`** — defines the available functions:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

**`function_calling_tests.json`** — the natural language prompts to process:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet john"}
]
```

### Output

The program writes `data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

---

## Algorithm Explanation

The pipeline has two stages, both using constrained decoding:

### Stage 1 — Function selection (`ai_name`)

A multiple-choice prompt is built listing all available functions with their descriptions. The model is then forced to output **only digit tokens** (plus space/newline as stop signals) by setting all other logits to `-inf`. This guarantees the output is always a valid index, which maps deterministically to a function name.

### Stage 2 — Parameter extraction (`ai_parameters`)

For each parameter in the selected function's schema:

- **`number` / `integer`** — `extrat_num`: only digit tokens, `-`, `.` (for floats), space and newline are permitted. The result is cast to `float` or `int`.
- **`string`** — `extrat_string`: no token filtering; the model generates freely until it produces a newline. The output is stripped and unquoted.

In both cases a few-shot prompt is prepended to steer the model toward the correct format. Previously extracted parameters are also injected into the prompt context so multi-parameter functions stay consistent.

---

## Design Decisions

- **Pydantic models** (`SchemaTypeInfo`, `FunctionsDefinition`, `FunctionCallingTests`) validate all input data at load time, catching malformed JSON early.
- **Vocabulary pre-scan**: the vocab JSON is read once per function call and used to build the set of permitted token IDs, rather than checking tokens on the fly.
- **Greedy decoding** (`argmax`) is used instead of sampling for determinism and speed.
- **Separation of concerns**: `ai_core.py` orchestrates selection and dispatch; `ai_extractors.py` handles type-specific generation; `__main__.py` owns I/O and CLI.
- **No forbidden libraries**: only `numpy`, `json`, `pydantic`, and `llm_sdk` are used. PyTorch, HuggingFace, `dspy`, `outlines`, etc. are absent.

---

## Performance Analysis

| Metric | Target | Notes |
|---|---|---|
| JSON validity | 100% | Guaranteed by constrained decoding — invalid tokens are masked to `-inf` |
| Function selection accuracy | ≥ 90% | Constrained digit output ensures a valid index is always returned |
| Argument extraction accuracy | ≥ 90% | Few-shot prompts + type-specific token filtering |
| Speed | < 5 min | Greedy decoding on Qwen3-0.6B; no batching overhead |

The constrained approach means even a 0.6B model achieves reliability comparable to much larger models on this structured task.

---

## Challenges Faced

- **Tokenizer quirks**: tokens in the vocabulary include prefix characters like `Ġ` (space marker) and `▂▃▄▅▆▇█`. These must be stripped before checking if a token represents a digit or punctuation.
- **Stop condition for strings**: since string values can contain any character, the generation loop runs freely and stops only on a newline token, then strips and unquotes the result.
- **Negative numbers**: `-` must be included in the allowed token set for numeric extraction, but only before digits have been produced — otherwise the model could generate `--3`.
- **Multi-parameter consistency**: already-extracted parameters are injected back into the prompt for subsequent parameters, preventing the model from repeating the same number for different arguments.

---

## Testing Strategy

- Ran the program against the provided example inputs (add, greet, reverse) and verified the output JSON structure and types manually.
- Tested edge cases: very large numbers, empty string arguments, prompts with no obvious match.
- Used `make lint` and `make lint-strict` to verify type correctness and flake8 compliance.
- Verified that all JSON error paths (missing file, invalid JSON) print a clear message and exit gracefully without exceptions propagating.

---

## Resources

- [Qwen3 model card — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained decoding overview — Outlines docs](https://outlines-dev.github.io/outlines/)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [flake8 documentation](https://flake8.pycqa.org/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [uv documentation](https://docs.astral.sh/uv/)

### AI usage

Claude (claude.ai) was used to:
- Generate the initial structure of this README from the project subject and source files.
- Suggest docstring wording for functions in `ai_core.py` and `ai_extractors.py`.

All generated content was reviewed, understood, and validated before inclusion. No AI tool was used to write or generate any logic in the source files.