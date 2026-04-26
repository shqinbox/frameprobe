"""
knowdobench_task.py

Defines the kbench SDK task for KnowDoBench.
Assembles the contextual prompt, queries the LLM, and evaluates the objective accuracy.
"""

import json
import threading
from pathlib import Path
import kaggle_benchmarks as kbench
from engine.assembler import PromptAssembler
from eval.accuracy import evaluate_accuracy

# Module-level assembler: set via configure() or lazy-loaded from default JSON
_assembler = None
_components_dict = None

# Thread-safe lock for parallel raw response logging
_write_lock = threading.Lock()

def log_raw_response(record: dict):
    """Thread-safe append to JSONL sidecar for taxonomy classification."""
    out_path = Path("results/raw_responses.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def configure(components_dict: dict):
    """Configure the task module with a components dictionary before evaluation."""
    global _assembler, _components_dict
    _components_dict = components_dict
    _assembler = PromptAssembler(components_dict)


def get_assembler() -> PromptAssembler:
    """Return the configured assembler, falling back to configs/components.json."""
    global _assembler
    if _assembler is None:
        _assembler = PromptAssembler.from_file("configs/components.json")
    return _assembler

@kbench.task(name="knowdobench_clinical_eval")
def evaluate_clinical_case(
    llm,
    scenario: str,
    task: str,
    expected_answerable: bool,
    expected_answer: str,
    evaluator: str,
    tolerance: float,
    track: str = "solvable",
    condition_id: str = "A0_P0",
    id: str = None,
    **kwargs
) -> bool:
    """
    Evaluates a single clinical scenario under a specific framing condition.
    The arguments correspond directly to columns in the pandas DataFrame.
    """

    # 1. Assemble the prompt using the configured factor profile
    full_prompt = get_assembler().assemble(scenario, task, condition_id)

    # 2. Query the model (temperature=0 is standard for strict benchmark evals)
    response_text = str(llm.prompt(full_prompt, temperature=0.0))

    # 3. Score the response deterministically
    eval_result = evaluate_accuracy(
        response_text=response_text,
        expected_answerable=expected_answerable,
        expected_answer=expected_answer,
        evaluator_type=evaluator,
        tolerance=tolerance
    )

    # 4. Log structured metadata for the taxonomy classifier (Removed 'kbench.')
    log_raw_response({
        "llm": str(llm),
        "id": id,
        "condition_id": condition_id,
        "track": track,
        "is_solvable": expected_answerable,
        "is_correct": eval_result["is_correct"],
        "error_type": eval_result["error_type"],
        "parsed_answerable": eval_result["model_answerable"],
        "parsed_answer": eval_result["model_answer"],
        "raw_text": response_text
    })

    # 5. Register assertion for the leaderboard scorer
    kbench.assertions.assert_true(
        eval_result["is_correct"],
        expectation=f"[{condition_id}] {eval_result['error_type']}"
    )
    return eval_result["is_correct"]
