"""
knowdobench_task.py

Defines the kbench SDK task for KnowDoBench.
Assembles the contextual prompt, queries the LLM, and evaluates the objective accuracy.
"""

import kaggle_benchmarks as kbench
from engine.assembler import PromptAssembler
from eval.accuracy import evaluate_accuracy

assembler = PromptAssembler.from_file("configs/components.json")

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
    **kwargs
) -> bool:
    """
    Evaluates a single clinical scenario under a specific framing condition.
    The arguments correspond directly to columns in the pandas DataFrame.
    """

    # 1. Assemble the prompt using the configured factor profile
    full_prompt = assembler.assemble(scenario, task, condition_id)

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

    # 4. Log structured metadata for the taxonomy classifier
    kbench.log_raw_response({
        "llm": str(llm),
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
