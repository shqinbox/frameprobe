"""
run_benchmark.py

The kbench execution script.
Generates the factorial conditions, joins them with the base dataset,
and runs the batch evaluation across selected models.
"""

import pandas as pd
import itertools
import json
from datasets import load_dataset
from pathlib import Path
import kaggle_benchmarks as kbench
from knowdobench_task import evaluate_clinical_case


def generate_conditions(factors_dict: dict) -> list[str]:
    """
    Generates all combination IDs for a given factor space.
    e.g., ['A0_P0', 'A0_P1', 'A1_P0', 'A1_P1']
    """
    clean_factors = {k: v for k, v in factors_dict.items() if k != "_meta"}

    level_lists = []
    for factor, levels in clean_factors.items():
        prefix = factor[0].upper()
        valid_levels = [f"{prefix}{level_key}" for level_key in levels.keys() if level_key != "_meta"]
        level_lists.append(valid_levels)

    combinations = list(itertools.product(*level_lists))
    return ["_".join(combo) for combo in combinations]


def main():
    # 1. Load the dataset directly from Hugging Face
    print("Loading KnowDoBench from Hugging Face Hub...")
    dataset = load_dataset("sammydman/KnowDoBench")
    df_base = dataset['train'].to_pandas()

    # 2. Generate the factorial context conditions
    with open("configs/components.json", "r") as f:
        components = json.load(f)
    conditions = generate_conditions(components)

    # 3. Create the cross-product DataFrame (Dataset x Conditions)
    print(f"Applying {len(conditions)} conditions to {len(df_base)} base cases...")
    df_conditions = pd.DataFrame({"condition_id": conditions})
    eval_df = df_base.merge(df_conditions, how="cross")

    print(f"Total evaluation rows: {len(eval_df)}")

    # 4. Define the models to evaluate
    models_to_test = [
        "google/gemini-1.5-pro",
        "anthropic/claude-3-sonnet"
    ]

    # 5. Execute via kbench SDK — raw per-row logs are written by kbench.log_raw_response
    print("Starting evaluation...")
    results_df = evaluate_clinical_case.evaluate(
        llm=models_to_test,
        evaluation_data=eval_df,
        max_workers=8
    )

    # 6. Save aggregate scores for downstream analysis
    output_path = Path("/kaggle/working/results_aggregate.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Aggregate results saved to {output_path}")


if __name__ == "__main__":
    main()
