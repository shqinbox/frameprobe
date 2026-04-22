"""
run_benchmark.py

The Kaggle execution script.
Generates the factorial conditions, joins them with the base dataset, 
and runs the batch evaluation across selected models.
"""

import pandas as pd
import itertools
from datasets import load_dataset
from pathlib import Path
import kaggle_benchmarks as kbench
from knowdobench_task import evaluate_clinical_case

def generate_conditions(factors_dict: dict) -> list[str]:
    """
    Generates all combination IDs for a given factor space.
    e.g., ['R0_A0_U0_O0', 'R1_A0_U0_O0', ...]
    """
    # Exclude _meta if it exists in your config
    clean_factors = {k: v for k, v in factors_dict.items() if k != "_meta"}
    
    # Extract prefixes and levels
    # e.g., {'Role': ['R0', 'R1'], 'Authority': ['A0', 'A1']}
    level_lists = []
    for factor, levels in clean_factors.items():
        prefix = factor[0].upper() # R, A, U, O
        # Filter out _meta inside the factor dict
        valid_levels = [f"{prefix}{level_key}" for level_key in levels.keys() if level_key != "_meta"]
        level_lists.append(valid_levels)
        
    # Cartesian product of all levels
    combinations = list(itertools.product(*level_lists))
    return ["_".join(combo) for combo in combinations]

def main():
    # 1. Load the dataset directly from Hugging Face
    print("Loading KnowDoBench from Hugging Face Hub...")
    # This pulls the 'flat' configuration or the default split
    dataset = load_dataset("sammydman/KnowDoBench")
    
    # Convert to pandas for the Kaggle SDK processing
    df_base = dataset['train'].to_pandas()    
    # 2. Generate the 16 context conditions
    import json
    with open("configs/components.json", "r") as f:
        components = json.load(f)
    conditions = generate_conditions(components)
    
    # 3. Create the cross-product DataFrame (Dataset x Conditions)
    # This turns 200 base cases into 3,200 evaluation rows
    print(f"Applying {len(conditions)} conditions to {len(df_base)} base cases...")
    df_conditions = pd.DataFrame({"condition_id": conditions})
    eval_df = df_base.merge(df_conditions, how="cross")
    
    print(f"Total evaluation rows: {len(eval_df)}")

    # 4. Define the models to evaluate
    models_to_test = [
        "google/gemini-1.5-pro",
        "anthropic/claude-3-sonnet"
    ]
    
    # 5. Execute via Kaggle Benchmarks SDK
    print("Starting evaluation...")
    results_df = kbench.evaluate(
        eval_df,
        task=evaluate_clinical_case,
        models=models_to_test,
        max_workers=8  # Parallelize API calls
    )
    
    # 6. Save raw responses for the taxonomy classifier
    output_path = Path("/kaggle/working/raw_responses.jsonl")
    results_df.to_json(output_path, orient="records", lines=True)
    print(f"Evaluation complete. Raw responses saved to {output_path}")

if __name__ == "__main__":
    main()
