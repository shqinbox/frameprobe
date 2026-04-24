"""
run_experiment.py

Generic YAML-driven orchestrator for FrameProbe experiments.
Reads a single YAML config and runs the full pipeline:
  data loading → condition generation → evaluation → taxonomy → analysis.

CLI:   python -m benchmarks.run_experiment experiments/clinical_coercion_v1.yaml
Code:  from benchmarks.run_experiment import run_pipeline
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd


def generate_conditions(prompt_factors: dict) -> list:
    """
    Generates all factorial condition IDs from a prompt_factors config.

    Uses _meta.prefix_map to resolve prefixes. Falls back to first-char-upper
    if no prefix_map is defined (backward compat with run_benchmark.py).

    Returns e.g. ['A0_P0', 'A0_P1', 'A1_P0', 'A1_P1']
    """
    meta = prompt_factors.get("_meta", {})
    prefix_map = meta.get("prefix_map", {})

    # Invert prefix_map: factor_name -> prefix_char
    factor_to_prefix = {v: k for k, v in prefix_map.items()}

    clean_factors = {k: v for k, v in prompt_factors.items() if k != "_meta"}

    level_lists = []
    for factor, levels in clean_factors.items():
        prefix = factor_to_prefix.get(factor, factor[0].upper())
        valid_levels = [
            f"{prefix}{level_key}"
            for level_key in levels.keys()
            if level_key != "_meta"
        ]
        level_lists.append(valid_levels)

    combinations = list(itertools.product(*level_lists))
    return ["_".join(combo) for combo in combinations]


def run_pipeline(config, skip_taxonomy: bool = False, skip_analysis: bool = False) -> pd.DataFrame:
    """
    Execute the full FrameProbe pipeline from an ExperimentConfig.

    Args:
        config: An ExperimentConfig instance.
        skip_taxonomy: If True, skip the taxonomy classification phase.
        skip_analysis: If True, skip the statistical analysis phase.

    Returns:
        The evaluation results DataFrame.
    """
    from datasets import load_dataset
    import kaggle_benchmarks as kbench

    output_dir = Path(config.execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Load dataset ---
    print(f"[1/6] Loading dataset from '{config.data.source}' (split={config.data.split})...")
    dataset = load_dataset(config.data.source)
    df_base = dataset[config.data.split].to_pandas()

    if config.data.max_rows is not None:
        df_base = df_base.head(config.data.max_rows)
        print(f"  Truncated to {len(df_base)} rows (max_rows={config.data.max_rows})")

    # --- Phase 2: Generate factorial conditions ---
    components_dict = config.get_components_dict()
    conditions = generate_conditions(components_dict)
    print(f"[2/6] Generated {len(conditions)} factorial conditions: {conditions}")

    df_conditions = pd.DataFrame({"condition_id": conditions})
    eval_df = df_base.merge(df_conditions, how="cross")
    # DEBUG SLICE — remove for full run
    eval_df = eval_df[eval_df['id'] == 'friedewald_ldl_01'].reset_index(drop=True)
    print(f"⚠️  DEBUG SLICE ACTIVE: {len(eval_df)} rows")

    print(f"  Total evaluation rows: {len(eval_df)}")

    # --- Phase 3: Configure the task module ---
    from benchmarks.knowdobench_task import configure, evaluate_clinical_case
    configure(components_dict)
    print("[3/6] Task module configured with experiment components.")

    # --- Phase 4: Run kbench evaluation ---
    print(f"[4/6] Running evaluation across {len(config.models)} model(s)...")

    assert len(eval_df) > 0, (
        "eval_df is empty — check your debug slice ID exists in the dataset. "
        f"Available IDs: {df_base['id'].tolist()[:10]}"
    )

    import json as _json
    raw_output_path = output_dir / "raw_responses.jsonl"
    checkpoint_path = output_dir / "checkpoint.jsonl"

    # Resume: find which (model, condition_id, id) triples are already done
    completed = set()
    resume_path = checkpoint_path if checkpoint_path.exists() else raw_output_path
    if resume_path.exists():
        with open(resume_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _json.loads(line)
                    completed.add((r.get("llm"), r.get("condition_id"), r.get("id")))
                except Exception:
                    pass
        if completed:
            print(f"  Resuming: {len(completed)} rows already done, skipping.")

    all_results = []

    for model_str in config.models:
        model_eval_df = eval_df[
            ~eval_df.apply(
                lambda row: (model_str, row["condition_id"], row["id"]) in completed, axis=1
            )
        ].reset_index(drop=True)

        if model_eval_df.empty:
            print(f"  {model_str}: all rows already done, skipping.")
            continue

        print(f"  {model_str}: evaluating {len(model_eval_df)} rows...")
        try:
            runs = evaluate_clinical_case.evaluate(
                llm=kbench.llms[model_str],
                evaluation_data=model_eval_df,
                n_jobs=config.execution.max_workers,
                max_attempts=3,
                retry_delay=5,
                timeout=config.execution.timeout,
            )
            model_df = runs.as_dataframe()
            all_results.append(model_df)

            # Checkpoint after each model completes
            with open(checkpoint_path, "a") as f:
                for _, row in model_df.iterrows():
                    f.write(_json.dumps(row.to_dict()) + "\n")
            print(f"  {model_str}: done, checkpoint saved.")

        except Exception as e:
            print(f"  {model_str}: failed with {e}. Skipping to next model.")
            continue

    if not all_results:
        print("  All rows already completed. Loading from checkpoint...")
        results_df = pd.read_json(str(resume_path), orient="records", lines=True)
    else:
        results_df = pd.concat(all_results, ignore_index=True)

    kbench_output_path = output_dir / "kbench_results.jsonl"
    results_df.to_json(str(kbench_output_path), orient="records", lines=True)
    print(f"  Saved {len(results_df)} kbench results to {kbench_output_path}")
    print(f"  Scored responses (error_type etc.) are in {raw_output_path} via log_raw_response()")


    # --- Phase 5: Taxonomy classification (optional) ---
    if not skip_taxonomy:
        print("[5/6] Running taxonomy classification...")
        from eval.taxonomy_classifier import BatchTaxonomyClassifier

        classifier = BatchTaxonomyClassifier(taxonomy_dict=config.get_taxonomy_dict())
        cache_path = output_dir / "failure_modes_cache.csv"
        classified_path = output_dir / "failure_modes_final.csv"

        classifier.run(
            raw_results_path=raw_output_path,
            cache_path=cache_path,
            output_path=classified_path,
            max_workers=config.execution.max_workers,
        )
    else:
        print("[5/6] Taxonomy classification skipped.")
        classified_path = None

    # --- Phase 6: Statistical analysis (optional) ---
    if not skip_analysis and classified_path and classified_path.exists():
        print("[6/6] Running statistical analysis...")
        from eval.analysis import FrameProbeAnalyzer

        analyzer = FrameProbeAnalyzer.from_config(config, str(classified_path))
        analyzer.print_overall_performance()
        analyzer.print_marginal_effects()
        analyzer.print_taxonomy_breakdown()
        analyzer.print_track_comparison()
        analyzer.fit_interaction_model()
    else:
        print("[6/6] Statistical analysis skipped.")

    print(f"\nExperiment '{config.name}' complete.")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run a YAML-driven FrameProbe experiment."
    )
    parser.add_argument("config", help="Path to the experiment YAML file.")
    parser.add_argument("--skip-taxonomy", action="store_true", help="Skip taxonomy classification.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip statistical analysis.")
    args = parser.parse_args()

    from configs.experiment_config import ExperimentConfig

    config = ExperimentConfig.from_yaml(args.config)
    run_pipeline(config, skip_taxonomy=args.skip_taxonomy, skip_analysis=args.skip_analysis)


if __name__ == "__main__":
    main()
