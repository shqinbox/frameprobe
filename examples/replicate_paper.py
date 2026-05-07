"""
replicate_paper.py

Reproduces the core findings of "Cannot, Should Not, Did Anyway" (NeurIPS 2026).

Runs taxonomy classification on raw Kaggle output and generates the paper's
primary statistics:
  - Accuracy by pressure condition (Figure 1 — Table form with Wilson CIs)
  - Collapse metrics: DAR and PRI per model (Table 1)
  - McNemar tests: knowledge probe vs coercive, neutral vs coercive (Table 2)
  - Normative failure taxonomy: sycophantic vs rationalized compliance (Figure 2)
  - Epistemic vs normative violation rates (Know-Do Gap)

Usage:
  python examples/replicate_paper.py
  python examples/replicate_paper.py --config experiments/clinical_coercion_sequential.yaml
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Replicate FrameProbe NeurIPS findings.")
    parser.add_argument(
        "--config",
        default="experiments/clinical_coercion_sequential.yaml",
        help="Path to the experiment YAML (default: sequential config).",
    )
    parser.add_argument(
        "--raw-results",
        default=None,
        help="Path to raw_responses.jsonl (defaults to output_dir from config).",
    )
    args = parser.parse_args()

    print("=== REPLICATING FRAME PROBE NeurIPS FINDINGS ===\n")

    from configs.experiment_config import ExperimentConfig
    config = ExperimentConfig.from_yaml(args.config)

    output_dir = Path(config.execution.output_dir)
    raw_results = Path(args.raw_results) if args.raw_results else output_dir / "raw_responses.jsonl"
    classified_csv = output_dir / "failure_modes_final.csv"
    cache_csv = output_dir / "failure_modes_cache.csv"

    # Step 1: Taxonomy classification
    if raw_results.exists():
        print(f"1. Running Taxonomy Classifier on {raw_results} ...")
        from eval.taxonomy_classifier import BatchTaxonomyClassifier
        classifier = BatchTaxonomyClassifier(taxonomy_dict=config.get_taxonomy_dict())
        classifier.run(raw_results, cache_csv, classified_csv)
    elif classified_csv.exists():
        print(f"1. raw_responses.jsonl not found — using existing classified results at {classified_csv}.")
    else:
        print(f"\nError: No evaluation results found.")
        print(f"  Expected raw responses at:   {raw_results}")
        print(f"  Expected classified CSV at:  {classified_csv}")
        print(f"\nRun the evaluation on Kaggle first:")
        print(f"  python -m benchmarks.run_experiment {args.config}")
        return

    print(f"\n2. Generating Paper Statistics from {classified_csv} ...\n")

    from eval.analysis import FrameProbeAnalyzer
    analyzer = FrameProbeAnalyzer.from_config(config, str(classified_csv))

    # --- Table / Figure equivalents from paper ---

    # Overall capacity vs enforcement
    analyzer.print_overall_performance()

    # Figure 1 equivalent: accuracy by pressure condition with Wilson 95% CIs
    analyzer.print_pressure_ladder_table()                       # full benchmark
    analyzer.print_pressure_ladder_table(track_filter="epistemic")
    analyzer.print_pressure_ladder_table(track_filter="normative")

    # Table 1: DAR and PRI per model
    analyzer.print_collapse_metrics()

    # Know-Do Gap: epistemic vs normative violation rates
    analyzer.print_track_comparison()

    # Figure 2 equivalent: normative failure taxonomy
    analyzer.print_taxonomy_breakdown()

    # Table 2: McNemar tests (BH-corrected)
    analyzer.print_mcnemar_results()  # knowledge_probe vs coercive (primary contrast)
    if len(analyzer.pressure_order) >= 3:
        analyzer.print_mcnemar_results(  # neutral vs coercive (secondary contrast)
            condition_a=analyzer.pressure_order[1],
            condition_b=analyzer.pressure_order[-1],
        )

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
