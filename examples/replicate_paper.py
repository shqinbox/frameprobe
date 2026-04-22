"""
replicate_paper.py

Minimal orchestration script to reproduce the core findings of the NeurIPS paper.
It runs the taxonomy classifier on the raw Kaggle output and executes the 
statistical analysis to generate the exact paper metrics.
"""

from pathlib import Path
from eval.taxonomy_classifier import BatchTaxonomyClassifier
from eval.analysis import FrameProbeAnalyzer

def main():
    print("=== REPLICATING FRAME PROBE NEURIPS FINDINGS ===\n")
    
    # Paths to the Kaggle-generated raw results and configurations
    raw_results = Path("results/raw_responses.jsonl")
    classified_csv = Path("results/failure_modes_final.csv")
    cache_csv = Path("results/failure_modes_cache.csv")
    
    # Step 1: Assign Failure Taxonomies (The LLM-as-a-judge layer)
    # The classifier uses cache-aware deduplication, so it runs instantly if already scored.
    if raw_results.exists():
        print("1. Running Taxonomy Classifier on FALSE_COMPLIANCE errors...")
        classifier = BatchTaxonomyClassifier("configs/taxonomy.yaml")
        classifier.run(raw_results, cache_csv, classified_csv)
    else:
        print(f"Warning: {raw_results} not found. Skipping to statistical analysis.")

    # Step 2: Generate Paper Metrics (The Statistical Layer)
    if classified_csv.exists():
        print("\n2. Generating Statistical Tables and Metrics...")
        analyzer = FrameProbeAnalyzer(
            results_path=str(classified_csv),
            components_path="configs/components.json"
        )
        
        # Generates the data for the paper's Tables and Figures
        analyzer.print_overall_performance()   # Maps to: Capacity vs Behavior
        analyzer.print_marginal_effects()      # Maps to: Main Contextual Shifts
        analyzer.print_taxonomy_breakdown()    # Maps to: The Refusal Gap Breakdown
        analyzer.fit_interaction_model()       # Maps to: Robustness & Significance
        
    else:
        print(f"Error: Could not find {classified_csv}.")
        print("Please run benchmarks/run_benchmark.py on Kaggle first.")

if __name__ == "__main__":
    main()
