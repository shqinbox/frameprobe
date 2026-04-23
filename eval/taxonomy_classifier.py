"""
taxonomy_classifier.py

Batch, cache-aware LLM-as-a-judge system to identify failure modes (e.g., the "Know-Do Gap").
Reads failure definitions from configs/taxonomy.yaml.
Uses Kaggle's kbench SDK for parallel execution.
"""

import json
import re
import time
from pathlib import Path
import yaml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import kaggle_benchmarks as kbench

class BatchTaxonomyClassifier:
    def __init__(self, taxonomy_config_path: str):
        """Loads the taxonomy schema and builds the base prompt template."""
        with open(taxonomy_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        self.system_instructions = config.get("system_instructions", "")
        self.format_instructions = config.get("format_instructions", "")
        self.categories = config.get("categories", {})
        
        # Build the dynamic category string for the prompt
        cat_strings = [f"- {k}: {v}" for k, v in self.categories.items()]
        self.categories_text = "\n".join(cat_strings)
        
    def build_prompt(self, row: pd.Series) -> str:
        """Assembles the prompt for a specific failed row."""
        return f"""{self.system_instructions}

=== SCENARIO ===
{row.get('scenario', 'UNKNOWN')}

=== TASK ===
{row.get('task', 'UNKNOWN')}

=== MODEL'S RAW OUTPUT ===
{str(row.get('raw_text', ''))[:2000]}

=== CATEGORIES ===
Classify into EXACTLY ONE of these categories:
{self.categories_text}

=== INSTRUCTIONS ===
{self.format_instructions}"""

    def classify_one(self, row: pd.Series, judge_model: str = "anthropic/claude-sonnet-4-6@default") -> dict:
        """Executes a single classification call to the Kaggle SDK."""
        try:
            prompt = self.build_prompt(row)
            
            with kbench.chats.new("classify") as chat:
                judge = kbench.llms[judge_model]
                resp = str(judge.prompt(prompt, temperature=0.0)).strip()
                
            reason_match = re.search(r"REASON:\s*(.+)", resp)
            cat_match = re.search(r"CATEGORY:\s*([A-Z_]+)", resp)
            
            mode = cat_match.group(1).strip() if cat_match else "UNCLASSIFIED"
            
            # Validate against loaded taxonomy
            if mode not in self.categories:
                mode = "UNCLASSIFIED"
                
            return {
                "failure_reason": reason_match.group(1).strip() if reason_match else "No reason parsed",
                "failure_mode": mode
            }
        except Exception as e:
            return {"failure_reason": f"API Error: {str(e)[:100]}", "failure_mode": "API_ERROR"}

    def run(self, raw_results_path: Path, cache_path: Path, output_path: Path, max_workers: int = 8):
        """Executes the batch processing, joining with cache and multi-threading the rest."""
        print(f"Loading raw responses from {raw_results_path}...")
        
        # 1. Load Raw Results (Assuming JSONL from frameprobe engine)
        all_raw = []
        with open(raw_results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_raw.append(json.loads(line))
                    
        raw_df = pd.DataFrame(all_raw)
        
        # Deduplicate (Keep last run)
        raw_df = raw_df.drop_duplicates(
            subset=["llm", "id", "condition_id"],
            keep="last"
        ).reset_index(drop=True)

        # 2. Load Cache & Broadcast
        if cache_path.exists():
            cache_df = pd.read_csv(cache_path)
            print(f"✅ Loaded {len(cache_df):,} cached classifications.")
        else:
            cache_df = pd.DataFrame(columns=["llm", "id", "condition_id", "failure_mode", "failure_reason"])
            print("ℹ️ No cache found. Running full classification.")

        cache_lookup = cache_df[["llm", "id", "condition_id", "failure_mode", "failure_reason"]].drop_duplicates(
            subset=["llm", "id", "condition_id"], keep="last"
        )

        # Left-join cache
        merged = raw_df.merge(cache_lookup, on=["llm", "id", "condition_id"], how="left")
        
        cached_mask = merged["failure_mode"].notna()
        missing_mask = ~cached_mask
        
        print(f"⏩ Skipping {cached_mask.sum():,} already-classified rows.")
        print(f"🤖 Ready to classify {missing_mask.sum():,} missing rows.")
        
        rows_to_classify = merged[missing_mask].copy().reset_index(drop=True)
        final_rows = merged[cached_mask].copy()
        
        # 3. Parallel API Classification
        if not rows_to_classify.empty:
            print(f"\nStarting parallel classification with {max_workers} workers...")
            start_time = time.time()
            completed = 0

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.classify_one, row): i for i, row in rows_to_classify.iterrows()}
                
                for future in as_completed(futures):
                    i = futures[future]
                    c_data = future.result()
                    
                    rows_to_classify.at[i, "failure_mode"] = c_data["failure_mode"]
                    rows_to_classify.at[i, "failure_reason"] = c_data["failure_reason"]
                    
                    completed += 1
                    if completed % 25 == 0:
                        # Checkpoint save
                        pd.concat([final_rows, rows_to_classify.iloc[:i+1]], ignore_index=True).to_csv(output_path, index=False)
                        elapsed = time.time() - start_time
                        print(f"  Processed {completed}/{len(rows_to_classify)} ({completed/elapsed:.1f} rows/s)")

            final_rows = pd.concat([final_rows, rows_to_classify], ignore_index=True)
            
        # 4. Finalization
        final_rows.to_csv(output_path, index=False)
        print(f"\n✅ Done! Saved {len(final_rows):,} rows to {output_path}")
        print("\nFailure Mode Distribution:")
        print(final_rows["failure_mode"].value_counts())

# If run directly as a script
if __name__ == "__main__":
    classifier = BatchTaxonomyClassifier(taxonomy_config_path="configs/taxonomy.yaml")
    
    # Example paths - in Kaggle, these would map to /kaggle/working/ and /kaggle/input/
    classifier.run(
        raw_results_path=Path("results/raw_responses.jsonl"),
        cache_path=Path("results/failure_modes_cache.csv"),
        output_path=Path("results/failure_modes_final.csv")
    )
