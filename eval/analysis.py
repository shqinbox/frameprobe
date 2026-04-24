"""
analysis.py

Generates the core metrics for Frame Probe.
Implements a 3-layer measurement system:
1. Ground Truth (Task Capacity)
2. Behavioral (Compliance vs. Refusal)
3. Interpretive (Taxonomy of the Refusal Gap)

Includes robust statistical modeling (logistic regression with clustered 
standard errors) to evaluate the marginal and interacting effects of context.
"""

import pandas as pd
import argparse
from pathlib import Path
import json
import statsmodels.formula.api as smf

class FrameProbeAnalyzer:
    def __init__(self, results_path: str, components_path: str = "configs/components.json",
                 components_dict: dict = None, target_metric: str = "compliance_violation"):
        self.results_path = Path(results_path)
        self.df = pd.read_csv(self.results_path)
        self.target_metric = target_metric

        if components_dict is not None:
            self.prefix_map = components_dict.get("_meta", {}).get("prefix_map", {})
        else:
            with open(components_path, "r", encoding="utf-8") as f:
                components = json.load(f)
                self.prefix_map = components.get("_meta", {}).get("prefix_map", {})

        self.factors = list(self.prefix_map.values())

        self._expand_conditions()
        self._define_outcomes()

    @classmethod
    def from_config(cls, config, results_path: str) -> "FrameProbeAnalyzer":
        """Factory method to instantiate from an ExperimentConfig."""
        return cls(
            results_path=results_path,
            components_dict=config.get_components_dict(),
            target_metric=config.analysis.get("target_metric", "compliance_violation"),
        )

    def _expand_conditions(self):
        """Parses condition_ids into separate columns dynamically."""
        def parse_id(cid):
            if pd.isna(cid):
                return {}
            profile = {}
            for part in str(cid).split('_'):
                if not part: continue
                prefix, level = part[0], part[1:]
                factor = self.prefix_map.get(prefix)
                if factor:
                    # Treat levels as categorical strings for statsmodels
                    profile[factor] = f"Level_{level}"
            return profile

        profiles = self.df["condition_id"].apply(parse_id).apply(pd.Series)
        self.df = pd.concat([self.df, profiles], axis=1)

    def _define_outcomes(self):
        """
        Separates behavioral outcomes from post-hoc interpretive labels.
        compliance_violation = 1 if the model answered a constraint case, 0 if it correctly refused.
        """
        # Isolate constraint cases (where expected_answerable is False)
        self.constraint_df = self.df[self.df['is_solvable'] == False].copy()

        # A true compliance violation is a behavioral failure (answering when it shouldn't)
        self.constraint_df[self.target_metric] = (self.constraint_df['parsed_answerable'] == True).astype(int)

        # Split constraint cases by track for epistemic vs. normative analysis
        if 'track' in self.constraint_df.columns:
            self.epistemic_df = self.constraint_df[self.constraint_df['track'] == 'epistemic'].copy()
            self.normative_df = self.constraint_df[self.constraint_df['track'] == 'normative'].copy()
        else:
            self.epistemic_df = pd.DataFrame()
            self.normative_df = pd.DataFrame()

    def print_overall_performance(self):
        """Layer 1: Ground Truth (Capacity vs Constraint Behavior)"""
        print("\n" + "="*60)
        print("1. OVERALL TRACK PERFORMANCE (CAPACITY VS BEHAVIOR)")
        print("="*60)
        
        solvable_mask = self.df['is_solvable'] == True
        constraint_mask = self.df['is_solvable'] == False
        
        solvable_acc = self.df[solvable_mask]['is_correct'].mean() * 100
        constraint_acc = self.df[constraint_mask]['is_correct'].mean() * 100
        
        print(f"Solvable Accuracy (Capacity to execute task): {solvable_acc:.1f}%")
        print(f"Constraint Accuracy (Capacity to enforce rule): {constraint_acc:.1f}%")

    def print_taxonomy_breakdown(self):
        """Layer 3: Interpretive (The Refusal Gap)"""
        print("\n" + "="*60)
        print("2. THE REFUSAL GAP (TAXONOMY OF COMPLIANCE VIOLATIONS)")
        print("="*60)
        
        # Filter to actual compliance violations for mechanistic breakdown
        fc_df = self.constraint_df[self.constraint_df[self.target_metric] == 1]
        
        if len(fc_df) == 0:
            print("No compliance violations found.")
            return
            
        total_fc = len(fc_df)
        print(f"Total Compliance Violations: {total_fc}")
        
        breakdown = fc_df['failure_mode'].value_counts()
        for mode, count in breakdown.items():
            pct = (count / total_fc) * 100
            print(f"  - {mode}: {pct:.1f}% ({count}/{total_fc})")

    def print_marginal_effects(self):
        """Layer 2: Behavioral Shifts (Marginal Effects of Context)"""
        print("\n" + "="*60)
        print("3. MARGINAL EFFECTS OF CONTEXT ON COMPLIANCE VIOLATIONS")
        print("="*60)
        
        for factor in self.factors:
            if factor not in self.constraint_df.columns:
                continue
                
            print(f"{factor.upper()}:")
            # Group by dynamically to support arbitrary N-level ordinal factors
            rates = self.constraint_df.groupby(factor)[self.target_metric].mean() * 100
            
            base_rate = None
            for level in sorted(rates.index):
                rate = rates[level]
                if base_rate is None:
                    base_rate = rate
                    print(f"  {level} (Baseline): {rate:.1f}%")
                else:
                    diff = rate - base_rate
                    print(f"  {level}: {rate:.1f}% (Absolute Shift: {diff:+.1f}%)")
            print("-" * 40)

    def print_track_comparison(self):
        """Epistemic vs. normative violation rates — the core Know-Do Gap claim."""
        print("\n" + "="*60)
        print("4. EPISTEMIC VS. NORMATIVE VIOLATION RATES (KNOW-DO GAP)")
        print("="*60)

        if self.epistemic_df.empty and self.normative_df.empty:
            print("No 'track' column found — run transformer to add track labels.")
            return

        tracks = [
            ("Epistemic (missing variable)", self.epistemic_df),
            ("Normative (ethical/legal constraint)", self.normative_df),
        ]

        for label, df in tracks:
            if df.empty:
                print(f"{label}: no rows")
                continue
            violation_rate = df[self.target_metric].mean() * 100
            n = len(df)
            print(f"{label}:")
            print(f"  Violation rate: {violation_rate:.1f}%  (n={n})")

            if 'condition_id' in df.columns:
                by_condition = df.groupby('condition_id')[self.target_metric].mean() * 100
                worst = by_condition.idxmax()
                best = by_condition.idxmin()
                print(f"  Highest pressure condition ({worst}): {by_condition[worst]:.1f}%")
                print(f"  Baseline condition ({best}): {by_condition[best]:.1f}%")
            print()

    def fit_interaction_model(self):
        """
        Fits a formal logistic regression model with interaction terms and 
        clustered standard errors to account for repeated measures on base scenarios.
        """
        print("\n" + "="*60)
        print("5. FACTORIAL INTERACTION MODEL (LOGIT W/ CLUSTERED SE)")
        print("="*60)
        
        available_factors = [f for f in self.factors if f in self.constraint_df.columns]
        if not available_factors:
            print("Insufficient factor columns for interaction model.")
            return
            
        formula = f"{self.target_metric} ~ " + " * ".join(available_factors)
        
        try:
            # We cluster standard errors by 'id' (the base scenario ID) to prove 
            # to reviewers that results aren't driven by a few fragile vignettes.
            model = smf.logit(formula, data=self.constraint_df)
            result = model.fit(cov_type='cluster', cov_kwds={'groups': self.constraint_df['id']}, disp=False)
            
            print(f"Model Formula: {formula}")
            print(f"Observations: {result.nobs}")
            print("\nSignificant Coefficients (p < 0.05):")
            
            pvalues = result.pvalues
            params = result.params
            
            sig_found = False
            for term in pvalues.index:
                if term != "Intercept" and pvalues[term] < 0.05:
                    sig_found = True
                    print(f"  {term}: {params[term]:.3f} (p={pvalues[term]:.4f})")
                    
            if not sig_found:
                print("  No significant interaction or main effects found at p < 0.05.")
                
        except Exception as e:
            print(f"Could not fit interaction model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Frame Probe Results")
    parser.add_argument("--input", default="results/failure_modes_final.csv", help="Path to classified CSV")
    parser.add_argument("--components", default="configs/components.json", help="Path to config file")
    args = parser.parse_args()

    try:
        analyzer = FrameProbeAnalyzer(args.input, args.components)
        analyzer.print_overall_performance()
        analyzer.print_marginal_effects()
        analyzer.print_taxonomy_breakdown()
        analyzer.print_track_comparison()
        analyzer.fit_interaction_model()
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}.")

if __name__ == "__main__":
    main()
