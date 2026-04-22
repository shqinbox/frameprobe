"""
knowdobench_transformer.py

Clinical instantiation of the Frame Probe dataset transformer.
Maps clinical records from Template_Cases.json into the framework schema.
"""

import re
from base_transformer import BaseTransformer

class KnowDoBenchTransformer(BaseTransformer):
    def __init__(self):
        super().__init__(domain="clinical_medicine")

    def derive_tolerance(self, answer) -> float | None:
        """Clinical heuristic for numeric tolerance."""
        if answer is None:
            return None
        if isinstance(answer, (int, float)):
            return 0.1 if abs(answer) < 10 else 1.0
        return None

    def split_task(self, note: str) -> tuple[str, str]:
        """Splits a clinical note into (scenario, task), capturing various task phrasings."""
        match = re.search(r'([^.]*[Cc]alculate[^.]+\.)\s*$', note.strip())
        if match:
            return note[:match.start()].strip(), match.group(1).strip()
        
        parts = note.strip().rsplit('. ', 1)
        if len(parts) == 2:
            return parts[0].strip() + '.', parts[1].strip()
        
        return note.strip(), ""

    def transform_record(self, raw: dict) -> list[dict]:
        rows = []
        category = raw.get("category")
        base_id = raw["id"]
        instrument = raw.get("instrument", "")
        
        if category == "formula":
            # 1. Solvable Base Case
            scenario, task = self.split_task(raw["complete_note"])
            ans = raw.get("ground_truth")
            rows.append(self.make_record(
                id_=base_id, track="solvable", scenario=scenario, task=task,
                expected_answerable=True, expected_answer=ans, evaluator="numeric",
                tags=[instrument], tolerance=self.derive_tolerance(ans)
            ))

            # 2. Epistemic Variants (Missing Variables)
            for idx, (removed_var, note) in enumerate(raw.get("removable", {}).items(), 1):
                scen_ep, task_ep = self.split_task(note)
                rows.append(self.make_record(
                    id_=f"{base_id}__missing_{removed_var}", track="epistemic",
                    scenario=scen_ep, task=task_ep, expected_answerable=False,
                    expected_answer=None, evaluator="exact_match", tags=[instrument, removed_var],
                    is_variant=True, variant_id=idx, base_id=base_id
                ))

        elif category == "normative_control":
            sub = "PHYSICAL_HARM" if raw.get("subcategory") == "FLOOR_ANCHOR" else raw.get("subcategory", "")
            
            if "clinical_vignette" in raw and "calculation_task" in raw:
                scenario, task = raw["clinical_vignette"], raw["calculation_task"]
            else:
                scenario, task = self.split_task(raw["complete_note"])
                
            rows.append(self.make_record(
                id_=base_id, track="normative", scenario=scenario, task=task,
                expected_answerable=False, expected_answer=None, evaluator="exact_match",
                tags=[instrument, sub] if sub else [instrument]
            ))
            
        return rows
