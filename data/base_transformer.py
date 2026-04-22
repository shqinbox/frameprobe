"""
base_transformer.py

Provides the generic BaseTransformer class for Frame Probe.
Domain researchers should subclass this to map their raw dataset 
into the strict Frame Probe schema.
"""

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable, Union

class BaseTransformer(ABC):
    def __init__(self, domain: str):
        """
        Args:
            domain: A string representing the domain (e.g., 'clinical_medicine', 'contract_law').
        """
        self.domain = domain
        # defaultdict allows researchers to define arbitrary tracks 
        # without breaking the framework's counting logic.
        self.stats = defaultdict(int)

    def make_record(
        self,
        id_: str,
        track: str,
        scenario: str,
        task: str,
        expected_answerable: bool,
        expected_answer: Any,
        evaluator: Union[str, Callable],
        tags: Optional[List[str]] = None,
        is_variant: bool = False,
        variant_id: int = 0,
        base_id: Optional[str] = None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assembles and strictly formats one flat Frame Probe record."""
        
        # If a custom evaluator function is passed, store its string name for JSON serialization
        evaluator_ref = evaluator.__name__ if callable(evaluator) else evaluator

        return {
            "id": id_,
            "base_id": base_id if base_id else id_,
            "is_variant": is_variant,
            "variant_id": variant_id,
            "domain": self.domain,
            "track": track,
            "tags": tags or [],
            "scenario": scenario,
            "task": task,
            "expected_answerable": expected_answerable,
            "expected_answer": str(expected_answer) if expected_answer is not None else None,
            "tolerance": tolerance,
            "evaluator": evaluator_ref,
        }

    @abstractmethod
    def transform_record(self, raw_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        USER IMPLEMENTED: Convert a single raw JSON record into one or more 
        flat Frame Probe records using self.make_record().
        """
        pass

    def run(self, input_path: str, output_path: str):
        """Executes the transformation pipeline."""
        with open(input_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        all_rows = []
        for raw in raw_records:
            try:
                rows = self.transform_record(raw)
                for row in rows:
                    self.stats[row["track"]] += 1
                all_rows.extend(rows)
            except Exception as e:
                print(f"WARNING: Error processing record {raw.get('id', 'UNKNOWN')}: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        print(f"\nTransformation Complete. Wrote {len(all_rows)} rows to {output_path}")
        for track, count in self.stats.items():
            print(f"  {track}: {count}")
