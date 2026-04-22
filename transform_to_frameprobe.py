"""
transform_to_frameprobe.py

Converts Template_Cases.json to the flat FrameProbe schema and writes
knowdobench_flat.jsonl — ready for HuggingFace upload.

Output rows per input record:
  - formula case   → 1 solvable row  +  N epistemic rows (one per removable variable)
  - normative case → 1 normative row

Run:
    python transform_to_frameprobe.py
    python transform_to_frameprobe.py --input my_cases.json --output my_output.jsonl
"""

import json
import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_task_from_note(complete_note: str) -> tuple[str, str]:
    """
    Splits a complete_note into (scenario, task).
    The task is the final sentence (starts with 'Calculate').
    Falls back to splitting on the last '. ' if no Calculate sentence found.
    """
    # Try to split on last 'Calculate' sentence
    match = re.search(r'(Calculate[^.]+\.)\s*$', complete_note.strip())
    if match:
        task = match.group(1).strip()
        scenario = complete_note[:match.start()].strip()
        return scenario, task

    # Fallback: last sentence
    parts = complete_note.strip().rsplit('. ', 1)
    if len(parts) == 2:
        return parts[0].strip() + '.', parts[1].strip()

    # Can't split — return full note as scenario, empty task
    return complete_note.strip(), ""


def instrument_to_tags(instrument: str, subcategory: str = None) -> list[str]:
    tags = [instrument] if instrument else []
    if subcategory and subcategory not in ("NORMATIVE_UNSOLVABLE",):
        tags.append(subcategory)
    return tags


def make_base_record(
    id_: str,
    track: str,
    instrument: str,
    subcategory: str,
    scenario: str,
    task: str,
    expected_answerable: bool,
    expected_answer,        # float | None
    is_variant: bool = False,
    variant_id: int = 0,
    base_id: str = None,
) -> dict:
    """Assembles one flat FrameProbe record."""
    return {
        # Identity
        "id": id_,
        "base_id": base_id if base_id else id_,
        "is_variant": is_variant,
        "variant_id": variant_id,
        "expert_validated": True,

        # Categorization
        "domain": "clinical_medicine",
        "track": track,
        "tags": instrument_to_tags(instrument, subcategory),

        # Content
        "scenario": scenario,
        "task": task,

        # Ground truth
        "expected_answerable": expected_answerable,
        "expected_answer": str(expected_answer) if expected_answer is not None else None,
        "tolerance": derive_tolerance(expected_answer),
        "evaluator": "numeric" if expected_answerable else "exact_match",
    }


def derive_tolerance(answer) -> float | None:
    """
    Simple heuristic: use 0.1 for answers < 10, 1.0 for answers >= 10.
    Returns None for non-numeric / refusal cases.
    Override manually for cases needing tighter bounds.
    """
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        return 0.1 if abs(answer) < 10 else 1.0
    return None


# ---------------------------------------------------------------------------
# Per-category transformers
# ---------------------------------------------------------------------------

def transform_formula(record: dict) -> list[dict]:
    """
    One formula record → 1 solvable row + N epistemic rows.
    """
    rows = []
    base_id = record["id"]
    instrument = record.get("instrument", "")
    ground_truth = record.get("ground_truth")

    # ── Solvable row (complete_note, all variables present) ──────────────────
    scenario, task = split_task_from_note(record["complete_note"])
    rows.append(make_base_record(
        id_=base_id,
        track="solvable",
        instrument=instrument,
        subcategory=None,
        scenario=scenario,
        task=task,
        expected_answerable=True,
        expected_answer=ground_truth,
        is_variant=False,
        variant_id=0,
        base_id=base_id,
    ))

    # ── Epistemic rows (one per removed variable) ─────────────────────────────
    removable = record.get("removable", {})
    for variant_idx, (removed_var, incomplete_note) in enumerate(removable.items(), start=1):
        # incomplete_note has the task sentence embedded just like complete_note
        scenario_ep, task_ep = split_task_from_note(incomplete_note)
        rows.append(make_base_record(
            id_=f"{base_id}__missing_{removed_var}",
            track="epistemic",
            instrument=instrument,
            subcategory=removed_var,
            scenario=scenario_ep,
            task=task_ep,
            expected_answerable=False,
            expected_answer=None,
            is_variant=True,
            variant_id=variant_idx,
            base_id=base_id,
        ))

    return rows


def transform_normative(record: dict) -> list[dict]:
    """
    One normative record → 1 normative row.
    Handles both field layouts:
      - clinical_vignette + calculation_task  (most normative cases)
      - complete_note only                    (FLOOR_ANCHOR cases)
    """
    base_id = record["id"]
    instrument = record.get("instrument", "")
    subcategory = record.get("subcategory", "")

    if subcategory == "FLOOR_ANCHOR":
        subcategory = "PHYSICAL_HARM"

    if "clinical_vignette" in record and "calculation_task" in record:
        scenario = record["clinical_vignette"]
        task = record["calculation_task"]
    else:
        # FLOOR_ANCHOR: task is the last sentence of complete_note
        scenario, task = split_task_from_note(record["complete_note"])

    return [make_base_record(
        id_=base_id,
        track="normative",
        instrument=instrument,
        subcategory=subcategory,
        scenario=scenario,
        task=task,
        expected_answerable=False,
        expected_answer=None,
        is_variant=False,
        variant_id=0,
        base_id=base_id,
    )]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def transform(input_path: str, output_path: str):
    with open(input_path) as f:
        records = json.load(f)

    all_rows = []
    stats = {"solvable": 0, "epistemic": 0, "normative": 0}

    for record in records:
        category = record.get("category")
        if category == "formula":
            rows = transform_formula(record)
        elif category == "normative_control":
            rows = transform_normative(record)
        else:
            print(f"  WARNING: unknown category '{category}' for id={record.get('id')} — skipping")
            continue

        for row in rows:
            stats[row["track"]] += 1
        all_rows.extend(rows)

    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. Wrote {len(all_rows)} rows to {output_path}")
    print(f"  solvable:  {stats['solvable']}")
    print(f"  epistemic: {stats['epistemic']}")
    print(f"  normative: {stats['normative']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="Template_Cases.json")
    parser.add_argument("--output", default="knowdobench_flat.jsonl")
    args = parser.parse_args()

    transform(args.input, args.output)
