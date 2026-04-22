"""
validate_dataset.py

Validates the flattened JSONL dataset against the strict Frame Probe schema.
Run this before uploading to Hugging Face or executing the evaluation loop.

Run:
    python validate_dataset.py
    python validate_dataset.py --input custom_flat.jsonl
"""

import json
import argparse
import sys
from typing import Any, Dict

# The strict schema definition required by Frame Probe
REQUIRED_FIELDS = {
    "id": str,
    "base_id": str,
    "is_variant": bool,
    "variant_id": int,
    "domain": str,
    "track": str,
    "tags": list,
    "scenario": str,
    "task": str,
    "expected_answerable": bool,
    "expected_answer": (str, type(None)), # We cast answers to str in the transformer
    "tolerance": (float, int, type(None)),
    "evaluator": str,
}

def validate_record(record: Dict[str, Any], line_num: int) -> list[str]:
    """Validates a single record and returns a list of error strings."""
    errors = []
    
    # 1. Check for missing or extra fields
    record_keys = set(record.keys())
    required_keys = set(REQUIRED_FIELDS.keys())
    
    missing = required_keys - record_keys
    if missing:
        errors.append(f"Line {line_num}: Missing required fields: {missing}")
        
    # 2. Check data types
    for key, expected_type in REQUIRED_FIELDS.items():
        if key in record:
            val = record[key]
            if not isinstance(val, expected_type):
                errors.append(f"Line {line_num}: Field '{key}' must be {expected_type}, got {type(val)}")
                
    # 3. Logical constraint checks
    if record.get("expected_answerable") is False:
        if record.get("expected_answer") is not None:
            errors.append(f"Line {line_num}: 'expected_answerable' is False, but 'expected_answer' is not None.")
            
    if record.get("is_variant") is True:
        if record.get("variant_id", 0) <= 0:
            errors.append(f"Line {line_num}: 'is_variant' is True, but 'variant_id' is <= 0.")
            
    if record.get("is_variant") is False:
        if record.get("variant_id") != 0:
            errors.append(f"Line {line_num}: 'is_variant' is False, but 'variant_id' is not 0.")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate Frame Probe JSONL datasets.")
    parser.add_argument("--input", default="knowdobench_flat.jsonl", help="Path to the JSONL file to validate.")
    args = parser.parse_args()

    errors = []
    seen_ids = set()
    total_records = 0

    print(f"Validating {args.input}...")

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                total_records += 1
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                    continue
                
                # Check for duplicate IDs
                record_id = record.get("id")
                if record_id in seen_ids:
                    errors.append(f"Line {line_num}: Duplicate ID found: '{record_id}'")
                if record_id:
                    seen_ids.add(record_id)

                # Validate schema and logic
                record_errors = validate_record(record, line_num)
                errors.extend(record_errors)

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)

    # Output results
    print("-" * 40)
    if errors:
        print(f"❌ Validation FAILED with {len(errors)} errors:")
        # Only print the first 20 errors to avoid console spam
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)
    else:
        print(f"✅ Validation PASSED! All {total_records} records conform to the Frame Probe schema.")
        sys.exit(0)

if __name__ == "__main__":
    main()
