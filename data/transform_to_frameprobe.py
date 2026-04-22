"""
transform_to_frameprobe.py

CLI entry point for converting raw KnowDoBench templates 
into the flat Frame Probe JSONL schema.

Run:
    python transform_to_frameprobe.py
    python transform_to_frameprobe.py --input my_cases.json --output my_output.jsonl
"""

import argparse
from knowdobench_transformer import KnowDoBenchTransformer

def main():
    parser = argparse.ArgumentParser(description="Convert KnowDoBench templates to Frame Probe JSONL.")
    parser.add_argument("--input", default="Template_Cases.json", help="Path to raw dataset.")
    parser.add_argument("--output", default="knowdobench_flat.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    # Instantiate the domain-specific transformer and run the pipeline
    transformer = KnowDoBenchTransformer()
    transformer.run(args.input, args.output)

if __name__ == "__main__":
    main()
