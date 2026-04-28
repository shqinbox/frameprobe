"""
local_runner.py

Runs FrameProbe evaluation locally using either a HuggingFace transformers model
or litellm as the inference backend.

Usage (via run_experiment.py):
    python -m benchmarks.run_experiment experiments/my_experiment.yaml \
        --hf-model google/gemma-2b-it

Usage (programmatic):
    from benchmarks.local_runner import LocalRunner
    runner = LocalRunner(hf_model="google/gemma-2b-it", ...)
    results_df = runner.run(eval_df, models=["google/gemma-2b-it"])
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Global HuggingFace model state (lazy-loaded on first use)
# ---------------------------------------------------------------------------

HF_MODEL = None
HF_TOKENIZER = None


def _load_hf_model(model_id: str):
    global HF_MODEL, HF_TOKENIZER
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[HF] Loading {model_id} ...")
    HF_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    HF_MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"[HF] Model ready.")


# ---------------------------------------------------------------------------
# LocalRunner
# ---------------------------------------------------------------------------

class LocalRunner:
    """
    A lightweight evaluation runner that replaces kbench for local / HF inference.

    Args:
        hf_model: HuggingFace model ID.  If set, loads the model via transformers.
                  If None, falls back to litellm for inference.
        components_dict: The assembled components dict from the experiment config.
        raw_output_path: Path to write raw JSONL responses (mirrors kbench behaviour).
        max_new_tokens: Maximum tokens to generate per response.
    """

    def __init__(
        self,
        hf_model: Optional[str] = None,
        components_dict: Optional[dict] = None,
        raw_output_path: Optional[Path] = None,
        max_new_tokens: int = 512,
    ):
        self.hf_model_id = hf_model
        self.components_dict = components_dict or {}
        self.raw_output_path = Path(raw_output_path) if raw_output_path else None
        self.max_new_tokens = max_new_tokens

        if hf_model is not None:
            _load_hf_model(hf_model)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_one_hf(self, prompt: str) -> str:
        """Generate a response using the loaded HuggingFace model."""
        import torch

        inputs = HF_TOKENIZER(prompt, return_tensors="pt").to(HF_MODEL.device)
        with torch.no_grad():
            output = HF_MODEL.generate(**inputs, max_new_tokens=self.max_new_tokens)
        response = HF_TOKENIZER.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response

    def _run_one_litellm(self, prompt: str, model: str) -> str:
        """Generate a response using litellm (fallback path)."""
        import litellm

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _run_one(self, prompt: str, model: str) -> str:
        """Dispatch to HF or litellm depending on configuration."""
        if HF_MODEL is not None:
            return self._run_one_hf(prompt)
        return self._run_one_litellm(prompt, model)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(self, row: pd.Series, condition_id: str) -> str:
        """
        Assemble a prompt from the row data and condition_id.
        Uses the engine assembler when components_dict is available,
        otherwise falls back to a minimal template.
        """
        try:
            from engine.assembler import PromptAssembler
            assembler = PromptAssembler(self.components_dict)
            return assembler.assemble(
                scenario=row.get("scenario", ""),
                task=row.get("task", ""),
                condition_id=condition_id,
            )
        except Exception:
            scenario = row.get("scenario", "")
            task = row.get("task", "")
            return f"{scenario}\n\n{task}"

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(
        self,
        eval_df: pd.DataFrame,
        models: list,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        """
        Evaluate every (model, row) pair in eval_df.

        Returns a DataFrame with columns mirroring the kbench output schema
        so the rest of the pipeline (taxonomy, analysis) works unchanged.
        """
        model_list = models if models else [self.hf_model_id or "local"]

        tasks = []
        for model in model_list:
            for _, row in eval_df.iterrows():
                tasks.append((model, row))

        records = []
        _raw_fh = None
        if self.raw_output_path is not None:
            self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            _raw_fh = open(self.raw_output_path, "a")

        def _process(model_row):
            model, row = model_row
            condition_id = row.get("condition_id", "")
            row_id = row.get("id", "")
            prompt = self._build_prompt(row, condition_id)
            try:
                response = self._run_one(prompt, model)
                error = None
            except Exception as exc:
                response = ""
                error = str(exc)

            record = {
                "llm": model,
                "id": row_id,
                "condition_id": condition_id,
                "prompt": prompt,
                "response": response,
                "error": error,
                **{k: v for k, v in row.items() if k not in ("condition_id",)},
            }
            return record

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, t): t for t in tasks}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                if _raw_fh is not None:
                    _raw_fh.write(json.dumps(record, default=str) + "\n")
                    _raw_fh.flush()

        if _raw_fh is not None:
            _raw_fh.close()

        return pd.DataFrame(records)
