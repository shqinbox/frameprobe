"""
experiment_config.py

Dataclass-based configuration loader for YAML-driven experiments.
A researcher edits ONE YAML file to control the entire pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset source and loading parameters."""
    source: str  # HuggingFace dataset ID, e.g. "sammydman/KnowDoBench"
    split: str = "train"
    max_rows: Optional[int] = None  # None = use full dataset


@dataclass
class ExecutionConfig:
    """Runtime parameters for the kbench evaluation loop."""
    max_workers: int = 8
    max_attempts: int = 3
    retry_delay: int = 5
    timeout: int = 180
    output_dir: str = "results"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration loaded from a YAML file."""
    name: str
    description: str
    data: DataConfig
    models: List[str]
    prompt_factors: Dict[str, Any]
    taxonomy: Dict[str, Any]
    execution: ExecutionConfig
    analysis: Dict[str, Any] = field(default_factory=lambda: {
        "target_metric": "compliance_violation"
    })

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an experiment configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        """Construct an ExperimentConfig from a raw dictionary."""
        data_cfg = DataConfig(**raw.get("data", {}))
        exec_cfg = ExecutionConfig(**raw.get("execution", {}))

        return cls(
            name=raw.get("name", "unnamed_experiment"),
            description=raw.get("description", ""),
            data=data_cfg,
            models=raw.get("models", []),
            prompt_factors=raw.get("prompt_factors", {}),
            taxonomy=raw.get("taxonomy", {}),
            execution=exec_cfg,
            analysis=raw.get("analysis", {"target_metric": "compliance_violation"}),
        )

    def get_components_dict(self) -> Dict[str, Any]:
        """
        Returns prompt_factors with all factor-level keys normalized to strings.

        YAML parses bare `0:` as integer 0, but PromptAssembler expects string
        keys like "0". This method ensures consistency.
        """
        def _normalize_keys(d):
            if not isinstance(d, dict):
                return d
            return {str(k): _normalize_keys(v) for k, v in d.items()}

        return _normalize_keys(self.prompt_factors)

    def get_taxonomy_dict(self) -> Dict[str, Any]:
        """Returns the taxonomy section as-is for the classifier."""
        return self.taxonomy
