"""
assembler.py

The core, domain-agnostic prompt engine for Frame Probe.
Assembles full LLM prompts by combining a fixed scenario and task 
with a dynamic instruction wrapper (context factors) dictated by a configuration file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, Tuple, Optional

if TYPE_CHECKING:
    from configs.experiment_config import ExperimentConfig


class PromptAssembler:
    def __init__(self, components: Dict[str, dict]):
        """
        Initializes the assembler with the prompt fragment library.
        
        Args:
            components: A dictionary mapping Factor -> Level -> Text Fragment.
                        Must optionally contain a "_meta" key for generic configuration.
        """
        self.components = components
        self.meta = self.components.get("_meta", {})
        
        # Extract generic routing/ordering from config
        self.prefix_map = self.meta.get("prefix_map", {})
        
        # Exclude _meta from factors if order isn't explicitly defined
        default_factors = [k for k in self.components.keys() if k != "_meta"]
        self.order = self.meta.get("order", default_factors)
        
        # Configurable structural labels (prevents hidden framing tokens)
        self.labels = self.meta.get("labels", {
            "instructions": "Instructions:",
            "context": "Context:",
            "task": "Task:"
        })
        
        self.output_format = self.meta.get("output_format", "")

    @classmethod
    def from_file(cls, components_path: str) -> "PromptAssembler":
        """Factory method to instantiate from a JSON components file."""
        with open(components_path, "r", encoding="utf-8") as f:
            components = json.load(f)
        return cls(components)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> PromptAssembler:
        """Factory method to instantiate from an ExperimentConfig."""
        return cls(config.get_components_dict())

    def parse_condition_id(self, condition_id: str) -> Dict[str, str]:
        """
        Parses a shorthand condition ID into a factor profile dynamically.
        """
        profile = {}
        parts = condition_id.split('_')
        
        for part in parts:
            if not part:
                continue
            
            # Assume 1-char prefix, rest is level (e.g., "R1" -> prefix "R", level "1")
            prefix, level = part[0], part[1:]
            
            factor_name = self.prefix_map.get(prefix)
            if not factor_name:
                raise ValueError(f"Unknown prefix '{prefix}' in condition '{condition_id}'. Please check '_meta.prefix_map'.")
            
            profile[factor_name] = level
                
        return profile

    def build_wrappers(self, factor_profile: Dict[str, str]) -> Tuple[str, str]:
        """
        Assembles pre-task and post-task context by concatenating fragments.
        Enforces strict validation against the defined component library.
        
        Returns:
            Tuple[str, str]: (pre_task_wrapper, post_task_wrapper)
        """
        pre_fragments = []
        post_fragments = []
        
        for factor in self.order:
            level = factor_profile.get(factor)
            
            # Silent skip if factor is completely absent from profile (allows partial condition testing)
            if level is None:
                continue
                
            # Strict Validation: Fail fast on invalid definitions
            if factor not in self.components:
                raise ValueError(f"Factor '{factor}' defined in profile but missing from components library.")
            
            factor_dict = self.components[factor]
            if level not in factor_dict:
                raise ValueError(f"Invalid level '{level}' for factor '{factor}'. Allowed levels: {list(factor_dict.keys())}")
                
            fragment = factor_dict[level]
            
            # Process fragment if it's not empty or null
            if fragment and not isinstance(fragment, dict): # Skip the nested _meta dict if present
                fragment = fragment.strip()
                
                # Check for positional instructions (default to "pre")
                factor_meta = factor_dict.get("_meta", {})
                position = factor_meta.get("position", "pre")
                
                if position == "pre":
                    pre_fragments.append(fragment)
                elif position == "post":
                    post_fragments.append(fragment)
                else:
                    raise ValueError(f"Invalid position '{position}' for factor '{factor}'. Must be 'pre' or 'post'.")
                    
        return " ".join(pre_fragments), " ".join(post_fragments)

    def assemble(self, scenario: str, task: str, condition_id: str) -> str:
        """
        Constructs the fully assembled prompt.
        
        Args:
            scenario: The domain vignette or context string.
            task: The specific instruction string.
            condition_id: The identifier for the context condition (e.g., 'R1_A1_U0_O0').
            
        Returns:
            The complete string prompt ready to be sent to the LLM.
        """
        factor_profile = self.parse_condition_id(condition_id)
        pre_wrap, post_wrap = self.build_wrappers(factor_profile)
        
        prompt_parts = []
        
        # 1. Pre-task Context (e.g., Role, Authority)
        if pre_wrap:
            prompt_parts.append(f"{self.labels['instructions']}\n{pre_wrap}\n")
            
        # 2. Base Content
        prompt_parts.append(f"{self.labels['context']}\n{scenario}\n")
        prompt_parts.append(f"{self.labels['task']}\n{task}")
        
        # 3. Post-task Context (e.g., Urgency/Obligation if defined as 'post')
        if post_wrap:
            prompt_parts.append(f"\n{post_wrap}")
            
        # 4. Enforced Output Schema
        if self.output_format:
            prompt_parts.append(f"\n{self.output_format}")
            
        return "\n".join(prompt_parts).strip()
