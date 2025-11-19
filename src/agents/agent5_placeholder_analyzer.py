"""Agent 5: Placeholder analysis and transformation rule generation."""
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from .base_agent import BaseAgent
from ..core.exceptions import MappingError
from ..core.config import config


class Agent5_PlaceholderAnalyzer(BaseAgent):
    """
    Analyzes placeholders and generates transformation rules.
    Implements REQ-A5-01: Context-aware rule generation
    """

    def __init__(self):
        super().__init__("agent5_placeholder_analyzer", temperature=0.0)

        api_key = config.api_key
        if api_key:
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                model_name=config.model_name,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "top_k": 1,
                    "response_mime_type": "application/json"
                }
            )
        else:
            self.model = None

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze placeholders and generate transformation rules.

        Args:
            input_data: {
                "mappings": List of mapping dicts,
                "placeholders": List of placeholder dicts,
                "cv_data": Dict of CV data
            }

        Returns:
            {
                "transformation_rules": Dict[placeholder_id, rules],
                "analysis_metadata": Dict
            }
        """
        mappings = input_data.get("mappings", [])
        placeholders = input_data.get("placeholders", [])
        cv_data = input_data.get("cv_data", {})

        if not mappings:
            raise MappingError("No mappings provided")

        # Generate transformation rules for each mapping
        transformation_rules: Dict[str, Dict[str, Any]] = {}

        for mapping in mappings:
            placeholder_id = mapping["placeholder_id"]

            # Find placeholder details
            placeholder = next(
                (ph for ph in placeholders if ph["placeholder_id"] == placeholder_id),
                None
            )

            if not placeholder:
                continue

            # Generate rules
            rules = self._generate_rules_for_placeholder(
                mapping,
                placeholder,
                cv_data
            )

            transformation_rules[placeholder_id] = rules

        self.logger.info(f"Generated {len(transformation_rules)} transformation rules")

        return {
            "transformation_rules": transformation_rules,
            "analysis_metadata": {
                "total_rules": len(transformation_rules),
                "avg_max_chars": (
                    sum(r.get("max_chars", 0) for r in transformation_rules.values())
                    / len(transformation_rules) if transformation_rules else 0
                )
            }
        }

    def _generate_rules_for_placeholder(
        self,
        mapping: Dict[str, Any],
        placeholder: Dict[str, Any],
        cv_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate transformation rules for a specific placeholder."""

        # Base rules from placeholder properties
        rules: Dict[str, Any] = {
            "placeholder_id": placeholder["placeholder_id"],
            "source_field": mapping["source_field"],
            "max_chars": placeholder.get("max_chars", 500),
            "tone": "professional",
            "formatting": {
                "bold_company": False,
                "italic_dates": False,
                "bullet_points": False
            },
            "transformation_type": "direct"  # direct, summarize, reformat, expand
        }

        # Analyze placeholder context to determine rules
        context = placeholder.get("current_text", "").lower()
        shape_name = placeholder.get("shape_name", "").lower()

        # Rule 1: Detect if it should be bullet points
        if any(
            keyword in context or keyword in shape_name
            for keyword in ["bullet", "list", "responsibilities", "achievements"]
        ):
            rules["formatting"]["bullet_points"] = True
            rules["transformation_type"] = "bullet_list"

        # Rule 2: Detect if company names should be bold
        if any(
            keyword in context or keyword in shape_name
            for keyword in ["experience", "company", "employer"]
        ):
            rules["formatting"]["bold_company"] = True

        # Rule 3: Detect if dates should be italic
        if any(
            keyword in context or keyword in shape_name
            for keyword in ["date", "when", "period", "duration"]
        ):
            rules["formatting"]["italic_dates"] = True

        # Rule 4: Determine if summarization is needed
        source_data = self._get_source_data(cv_data, mapping["source_field"])
        if source_data and isinstance(source_data, (list, str)):
            source_length = len(str(source_data))
            if source_length > rules["max_chars"] * 2:
                rules["transformation_type"] = "summarize"

        # Rule 5: Tone adjustment based on placeholder position
        slide_index = placeholder.get("slide_index", 0)
        if slide_index == 0:
            rules["tone"] = "formal"  # First slide more formal
        elif slide_index >= 5:
            rules["tone"] = "concise"  # Later slides more concise

        # Rule 6: Active voice for responsibility sections
        if "responsibilities" in mapping["source_field"]:
            rules["tone"] = "active"
            rules["voice_conversion"] = True

        return rules

    def _get_source_data(self, cv_data: Dict[str, Any], field_path: str) -> Any:
        """Extract data from CV dict using field path."""
        parts = field_path.split('.')
        current: Any = cv_data

        try:
            for part in parts:
                if '[' in part and ']' in part:
                    key = part.split('[')[0]
                    index = int(part.split('[')[1].split(']')[0])
                    current = current[key][index]
                else:
                    current = current[part]
            return current
        except (KeyError, IndexError, TypeError):
            return None
