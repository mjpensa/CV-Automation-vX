"""Agent 4: CV-to-Template intelligent mapping."""
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from .base_agent import BaseAgent
from ..core.schemas import CVData, PlaceholderMapping
from ..core.exceptions import MappingError
from ..core.config import config


class Agent4_MappingEngine(BaseAgent):
    """
    Intelligent mapping between CV data and PowerPoint placeholders.
    Implements REQ-A4-01: Context-aware mapping
    """

    def __init__(self):
        super().__init__("agent4_mapping_engine", temperature=0.0)

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
            self.logger.warning("Gemini API not configured")

        self.mapping_log = []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create intelligent mappings between CV and template.

        Args:
            input_data: {
                "cv_data": CVData object or dict,
                "placeholders": List of placeholder dicts,
                "mapping_strategy": str (optional)
            }

        Returns:
            {
                "mappings": List[Dict],
                "mapping_metadata": Dict,
                "unmapped_placeholders": List,
                "unmapped_cv_fields": List
            }
        """
        cv_data = input_data.get("cv_data")
        placeholders = input_data.get("placeholders")

        if not cv_data or not placeholders:
            raise MappingError("Both cv_data and placeholders are required")

        # Convert cv_data to dict if it's an object
        if isinstance(cv_data, CVData):
            cv_data_dict = cv_data.to_dict()
        else:
            cv_data_dict = cv_data

        # Generate mappings using AI or rule-based
        if self.model:
            mappings = self._generate_mappings(cv_data_dict, placeholders)
        else:
            mappings = self._rule_based_mapping(cv_data_dict, placeholders)

        # Validate mappings
        validated_mappings = self._validate_mappings(mappings, cv_data_dict, placeholders)

        # Identify unmapped items
        unmapped_placeholders = self._find_unmapped_placeholders(
            validated_mappings, placeholders
        )
        unmapped_cv_fields = self._find_unmapped_cv_fields(
            validated_mappings, cv_data_dict
        )

        self.logger.info(
            f"Generated {len(validated_mappings)} mappings, "
            f"{len(unmapped_placeholders)} placeholders unmapped"
        )

        return {
            "mappings": validated_mappings,
            "mapping_metadata": {
                "total_placeholders": len(placeholders),
                "mapped_placeholders": len(validated_mappings),
                "mapping_coverage": (
                    len(validated_mappings) / len(placeholders)
                    if placeholders else 0
                )
            },
            "unmapped_placeholders": unmapped_placeholders,
            "unmapped_cv_fields": unmapped_cv_fields
        }

    def _generate_mappings(
        self, cv_data: Dict[str, Any], placeholders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate mappings using AI."""

        # Create simplified CV summary for mapping
        cv_summary = self._create_cv_summary(cv_data)

        # Create placeholder summary
        placeholder_summary = []
        for ph in placeholders:
            placeholder_summary.append({
                "id": ph["placeholder_id"],
                "type": ph["placeholder_type"],
                "context": ph.get("current_text", "")[:100],
                "slide": ph["slide_index"]
            })

        prompt = f"""
TASK: Create mappings between CV data fields and PowerPoint placeholders.

CV DATA FIELDS:
{json.dumps(cv_summary, indent=2)}

PLACEHOLDERS:
{json.dumps(placeholder_summary, indent=2)}

MAPPING RULES:
1. Match placeholder context to CV data semantically
2. Prioritize current/recent work experience for prominent placeholders
3. Map education to education-related placeholders
4. Map skills to skills/expertise placeholders
5. Consider slide order - earlier slides should have more important info

OUTPUT FORMAT (JSON array):
[
  {{
    "placeholder_id": "slide_0_shape_1",
    "source_field": "personal_info.full_name",
    "confidence": 0.95,
    "reasoning": "Placeholder labeled 'Name' matches personal name field"
  }},
  ...
]

OUTPUT ONLY THE JSON ARRAY. NO EXPLANATIONS.
"""

        try:
            response = self.model.generate_content(prompt)
            mappings = json.loads(response.text)
            return mappings
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"AI mapping failed: {e}. Using rule-based fallback.")
            return self._rule_based_mapping(cv_data, placeholders)

    def _create_cv_summary(self, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simplified summary of CV data for mapping."""
        summary: Dict[str, Any] = {
            "personal_info": {
                "full_name": cv_data.get("personal_info", {}).get("full_name"),
                "email": cv_data.get("personal_info", {}).get("email"),
                "location": cv_data.get("personal_info", {}).get("location")
            },
            "work_experience": []
        }

        # Summarize work experience (top 3 roles)
        for idx, exp in enumerate(cv_data.get("work_experience", [])[:3]):
            summary["work_experience"].append({
                "index": idx,
                "company": exp.get("company_name"),
                "title": exp.get("job_title"),
                "is_current": exp.get("is_current"),
                "duration_months": exp.get("duration_months"),
                "responsibilities_count": len(exp.get("responsibilities", []))
            })

        # Summarize education
        summary["education"] = [
            {
                "institution": edu.get("institution"),
                "degree": edu.get("degree")
            }
            for edu in cv_data.get("education", [])[:2]
        ]

        # Summarize skills
        skills = cv_data.get("skills", {})
        summary["skills"] = {
            "technical_count": len(skills.get("technical", [])),
            "certifications_count": len(skills.get("certifications", []))
        }

        return summary

    def _rule_based_mapping(
        self, cv_data: Dict[str, Any], placeholders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback rule-based mapping."""
        mappings = []

        # Rule 1: Map name to first text placeholder on first slide
        name_placeholder = next(
            (ph for ph in placeholders
             if ph["slide_index"] == 0 and ph["placeholder_type"] == "text"),
            None
        )
        if name_placeholder:
            mappings.append({
                "placeholder_id": name_placeholder["placeholder_id"],
                "source_field": "personal_info.full_name",
                "confidence": 0.8,
                "reasoning": "Rule-based: First text placeholder on first slide"
            })

        # Rule 2: Map current job to second text placeholder
        work_exp = cv_data.get("work_experience", [])
        if work_exp:
            job_placeholder = next(
                (ph for ph in placeholders
                 if ph["slide_index"] <= 1 and ph["placeholder_type"] == "text"
                 and ph["placeholder_id"] != name_placeholder.get("placeholder_id", "")),
                None
            )
            if job_placeholder:
                mappings.append({
                    "placeholder_id": job_placeholder["placeholder_id"],
                    "source_field": "work_experience.0.job_title",
                    "confidence": 0.7,
                    "reasoning": "Rule-based: Current/recent job title"
                })

        return mappings

    def _validate_mappings(
        self,
        mappings: List[Dict[str, Any]],
        cv_data: Dict[str, Any],
        placeholders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and clean mappings."""
        validated = []

        for mapping in mappings:
            # Check if placeholder exists
            ph_exists = any(
                ph["placeholder_id"] == mapping["placeholder_id"]
                for ph in placeholders
            )
            if not ph_exists:
                continue

            # Check if source field exists in CV data
            field_exists = self._check_field_exists(cv_data, mapping["source_field"])
            if not field_exists:
                continue

            validated.append(mapping)

        return validated

    def _check_field_exists(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a field path exists in nested dict."""
        parts = field_path.split('.')
        current: Any = data

        for part in parts:
            # Handle array indices
            if '[' in part and ']' in part:
                key = part.split('[')[0]
                index = int(part.split('[')[1].split(']')[0])
                if key not in current or not isinstance(current[key], list):
                    return False
                if index >= len(current[key]):
                    return False
                current = current[key][index]
            else:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]

        return True

    def _find_unmapped_placeholders(
        self,
        mappings: List[Dict[str, Any]],
        placeholders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find placeholders that weren't mapped."""
        mapped_ids = {m["placeholder_id"] for m in mappings}
        return [
            ph for ph in placeholders
            if ph["placeholder_id"] not in mapped_ids
        ]

    def _find_unmapped_cv_fields(
        self,
        mappings: List[Dict[str, Any]],
        cv_data: Dict[str, Any]
    ) -> List[str]:
        """Find CV fields that weren't mapped to any placeholder."""
        mapped_fields = {m["source_field"] for m in mappings}

        # Generate all available fields
        all_fields = []
        all_fields.append("personal_info.full_name")
        all_fields.append("personal_info.email")

        work_exp = cv_data.get("work_experience", [])
        for idx in range(len(work_exp)):
            all_fields.append(f"work_experience.{idx}.company_name")
            all_fields.append(f"work_experience.{idx}.job_title")

        unmapped = [f for f in all_fields if f not in mapped_fields]
        return unmapped
