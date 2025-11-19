"""Agent 6: Deterministic content transformation with hard limits."""
import re
import json
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
from datetime import datetime

from .base_agent import BaseAgent
from ..core.schemas import TransformationResult
from ..core.validators import TransformationValidator
from ..core.exceptions import TransformationError
from ..core.config import config


class Agent6_ContentTransformer(BaseAgent):
    """
    Content Transformation with Hard Limits and Mixed Formatting.
    Implements REQ-A6-01, REQ-A6-02, REQ-CON-01, REQ-CON-02
    """

    def __init__(self):
        super().__init__("agent6_content_transformer", temperature=0.0)

        api_key = config.api_key
        if api_key:
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                model_name=config.model_name,
                generation_config={
                    "temperature": 0.0,  # Deterministic (REQ-DET-01)
                    "top_p": 0.1,
                    "top_k": 1
                }
            )
        else:
            self.model = None
            self.logger.warning("Gemini API not configured")

        self.transformation_log: List[TransformationResult] = []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform content according to rules.

        Args:
            input_data: {
                "content_mappings": List[Dict] with source_text and rules,
                "cv_data": Dict of CV data
            }

        Returns:
            {
                "transformed_content": List[Dict],
                "traceability_matrix": List[Dict],
                "transformation_summary": Dict
            }
        """
        content_mappings = input_data.get("content_mappings", [])
        cv_data = input_data.get("cv_data", {})

        if not content_mappings:
            raise TransformationError("No content mappings provided")

        transformed_results = []

        for mapping in content_mappings:
            # Extract source content
            source_field = mapping.get("source_field")
            source_text = self._extract_source_text(cv_data, source_field)

            if not source_text:
                continue

            # Apply transformation
            result = self.transform_content(
                source_text=str(source_text),
                rules=mapping.get("rules", {}),
                placeholder_id=mapping.get("placeholder_id", "unknown")
            )

            transformed_results.append(result)

        # Generate traceability matrix
        traceability = self.generate_traceability_matrix()

        self.logger.info(
            f"Transformed {len(transformed_results)} content items, "
            f"{sum(1 for r in transformed_results if r.compliance_status == 'full')} "
            f"fully compliant"
        )

        return {
            "transformed_content": [r.to_dict() for r in transformed_results],
            "traceability_matrix": traceability,
            "transformation_summary": {
                "total_transformations": len(transformed_results),
                "full_compliance": sum(
                    1 for r in transformed_results
                    if r.compliance_status == "full"
                ),
                "partial_compliance": sum(
                    1 for r in transformed_results
                    if r.compliance_status == "partial"
                )
            }
        }

    def transform_content(
        self,
        source_text: str,
        rules: Dict[str, Any],
        placeholder_id: str
    ) -> TransformationResult:
        """
        Apply transformation rules with strict enforcement (REQ-A6-01).
        """
        result = TransformationResult(
            placeholder_id=placeholder_id,
            original_text=source_text,
            transformed_text="",
            rules_applied=rules,
            compliance_status="full",
            transformations=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "original_length": len(source_text)
            }
        )

        transformed_text = source_text

        # Step 1: Apply tone/voice transformations (REQ-CON-02)
        if rules.get("tone") == "active" or rules.get("voice_conversion"):
            transformed_text = self._convert_to_active_voice(transformed_text)
            result.transformations.append({
                "type": "tone",
                "action": "convert_to_active_voice",
                "before_length": len(source_text),
                "after_length": len(transformed_text)
            })

        # Step 2: Apply transformation type
        transformation_type = rules.get("transformation_type", "direct")

        if transformation_type == "bullet_list":
            transformed_text = self._convert_to_bullet_list(transformed_text)
            result.transformations.append({
                "type": "format",
                "action": "convert_to_bullet_list"
            })

        elif transformation_type == "summarize":
            max_chars = rules.get("max_chars", 500)
            transformed_text = self._summarize_text(transformed_text, max_chars)
            result.transformations.append({
                "type": "summarize",
                "action": "ai_summarization",
                "target_length": max_chars
            })

        # Step 3: Apply hard character limit (REQ-A6-01)
        max_chars = rules.get("max_chars")
        if max_chars:
            transformed_text, was_truncated = self._enforce_hard_limit(
                transformed_text,
                max_chars
            )

            if was_truncated:
                result.compliance_status = "partial"
                result.transformations.append({
                    "type": "truncation",
                    "action": "hard_limit_enforcement",
                    "original_length": len(source_text),
                    "final_length": len(transformed_text),
                    "max_chars": max_chars,
                    "truncated": True
                })

        # Step 4: Apply formatting (REQ-CON-01, REQ-A6-02)
        formatted_text = self._apply_mixed_formatting(
            transformed_text,
            rules.get("formatting", {})
        )

        if formatted_text != transformed_text:
            result.transformations.append({
                "type": "formatting",
                "action": "apply_markdown",
                "formats": rules.get("formatting", {})
            })

        transformed_text = formatted_text

        # Final validation
        is_compliant, violations = TransformationValidator.validate_transformation_compliance(
            source_text,
            transformed_text,
            rules
        )

        if not is_compliant:
            result.compliance_status = "failed"
            result.metadata["violations"] = violations

        result.transformed_text = transformed_text
        result.metadata["final_length"] = len(transformed_text)
        result.metadata["transformations_count"] = len(result.transformations)

        # Log transformation for traceability
        self.logger.log_transformation(
            transformation_type="content_transform",
            original=source_text[:200],
            transformed=transformed_text[:200],
            metadata={"placeholder_id": placeholder_id, "rules": rules}
        )

        # Add to transformation log
        self.transformation_log.append(result)

        return result

    def _convert_to_active_voice(self, text: str) -> str:
        """
        Convert passive voice to active voice deterministically (REQ-CON-02).
        """
        if not self.model:
            return self._rule_based_active_voice(text)

        # Try AI conversion
        prompt = f"""
Convert ALL passive voice sentences to active voice. Keep facts and meaning exact.

RULES:
- "Was led by me" → "Led"
- "Projects were managed" → "Managed projects"
- "Was responsible for" → "Responsible for" or "Led"
- Remove wordiness
- Keep all technical terms
- Maintain bullet point format if present

TEXT:
{text}

OUTPUT ONLY THE CONVERTED TEXT. NO EXPLANATIONS. NO MARKDOWN CODE BLOCKS.
"""

        try:
            response = self.model.generate_content(prompt)
            converted = response.text.strip()

            # Validate conversion didn't change meaning significantly
            if len(converted) > len(text) * 1.5:
                # If significantly longer, use rule-based fallback
                return self._rule_based_active_voice(text)

            return converted
        except Exception:
            return self._rule_based_active_voice(text)

    def _rule_based_active_voice(self, text: str) -> str:
        """Rule-based passive to active voice conversion."""
        # Common passive patterns
        replacements = [
            (r'\bwas led by\b', 'led'),
            (r'\bwere led by\b', 'led'),
            (r'\bwas managed by\b', 'managed'),
            (r'\bwere managed by\b', 'managed'),
            (r'\bwas responsible for\b', 'led'),
            (r'\bwere responsible for\b', 'led'),
            (r'\bwas developed by\b', 'developed'),
            (r'\bwere developed by\b', 'developed'),
        ]

        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _convert_to_bullet_list(self, text: str) -> str:
        """Convert text to bullet point format."""
        # Split by common delimiters
        sentences = re.split(r'[.;]\s+', text)

        # Filter empty and very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= 1:
            return text  # Don't convert single items

        # Create bullet list
        bullets = [f"• {s.strip()}" for s in sentences]
        return '\n'.join(bullets)

    def _summarize_text(self, text: str, target_length: int) -> str:
        """
        Iteratively summarize until target length is met.
        """
        if len(text) <= target_length:
            return text

        if not self.model:
            # Fallback to truncation
            truncated, _ = self._enforce_hard_limit(text, target_length)
            return truncated

        prompt = f"""
Summarize this text to MAXIMUM {target_length} characters.
Preserve key facts, achievements, and technical details.
Be concise but complete.

TEXT:
{text}

OUTPUT ONLY THE SUMMARY. NO EXPLANATIONS. NO MARKDOWN CODE BLOCKS.
"""

        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()

            # If still too long, truncate smartly
            if len(summary) > target_length:
                summary, _ = self._enforce_hard_limit(summary, target_length)

            return summary
        except Exception:
            # Fallback to truncation
            truncated, _ = self._enforce_hard_limit(text, target_length)
            return truncated

    def _enforce_hard_limit(self, text: str, max_chars: int) -> Tuple[str, bool]:
        """
        Enforce character limit with smart truncation (REQ-A6-01).
        Returns: (truncated_text, was_truncated)
        """
        if len(text) <= max_chars:
            return text, False

        # Strategy 1: If bullets, remove last bullets until fits
        if '•' in text or '\n-' in text:
            lines = text.split('\n')
            while len('\n'.join(lines)) > max_chars and len(lines) > 1:
                lines.pop()
            if len('\n'.join(lines)) <= max_chars:
                return '\n'.join(lines), True

        # Strategy 2: Truncate at sentence boundary
        truncated = text[:max_chars]

        # Find last sentence ending
        last_period = truncated.rfind('.')
        last_semicolon = truncated.rfind(';')
        last_newline = truncated.rfind('\n')

        cut_point = max(last_period, last_semicolon, last_newline)

        # Only use sentence boundary if we're not losing too much
        if cut_point > max_chars * 0.75:
            return truncated[:cut_point + 1], True

        # Strategy 3: Truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:
            return truncated[:last_space] + "...", True

        # Final fallback: Hard truncate
        return truncated[:max_chars - 3] + "...", True

    def _apply_mixed_formatting(
        self, text: str, formatting_rules: Dict[str, Any]
    ) -> str:
        """
        Apply consistent formatting with markdown (REQ-CON-01, REQ-A6-02).
        """
        result = text

        # Apply bold to company names
        if formatting_rules.get("bold_company"):
            result = self._bold_company_names(result)

        # Apply italic to dates
        if formatting_rules.get("italic_dates"):
            result = self._italicize_dates(result)

        # Ensure bullet points are consistent
        if formatting_rules.get("bullet_points"):
            result = self._normalize_bullet_points(result)

        return result

    def _bold_company_names(self, text: str) -> str:
        """Apply bold formatting to company names."""
        # Pattern for company names (simplified)
        patterns = [
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Group|Company)))\b',
            r'\b(McKinsey|Deloitte|PwC|EY|KPMG|Accenture|BCG|Bain)\b'
        ]

        result = text
        for pattern in patterns:
            # Only bold if not already bold
            result = re.sub(
                pattern,
                lambda m: (
                    f"**{m.group(1)}**"
                    if not text[max(0, m.start()-2):m.start()] == '**'
                    else m.group(1)
                ),
                result
            )

        return result

    def _italicize_dates(self, text: str) -> str:
        """Italicize all date patterns."""
        date_patterns = [
            r'\b(\d{4}[-/]\d{2})\b',  # YYYY-MM
            r'\b(\d{1,2}[-/]\d{4})\b',  # MM-YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
            r'\b\d{4}\s*[-–]\s*(?:\d{4}|Present|Current)\b'  # Year ranges
        ]

        result = text
        for pattern in date_patterns:
            # Only italicize if not already italicized
            result = re.sub(
                pattern,
                lambda m: (
                    f"*{m.group(0)}*"
                    if not text[max(0, m.start()-1):m.start()] == '*'
                    else m.group(0)
                ),
                result,
                flags=re.IGNORECASE
            )

        return result

    def _normalize_bullet_points(self, text: str) -> str:
        """Normalize bullet point formatting."""
        # Convert various bullet styles to consistent format
        lines = text.split('\n')
        normalized = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                normalized.append('')
                continue

            # Remove existing bullets
            stripped = re.sub(r'^[-*•◦▪▫]\s*', '', stripped)

            # Add consistent bullet if line has content
            if stripped:
                normalized.append(f"• {stripped}")
            else:
                normalized.append('')

        return '\n'.join(normalized)

    def _extract_source_text(self, cv_data: Dict[str, Any], field_path: str) -> Any:
        """Extract text from CV data using field path."""
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

    def generate_traceability_matrix(self) -> List[Dict[str, Any]]:
        """
        Generate complete traceability matrix (REQ-EXP-01, REQ-EXP-02).
        """
        matrix = []

        for transformation in self.transformation_log:
            entry = {
                "placeholder_id": transformation.placeholder_id,
                "source_text_preview": (
                    transformation.original_text[:100] + "..."
                    if len(transformation.original_text) > 100
                    else transformation.original_text
                ),
                "transformed_text_preview": (
                    transformation.transformed_text[:100] + "..."
                    if len(transformation.transformed_text) > 100
                    else transformation.transformed_text
                ),
                "rules_applied": transformation.rules_applied,
                "transformations": transformation.transformations,
                "compliance_status": transformation.compliance_status,
                "original_length": transformation.metadata.get("original_length", 0),
                "final_length": transformation.metadata.get("final_length", 0),
                "timestamp": transformation.metadata.get("timestamp")
            }
            matrix.append(entry)

        return matrix
