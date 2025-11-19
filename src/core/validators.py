"""Validation framework for CV data and transformations."""
from typing import Dict, Any, Tuple, List
import re
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError

from .schemas import CVData, WorkExperience, CV_JSON_SCHEMA


class DataValidator:
    """Static validation methods for data integrity."""

    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            validate(instance=data, schema=schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    @staticmethod
    def validate_cv_data(cv_data: CVData) -> Tuple[bool, List[str]]:
        """
        Validate complete CV data structure.

        Args:
            cv_data: CVData object to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate personal info
        if not cv_data.personal_info.full_name:
            errors.append("Personal info: full_name is required")

        # Validate work experience
        for idx, exp in enumerate(cv_data.work_experience):
            exp_errors = DataValidator._validate_work_experience(exp, idx)
            errors.extend(exp_errors)

        # Validate education
        for idx, edu in enumerate(cv_data.education):
            if not edu.institution or not edu.degree or not edu.field:
                errors.append(
                    f"Education[{idx}]: institution, degree, and field are required"
                )

        return len(errors) == 0, errors

    @staticmethod
    def _validate_work_experience(exp: WorkExperience, idx: int) -> List[str]:
        """Validate individual work experience entry."""
        errors = []

        # Required fields
        if not exp.company_name:
            errors.append(f"WorkExperience[{idx}]: company_name is required")
        if not exp.job_title:
            errors.append(f"WorkExperience[{idx}]: job_title is required")
        if not exp.start_date:
            errors.append(f"WorkExperience[{idx}]: start_date is required")

        # Date format validation (YYYY-MM)
        date_pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
        if exp.start_date and not re.match(date_pattern, exp.start_date):
            errors.append(
                f"WorkExperience[{idx}]: start_date must be in YYYY-MM format, got '{exp.start_date}'"
            )

        if exp.end_date and not re.match(date_pattern, exp.end_date):
            errors.append(
                f"WorkExperience[{idx}]: end_date must be in YYYY-MM format or null, got '{exp.end_date}'"
            )

        # Temporal consistency (REQ-SEM-01)
        if exp.is_current and exp.end_date:
            errors.append(
                f"WorkExperience[{idx}]: is_current=True but end_date is set"
            )

        # Date logic validation
        if exp.start_date and exp.end_date:
            try:
                start = datetime.strptime(exp.start_date, "%Y-%m")
                end = datetime.strptime(exp.end_date, "%Y-%m")
                if end < start:
                    errors.append(
                        f"WorkExperience[{idx}]: end_date cannot be before start_date"
                    )
            except ValueError as e:
                errors.append(
                    f"WorkExperience[{idx}]: date parsing error - {str(e)}"
                )

        return errors

    @staticmethod
    def validate_temporal_consistency(work_experiences: List[WorkExperience]) -> Tuple[bool, List[str]]:
        """
        Validate temporal consistency across work experiences.
        REQ-SEM-01: Temporal anchoring verification

        Args:
            work_experiences: List of work experience entries

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for multiple current roles
        current_roles = [exp for exp in work_experiences if exp.is_current]
        if len(current_roles) > 1:
            errors.append(
                f"Temporal inconsistency: Found {len(current_roles)} current roles, expected 0 or 1"
            )

        # Check date ordering (most recent first)
        for i in range(len(work_experiences) - 1):
            current_exp = work_experiences[i]
            next_exp = work_experiences[i + 1]

            try:
                current_start = datetime.strptime(current_exp.start_date, "%Y-%m")
                next_start = datetime.strptime(next_exp.start_date, "%Y-%m")

                if next_start > current_start:
                    errors.append(
                        f"Work experiences not in chronological order: "
                        f"{next_exp.company_name} ({next_exp.start_date}) listed after "
                        f"{current_exp.company_name} ({current_exp.start_date})"
                    )
            except (ValueError, TypeError):
                # Skip if dates can't be parsed
                pass

        return len(errors) == 0, errors

    @staticmethod
    def validate_character_limit(text: str, max_chars: int) -> bool:
        """
        Validate that text meets character limit.
        REQ-A6-01: Hard character limit enforcement

        Args:
            text: Text to validate
            max_chars: Maximum allowed characters

        Returns:
            True if text is within limit
        """
        return len(text) <= max_chars

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not email:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format (basic)."""
        if not phone:
            return False
        # Remove common formatting
        cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
        # Check if it's mostly digits
        return len(cleaned) >= 10 and cleaned.replace('+', '').isdigit()

    @staticmethod
    def validate_date_format(date_str: str, format_str: str = "%Y-%m") -> bool:
        """Validate date string format."""
        if not date_str:
            return False
        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False


class TransformationValidator:
    """Validator for content transformations."""

    @staticmethod
    def validate_transformation_compliance(
        original: str,
        transformed: str,
        rules: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Validate that transformation complies with rules.

        Args:
            original: Original text
            transformed: Transformed text
            rules: Transformation rules applied

        Returns:
            Tuple of (compliance_status, violations)
            compliance_status: "full", "partial", "failed"
            violations: List of rule violations
        """
        violations = []

        # Check character limit (REQ-A6-01)
        max_chars = rules.get('max_characters')
        if max_chars and len(transformed) > max_chars:
            violations.append(
                f"Character limit violation: {len(transformed)} > {max_chars}"
            )

        # Check if text was modified when required
        if rules.get('must_transform', False) and original == transformed:
            violations.append("Transformation required but text unchanged")

        # Check for forbidden patterns
        forbidden = rules.get('forbidden_patterns', [])
        for pattern in forbidden:
            if re.search(pattern, transformed, re.IGNORECASE):
                violations.append(f"Forbidden pattern found: {pattern}")

        # Determine compliance status
        if len(violations) == 0:
            return "full", []
        elif len(violations) <= 2:
            return "partial", violations
        else:
            return "failed", violations

    @staticmethod
    def validate_format_requirements(
        text: str,
        requirements: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate text meets formatting requirements.

        Args:
            text: Text to validate
            requirements: Format requirements

        Returns:
            Tuple of (is_valid, missing_requirements)
        """
        missing = []

        # Check for bullet points
        if requirements.get('requires_bullets', False):
            if not re.search(r'[•\-\*]', text):
                missing.append("Bullet points required")

        # Check for bold markers (markdown)
        if requirements.get('requires_bold', False):
            if not re.search(r'\*\*.+?\*\*', text):
                missing.append("Bold formatting required")

        # Check for italic markers (markdown)
        if requirements.get('requires_italic', False):
            if not re.search(r'\*.+?\*', text):
                missing.append("Italic formatting required")

        return len(missing) == 0, missing
