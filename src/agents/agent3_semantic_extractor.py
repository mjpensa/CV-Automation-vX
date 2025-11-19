"""Agent 3: Deep semantic CV extraction with deterministic processing."""
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from .base_agent import BaseAgent
from ..core.schemas import (
    CVData, PersonalInfo, WorkExperience, Education, Skills,
    Project, DateStatus, ConfidenceLevel, CV_JSON_SCHEMA
)
from ..core.validators import DataValidator
from ..core.exceptions import ExtractionError, SchemaViolationError
from ..core.config import config


class Agent3_SemanticExtractor(BaseAgent):
    """
    Deep Semantic CV Extraction with Deterministic Processing.
    Implements REQ-A3-01, REQ-A3-02, REQ-SEM-01 through REQ-SEM-04
    """

    def __init__(self):
        super().__init__("agent3_semantic_extractor", temperature=0.0)

        # Initialize Gemini with deterministic settings
        api_key = config.api_key
        if api_key:
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                model_name=config.model_name,  # gemini-2.5-flash-preview-09-2025
                generation_config={
                    "temperature": 0.0,  # Zero-variance mandate (REQ-DET-01)
                    "top_p": 0.1,
                    "top_k": 1,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json"
                }
            )
        else:
            self.model = None
            self.logger.warning(
                "Gemini API key not configured. Agent will fail at runtime."
            )

        self.extraction_attempts = []
        self.semantic_enrichment_log = []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract CV data with semantic enrichment.

        Args:
            input_data: {
                "text": str,
                "file_path": str (optional)
            }

        Returns:
            {
                "cv_data": CVData object,
                "cv_data_dict": Dict,
                "extraction_metadata": Dict,
                "semantic_enrichment": Dict
            }
        """
        if not self.model:
            raise ExtractionError(
                "Gemini API not configured. Set GEMINI_API_KEY in .env file"
            )

        cv_text = input_data.get("text")
        if not cv_text:
            raise ExtractionError("CV text is required")

        # Multi-stage extraction with retry
        extracted_data = self._extract_with_retry(cv_text, max_retries=3)

        # Validate extracted data
        is_valid, errors = DataValidator.validate_json_schema(
            extracted_data, CV_JSON_SCHEMA
        )
        if not is_valid:
            self.logger.warning(
                f"Schema validation failed: {errors}. Attempting to fix..."
            )
            # Continue with best effort

        # Apply semantic enrichment
        enriched_data = self._apply_semantic_enrichment(extracted_data, cv_text)

        # Convert to structured objects
        cv_data = self._convert_to_cv_data(enriched_data)

        # Final validation
        is_valid, errors = DataValidator.validate_cv_data(cv_data)
        if not is_valid:
            self.logger.warning(f"CV data validation warnings: {'; '.join(errors)}")

        self.logger.info(
            f"Successfully extracted CV: {cv_data.personal_info.full_name}, "
            f"{len(cv_data.work_experience)} positions"
        )

        return {
            "cv_data": cv_data,
            "cv_data_dict": cv_data.to_dict(),
            "extraction_metadata": {
                "attempts": len(self.extraction_attempts),
                "confidence": cv_data.extraction_confidence.value,
                "timestamp": cv_data.extraction_timestamp
            },
            "semantic_enrichment": self.semantic_enrichment_log
        }

    def _extract_with_retry(self, cv_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract with progressive retry mechanism (REQ-DET-02).
        """
        for attempt in range(max_retries):
            try:
                # Build extraction prompt with increasing strictness
                prompt = self._build_extraction_prompt(cv_text, attempt)

                # Call Gemini API
                response = self.model.generate_content(prompt)

                # Parse JSON response
                extracted_data = json.loads(response.text)

                # Log successful attempt
                self.extraction_attempts.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "prompt_version": f"v{attempt + 1}",
                    "data_keys": list(extracted_data.keys())
                })

                return extracted_data

            except (json.JSONDecodeError, KeyError, Exception) as e:
                self.extraction_attempts.append({
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                })

                if attempt == max_retries - 1:
                    # Final fallback (REQ-DET-03)
                    self.logger.warning(
                        "All extraction attempts failed. Using fallback structure."
                    )
                    return self._create_fallback_structure(cv_text)

        return self._create_fallback_structure(cv_text)

    def _build_extraction_prompt(self, cv_text: str, attempt: int = 0) -> str:
        """
        Build extraction prompt with progressive strictness.
        """
        strictness_levels = [
            "STANDARD",
            "STRICT - Focus on required fields only",
            "ULTRA-STRICT - Use 'N/A' for any ambiguous data"
        ]

        strictness = strictness_levels[min(attempt, len(strictness_levels) - 1)]

        prompt = f"""
EXTRACTION LEVEL: {strictness}

CRITICAL INSTRUCTIONS:
1. Extract CV information into EXACT JSON format following the schema
2. ALL dates MUST be in YYYY-MM format (e.g., "2023-01", "2020-06")
3. For current roles: set end_date to null AND is_current to true
4. Distinguish between employer_industry and client_industries:
   - employer_industry: The industry of the company the person works for
   - client_industries: Industries of clients they served (for consultants/advisors)
5. NO HALLUCINATION - Use "N/A" or null for missing information
6. Extract ALL responsibilities and projects with full detail
7. Identify technical skills, tools, and certifications explicitly

JSON SCHEMA REQUIREMENTS:
{json.dumps(CV_JSON_SCHEMA, indent=2)}

EXTRACTION RULES:
- personal_info.full_name: REQUIRED - Extract full name
- personal_info.email: Extract if present, null if not
- personal_info.phone: Extract if present, null if not
- personal_info.location: City, State, Country if available
- personal_info.linkedin: Full LinkedIn URL if present

- work_experience: Array of positions
  - company_name: REQUIRED - Company name
  - job_title: REQUIRED - Exact job title
  - start_date: REQUIRED - Format YYYY-MM
  - end_date: YYYY-MM or null if current
  - is_current: REQUIRED - Boolean (true if currently employed here)
  - employer_industry: Industry of employer (e.g., "Consulting", "Technology", "Finance")
  - client_industries: Array of industries served (e.g., ["Oil & Gas", "Banking"])
  - responsibilities: Array of detailed responsibility statements
  - projects: Array of project objects with name, description, role, technologies
  - skills_used: Array of technical skills used in this role

- education: Array of degrees
  - institution: University/school name
  - degree: Degree type (BS, MS, PhD, etc.)
  - field: Field of study
  - graduation_date: YYYY-MM format if available

- skills:
  - technical: Programming languages, frameworks, tools
  - soft: Leadership, communication, etc.
  - languages: Spoken languages
  - certifications: Professional certifications
  - tools: Tools and platforms used

CV TEXT TO EXTRACT:
{cv_text}

OUTPUT ONLY VALID JSON. NO EXPLANATIONS. NO MARKDOWN CODE BLOCKS.
"""
        return prompt

    def _apply_semantic_enrichment(
        self, data: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """
        Apply semantic post-processing for temporal and entity relationships.
        Implements REQ-SEM-01 through REQ-SEM-04
        """
        enrichment_log = {
            "temporal_anchoring": [],
            "industry_disambiguation": [],
            "skill_extraction": [],
            "duration_calculations": []
        }

        # Process work experiences
        for idx, exp in enumerate(data.get("work_experience", [])):

            # 1. Temporal Anchoring (REQ-SEM-01)
            temporal_result = self._determine_temporal_status(
                exp.get("start_date"),
                exp.get("end_date"),
                original_text,
                exp.get("company_name")
            )
            exp["is_current"] = temporal_result["is_current"]
            exp["temporal_confidence"] = temporal_result["confidence"]
            exp["date_status"] = temporal_result["status"]

            enrichment_log["temporal_anchoring"].append({
                "experience_index": idx,
                "company": exp.get("company_name"),
                "result": temporal_result
            })

            # 2. Calculate duration
            duration = self._calculate_duration(
                exp.get("start_date"),
                exp.get("end_date"),
                exp.get("is_current")
            )
            exp["duration_months"] = duration

            enrichment_log["duration_calculations"].append({
                "experience_index": idx,
                "duration_months": duration
            })

            # 3. Industry Disambiguation (REQ-SEM-03)
            employer_industry_lower = exp.get("employer_industry", "").lower()
            if any(keyword in employer_industry_lower for keyword in [
                "consulting", "advisory", "professional services"
            ]):
                client_industries = self._extract_client_industries(
                    exp.get("projects", []),
                    exp.get("responsibilities", []),
                    original_text
                )
                exp["client_industries"] = client_industries

                enrichment_log["industry_disambiguation"].append({
                    "experience_index": idx,
                    "employer_industry": exp.get("employer_industry"),
                    "client_industries": client_industries
                })

            # 4. Skill Extraction Enhancement (REQ-SEM-04)
            enhanced_skills = self._enhance_skill_extraction(
                exp.get("skills_used", []),
                exp.get("responsibilities", []),
                exp.get("projects", [])
            )
            exp["skills_used"] = enhanced_skills

            enrichment_log["skill_extraction"].append({
                "experience_index": idx,
                "skills_count": len(enhanced_skills)
            })

        self.semantic_enrichment_log = enrichment_log
        return data

    def _determine_temporal_status(
        self,
        start_date: str,
        end_date: Optional[str],
        context: str,
        company_name: str
    ) -> Dict[str, Any]:
        """
        Deterministic current role detection (REQ-SEM-01).
        Returns: {is_current, confidence, status, reasoning}
        """
        result = {
            "is_current": False,
            "confidence": 0.0,
            "status": "unknown",
            "reasoning": []
        }

        # Check 1: No end date = current
        if not end_date or end_date is None:
            result["is_current"] = True
            result["confidence"] = 0.9
            result["status"] = "current"
            result["reasoning"].append("No end date provided")
            return result

        # Check 2: Explicit current indicators
        current_indicators = ["present", "current", "ongoing", "now", "today"]
        end_date_lower = str(end_date).lower()

        if any(indicator in end_date_lower for indicator in current_indicators):
            result["is_current"] = True
            result["confidence"] = 1.0
            result["status"] = "current"
            result["reasoning"].append(f"End date contains current indicator: {end_date}")
            return result

        # Check 3: Date is in future
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m")
            now = datetime.now()

            if end_dt >= now:
                result["is_current"] = True
                result["confidence"] = 0.95
                result["status"] = "current"
                result["reasoning"].append(
                    f"End date {end_date} is in future or current month"
                )
                return result

            # Check if within last 2 months
            months_diff = (now.year - end_dt.year) * 12 + (now.month - end_dt.month)

            if months_diff <= 2:
                # Check context for current indicators
                context_window = self._extract_context_window(context, company_name, 200)
                if any(
                    indicator in context_window.lower()
                    for indicator in current_indicators
                ):
                    result["is_current"] = True
                    result["confidence"] = 0.8
                    result["status"] = "current"
                    result["reasoning"].append(
                        "Recent end date with current indicators in context"
                    )
                    return result

            # Past role
            result["is_current"] = False
            result["confidence"] = 0.95
            result["status"] = "past"
            result["reasoning"].append(f"End date {end_date} is {months_diff} months ago")
            return result

        except ValueError:
            # Invalid date format - check context
            result["confidence"] = 0.5
            result["status"] = "unknown"
            result["reasoning"].append(f"Invalid date format: {end_date}")

        return result

    def _calculate_duration(
        self,
        start_date: str,
        end_date: Optional[str],
        is_current: bool
    ) -> int:
        """Calculate duration in months."""
        try:
            start = datetime.strptime(start_date, "%Y-%m")

            if is_current or not end_date:
                end = datetime.now()
            else:
                end = datetime.strptime(end_date, "%Y-%m")

            duration = (end.year - start.year) * 12 + (end.month - start.month)
            return max(duration, 0)  # Ensure non-negative

        except (ValueError, TypeError):
            return 0

    def _extract_client_industries(
        self,
        projects: List[Any],
        responsibilities: List[Any],
        context: str
    ) -> List[str]:
        """
        Extract client industries from project context (REQ-SEM-03).
        """
        # Comprehensive industry keyword mapping
        industry_keywords = {
            "Banking & Finance": [
                "bank", "banking", "financial services",
                "investment", "capital markets"
            ],
            "Oil & Gas": [
                "oil", "gas", "petroleum", "energy",
                "upstream", "downstream"
            ],
            "Healthcare": [
                "healthcare", "hospital", "medical",
                "pharma", "pharmaceutical", "biotech"
            ],
            "Technology": ["tech", "software", "saas", "cloud", "digital"],
            "Retail": ["retail", "ecommerce", "consumer goods", "fmcg"],
            "Telecommunications": [
                "telecom", "telecommunications", "mobile", "network"
            ],
            "Automotive": ["automotive", "automobile", "vehicle", "oem"],
            "Manufacturing": [
                "manufacturing", "industrial", "production", "factory"
            ],
            "Insurance": ["insurance", "actuarial", "underwriting", "claims"],
            "Real Estate": ["real estate", "property", "construction", "development"],
            "Government": ["government", "public sector", "federal", "municipal"],
            "Education": ["education", "university", "school", "academic"],
            "Media & Entertainment": [
                "media", "entertainment", "broadcasting", "streaming"
            ]
        }

        client_industries = set()

        # Scan projects
        for project in projects:
            if isinstance(project, dict):
                text = (
                    f"{project.get('name', '')} {project.get('description', '')}"
                ).lower()
            else:
                text = str(project).lower()

            for industry, keywords in industry_keywords.items():
                if any(keyword in text for keyword in keywords):
                    client_industries.add(industry)

        # Scan responsibilities
        for resp in responsibilities:
            text = str(resp).lower()
            for industry, keywords in industry_keywords.items():
                if any(keyword in text for keyword in keywords):
                    client_industries.add(industry)

        return sorted(list(client_industries))

    def _enhance_skill_extraction(
        self,
        existing_skills: List[str],
        responsibilities: List[Any],
        projects: List[Any]
    ) -> List[str]:
        """
        Enhanced skill extraction from context (REQ-SEM-04).
        """
        # Technical skill patterns
        skill_patterns = {
            "Programming Languages": (
                r'\b(Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Swift|'
                r'Kotlin|TypeScript|R|MATLAB|Scala|PHP|Perl|Dart)\b'
            ),
            "Frameworks": (
                r'\b(React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|'
                r'FastAPI|Rails|Laravel|\.NET|ASP\.NET)\b'
            ),
            "Databases": (
                r'\b(SQL|MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|'
                r'DynamoDB|Elasticsearch|MariaDB|SQLite)\b'
            ),
            "Cloud": (
                r'\b(AWS|Azure|GCP|Google Cloud|Kubernetes|Docker|Terraform|'
                r'CloudFormation|Ansible|Jenkins)\b'
            ),
            "Tools": (
                r'\b(Git|GitHub|GitLab|CircleCI|Jira|Confluence|Tableau|'
                r'Power BI|Excel|SAP|Salesforce)\b'
            ),
            "ML/AI": (
                r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|'
                r'TensorFlow|PyTorch|Scikit-learn|Keras|OpenCV)\b'
            )
        }

        skills = set(existing_skills)

        # Combine all text
        all_text = ' '.join(
            [str(r) for r in responsibilities] +
            [str(p) for p in projects]
        )

        # Extract skills using patterns
        for category, pattern in skill_patterns.items():
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            skills.update(matches)

        return sorted(list(skills))

    def _extract_context_window(self, text: str, anchor: str, window_size: int) -> str:
        """Extract text window around anchor text."""
        try:
            pos = text.lower().find(anchor.lower())
            if pos == -1:
                return ""

            start = max(0, pos - window_size)
            end = min(len(text), pos + len(anchor) + window_size)

            return text[start:end]
        except Exception:
            return ""

    def _convert_to_cv_data(self, data: Dict[str, Any]) -> CVData:
        """Convert dictionary to CVData object."""

        # Personal info
        personal_info = PersonalInfo(
            **data.get("personal_info", {"full_name": "N/A"})
        )

        # Work experience
        work_experience = []
        for exp_data in data.get("work_experience", []):
            # Convert projects
            projects = []
            for proj_data in exp_data.get("projects", []):
                if isinstance(proj_data, dict):
                    projects.append(Project(
                        name=proj_data.get("name", ""),
                        description=proj_data.get("description", ""),
                        role=proj_data.get("role", ""),
                        technologies=proj_data.get("technologies", []),
                        outcomes=proj_data.get("outcomes", []),
                        client_industry=proj_data.get("client_industry")
                    ))

            exp = WorkExperience(
                company_name=exp_data.get("company_name", ""),
                job_title=exp_data.get("job_title", ""),
                start_date=exp_data.get("start_date", ""),
                end_date=exp_data.get("end_date"),
                is_current=exp_data.get("is_current", False),
                employer_industry=exp_data.get("employer_industry", ""),
                client_industries=exp_data.get("client_industries", []),
                responsibilities=exp_data.get("responsibilities", []),
                projects=projects,
                skills_used=exp_data.get("skills_used", []),
                temporal_confidence=exp_data.get("temporal_confidence", 1.0),
                date_status=DateStatus(exp_data.get("date_status", "unknown")),
                duration_months=exp_data.get("duration_months")
            )
            work_experience.append(exp)

        # Education
        education = [Education(**edu) for edu in data.get("education", [])]

        # Skills
        skills_data = data.get("skills", {})
        skills = Skills(
            technical=skills_data.get("technical", []),
            soft=skills_data.get("soft", []),
            languages=skills_data.get("languages", []),
            certifications=skills_data.get("certifications", []),
            tools=skills_data.get("tools", [])
        )

        return CVData(
            personal_info=personal_info,
            work_experience=work_experience,
            education=education,
            skills=skills,
            extraction_confidence=ConfidenceLevel.HIGH
        )

    def _create_fallback_structure(self, cv_text: str) -> Dict[str, Any]:
        """
        Deterministic fallback with "N/A" values (REQ-DET-03).
        """
        # Attempt basic regex extraction as fallback
        name = self._extract_name_fallback(cv_text)
        email = self._extract_email_fallback(cv_text)

        return {
            "personal_info": {
                "full_name": name or "N/A",
                "email": email,
                "phone": None,
                "location": None,
                "linkedin": None
            },
            "work_experience": [],
            "education": [],
            "skills": {
                "technical": [],
                "soft": [],
                "languages": [],
                "certifications": [],
                "tools": []
            }
        }

    def _extract_name_fallback(self, text: str) -> Optional[str]:
        """Fallback name extraction using regex."""
        # Look for name at the beginning of CV
        lines = text.strip().split('\n')
        for line in lines[:5]:
            # Pattern for name: 2-3 capitalized words
            match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', line)
            if match:
                return match.group(1)
        return None

    def _extract_email_fallback(self, text: str) -> Optional[str]:
        """Fallback email extraction using regex."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else None
