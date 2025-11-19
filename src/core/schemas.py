"""Strict data schemas for the CV automation system."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json


class DateStatus(Enum):
    """Temporal status of work experience."""
    CURRENT = "current"
    PAST = "past"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted data."""
    HIGH = "high"          # > 0.8
    MEDIUM = "medium"      # 0.5 - 0.8
    LOW = "low"            # < 0.5


@dataclass
class PersonalInfo:
    """Personal information schema."""
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Project:
    """Project within work experience."""
    name: str
    description: str
    role: str
    technologies: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    client_industry: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WorkExperience:
    """Work experience with semantic metadata."""
    company_name: str
    job_title: str
    start_date: str  # YYYY-MM format
    end_date: Optional[str]  # YYYY-MM format or None
    is_current: bool
    employer_industry: str
    client_industries: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)

    # Semantic metadata (REQ-SEM-01)
    temporal_confidence: float = 1.0
    date_status: DateStatus = DateStatus.UNKNOWN
    duration_months: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['date_status'] = self.date_status.value
        # Convert Project objects to dicts
        data['projects'] = [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.projects]
        return data

    def calculate_duration(self) -> int:
        """Calculate duration in months."""
        try:
            start = datetime.strptime(self.start_date, "%Y-%m")
            if self.is_current:
                end = datetime.now()
            else:
                end = datetime.strptime(self.end_date, "%Y-%m")

            self.duration_months = (end.year - start.year) * 12 + (end.month - start.month)
            return self.duration_months
        except (ValueError, TypeError):
            return 0


@dataclass
class Education:
    """Education schema."""
    institution: str
    degree: str
    field: str
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    honors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Skills:
    """Skills categorization."""
    technical: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CVData:
    """Complete CV data structure with strict typing."""
    personal_info: PersonalInfo
    work_experience: List[WorkExperience]
    education: List[Education]
    skills: Skills

    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "personal_info": self.personal_info.to_dict(),
            "work_experience": [exp.to_dict() for exp in self.work_experience],
            "education": [edu.to_dict() for edu in self.education],
            "skills": self.skills.to_dict(),
            "metadata": {
                "extraction_timestamp": self.extraction_timestamp,
                "extraction_confidence": self.extraction_confidence.value,
                "source_file": self.source_file
            }
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PlaceholderMapping:
    """Mapping between CV data and PowerPoint placeholder."""
    placeholder_id: str
    placeholder_type: str  # text, image, table
    source_field: str      # Path to CV data field
    transformation_rules: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TransformationResult:
    """Result of content transformation."""
    placeholder_id: str
    original_text: str
    transformed_text: str
    rules_applied: Dict[str, Any]
    compliance_status: str  # "full", "partial", "failed"
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# JSON Schema Definitions for validation
CV_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "personal_info": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string", "minLength": 1},
                "email": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]},
                "location": {"type": ["string", "null"]},
                "linkedin": {"type": ["string", "null"]}
            },
            "required": ["full_name"]
        },
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "minLength": 1},
                    "job_title": {"type": "string", "minLength": 1},
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-(0[1-9]|1[0-2])$"
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "pattern": "^\\d{4}-(0[1-9]|1[0-2])$|^$"
                    },
                    "is_current": {"type": "boolean"},
                    "employer_industry": {"type": "string"},
                    "client_industries": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "responsibilities": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "projects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "role": {"type": "string"},
                                "technologies": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "outcomes": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "client_industry": {"type": ["string", "null"]}
                            },
                            "required": ["name", "description", "role"]
                        }
                    },
                    "skills_used": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "company_name", "job_title", "start_date",
                    "is_current", "employer_industry"
                ]
            }
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string", "minLength": 1},
                    "degree": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                    "graduation_date": {"type": ["string", "null"]},
                    "gpa": {"type": ["number", "null"]},
                    "honors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["institution", "degree", "field"]
            }
        },
        "skills": {
            "type": "object",
            "properties": {
                "technical": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "soft": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "certifications": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": ["personal_info", "work_experience", "education", "skills"]
}
