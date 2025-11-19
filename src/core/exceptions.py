"""Custom exception hierarchy for the CV automation system."""


class CVAutomationError(Exception):
    """Base exception for all CV automation errors."""
    pass


class ConfigurationError(CVAutomationError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(CVAutomationError):
    """Raised when data validation fails."""
    pass


class SchemaViolationError(ValidationError):
    """Raised when data violates JSON schema."""
    pass


class TemporalConsistencyError(ValidationError):
    """Raised when temporal data is inconsistent (REQ-SEM-01)."""
    pass


class ExtractionError(CVAutomationError):
    """Raised when CV extraction fails."""
    pass


class ParsingError(ExtractionError):
    """Raised when file parsing fails."""
    pass


class APIError(CVAutomationError):
    """Raised when external API calls fail."""
    pass


class GeminiAPIError(APIError):
    """Raised when Gemini API calls fail."""
    pass


class TransformationError(CVAutomationError):
    """Raised when content transformation fails."""
    pass


class CharacterLimitError(TransformationError):
    """Raised when character limit cannot be met (REQ-A6-01)."""
    pass


class DeterminismError(CVAutomationError):
    """Raised when deterministic processing fails."""
    pass


class TemplateError(CVAutomationError):
    """Raised when PowerPoint template processing fails."""
    pass


class PlaceholderError(TemplateError):
    """Raised when placeholder operations fail."""
    pass


class MappingError(CVAutomationError):
    """Raised when CV-to-template mapping fails."""
    pass


class PipelineError(CVAutomationError):
    """Raised when pipeline orchestration fails."""
    pass


class FileIOError(CVAutomationError):
    """Raised when file I/O operations fail."""
    pass


class UnsupportedFormatError(FileIOError):
    """Raised when file format is not supported."""
    pass
