"""Agent 1: Input validation and file processing."""
import os
from pathlib import Path
from typing import Dict, Any
from docx import Document
import PyPDF2

from .base_agent import BaseAgent
from ..core.exceptions import ValidationError, UnsupportedFormatError, FileIOError


class Agent1_InputValidator(BaseAgent):
    """
    Validates input CV files and extracts text.
    Implements REQ-A1-01: Multi-format support
    """

    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt'}
    MAX_FILE_SIZE_MB = 10

    def __init__(self):
        super().__init__("agent1_input_validator", temperature=0.0)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate input CV file.

        Args:
            input_data: {
                "file_path": str,
                "format": str (optional)
            }

        Returns:
            {
                "text": str,
                "format": str,
                "file_size_mb": float,
                "file_path": str,
                "validation_status": str,
                "text_length": int
            }

        Raises:
            ValidationError: If validation fails
            UnsupportedFormatError: If file format not supported
            FileIOError: If file cannot be read
        """
        file_path = input_data.get("file_path")
        if not file_path:
            raise ValidationError("file_path is required")

        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileIOError(f"File not found: {file_path}")

        # Validate format
        file_format = file_path.suffix.lower()
        if file_format not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format: {file_format}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Validate file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum: {self.MAX_FILE_SIZE_MB}MB"
            )

        # Extract text
        text = self._extract_text(file_path, file_format)

        if not text or len(text.strip()) < 50:
            raise ValidationError("Extracted text is too short or empty (minimum 50 characters)")

        self.logger.info(
            f"Successfully validated and extracted {len(text)} characters from {file_path.name}"
        )

        return {
            "text": text,
            "format": file_format,
            "file_size_mb": round(file_size_mb, 2),
            "file_path": str(file_path),
            "validation_status": "valid",
            "text_length": len(text)
        }

    def _extract_text(self, file_path: Path, file_format: str) -> str:
        """Extract text from file based on format."""
        if file_format == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_format == '.docx':
            return self._extract_from_docx(file_path)
        elif file_format == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise UnsupportedFormatError(f"Unsupported format: {file_format}")

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF."""
        text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return '\n'.join(text)
        except Exception as e:
            raise FileIOError(f"Failed to extract from PDF: {str(e)}")

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        try:
            doc = Document(file_path)
            text = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(text)
        except Exception as e:
            raise FileIOError(f"Failed to extract from DOCX: {str(e)}")

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                raise FileIOError(f"Failed to extract from TXT: {str(e)}")
        except Exception as e:
            raise FileIOError(f"Failed to extract from TXT: {str(e)}")
