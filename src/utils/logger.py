"""Centralized logging system with traceability."""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemLogger:
    """Enhanced logger with structured output and traceability."""

    def __init__(self, name: str, log_file: str = "outputs/system.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.trace_log: List[Dict[str, Any]] = []
        self.transformation_log: List[Dict[str, Any]] = []

    def log_agent_execution(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        metadata: Dict[str, Any]
    ):
        """Log agent execution with full traceability."""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "input_hash": hash(str(input_data)),
            "output_hash": hash(str(output_data)),
            "metadata": metadata,
            "status": metadata.get('status', 'unknown')
        }
        self.trace_log.append(trace_entry)

        self.logger.info(
            f"Agent {agent_name} executed: {metadata.get('status', 'unknown')}"
        )

    def log_transformation(
        self,
        transformation_type: str,
        original: str,
        transformed: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log content transformation for determinism verification.
        CRITICAL for REQ-EXP-01: Before/after logging.
        """
        transformation_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": transformation_type,
            "original": original,
            "transformed": transformed,
            "original_length": len(original),
            "transformed_length": len(transformed),
            "metadata": metadata or {}
        }
        self.transformation_log.append(transformation_entry)

        self.logger.debug(
            f"Transformation [{transformation_type}]: "
            f"{len(original)} → {len(transformed)} chars"
        )

    def export_trace_log(self, output_path: str):
        """Export complete trace log."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.trace_log, f, indent=2)
        self.logger.info(f"Trace log exported to {output_path}")

    def export_transformation_log(self, output_path: str):
        """Export complete transformation log."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.transformation_log, f, indent=2)
        self.logger.info(f"Transformation log exported to {output_path}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance
logger = SystemLogger("CVAutomation")
