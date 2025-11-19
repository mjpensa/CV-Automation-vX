"""Configuration management for the CV automation system."""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv


class SystemConfig:
    """Centralized configuration manager."""

    def __init__(self, config_path: str = "config/system_config.yaml"):
        load_dotenv()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        self.validate()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def validate(self):
        """Validate configuration."""
        # API key check can be deferred for setup, but warn
        if not self.api_key:
            import warnings
            warnings.warn(
                "GEMINI_API_KEY not found in environment. "
                "Set it in .env file before running agents."
            )

        # Verify determinism settings
        if self.temperature != 0.0:
            raise ValueError(
                f"DETERMINISM VIOLATION: Temperature must be 0.0, got {self.temperature}"
            )

        required_sections = ['agents', 'pipeline', 'validation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for specific agent."""
        return self.config['agents'].get(agent_name, {})

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config['pipeline']

    def get_determinism_config(self) -> Dict[str, Any]:
        """Get determinism verification configuration."""
        return self.config.get('determinism', {
            'verify_on_completion': True,
            'log_all_transformations': True,
            'hash_outputs': True
        })


# Global config instance
config = SystemConfig()
