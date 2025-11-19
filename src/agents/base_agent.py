"""Base agent class for deterministic processing."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import json

from ..utils.logger import logger
from ..core.exceptions import DeterminismError, CVAutomationError


class BaseAgent(ABC):
    """
    Base class for all agents with deterministic processing enforcement.
    Implements REQ-DET-01: Temperature=0.0 mandate
    """

    def __init__(self, name: str, temperature: float = 0.0):
        """
        Initialize base agent.

        Args:
            name: Agent name
            temperature: Temperature for AI models (must be 0.0 for determinism)

        Raises:
            DeterminismError: If temperature is not 0.0
        """
        if temperature != 0.0:
            raise DeterminismError(
                f"DETERMINISM VIOLATION: Agent {name} temperature must be 0.0, got {temperature}"
            )

        self.name = name
        self.temperature = temperature
        self.execution_history = []
        self.logger = logger

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return output.
        Must be implemented by all agents.

        Args:
            input_data: Input data dictionary

        Returns:
            Output data dictionary
        """
        pass

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with full traceability.

        Args:
            input_data: Input data dictionary

        Returns:
            Output data dictionary with metadata

        Raises:
            CVAutomationError: If execution fails
        """
        execution_start = datetime.now()

        try:
            # Generate input hash for determinism verification
            input_hash = self._hash_data(input_data)

            self.logger.info(f"Agent {self.name} starting execution")

            # Execute the main processing logic
            output_data = self.process(input_data)

            # Generate output hash
            output_hash = self._hash_data(output_data)

            # Calculate execution time
            execution_time = (datetime.now() - execution_start).total_seconds()

            # Record execution
            execution_record = {
                "timestamp": execution_start.isoformat(),
                "agent": self.name,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "execution_time_seconds": execution_time,
                "status": "success"
            }
            self.execution_history.append(execution_record)

            # Log execution
            self.logger.log_agent_execution(
                agent_name=self.name,
                input_data=input_data,
                output_data=output_data,
                metadata=execution_record
            )

            self.logger.info(
                f"Agent {self.name} completed successfully in {execution_time:.2f}s"
            )

            return output_data

        except Exception as e:
            execution_time = (datetime.now() - execution_start).total_seconds()

            # Record failure
            execution_record = {
                "timestamp": execution_start.isoformat(),
                "agent": self.name,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_seconds": execution_time
            }
            self.execution_history.append(execution_record)

            self.logger.error(
                f"Agent {self.name} failed after {execution_time:.2f}s: {str(e)}"
            )

            raise

    def _hash_data(self, data: Any) -> str:
        """
        Generate deterministic hash of data for comparison.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash string
        """
        try:
            # Convert to stable JSON representation
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(data).encode()).hexdigest()

    def verify_determinism(
        self,
        input_data: Dict[str, Any],
        runs: int = 2
    ) -> Dict[str, Any]:
        """
        Verify deterministic output by running multiple times.
        CRITICAL for determinism validation.

        Args:
            input_data: Input data to test with
            runs: Number of runs to compare (default 2)

        Returns:
            Dictionary with verification results

        Raises:
            DeterminismError: If outputs are not identical
        """
        self.logger.info(
            f"Starting determinism verification for {self.name} with {runs} runs"
        )

        outputs = []
        hashes = []

        for i in range(runs):
            output = self.process(input_data)
            output_hash = self._hash_data(output)

            outputs.append(output)
            hashes.append(output_hash)

            self.logger.debug(f"Run {i+1} hash: {output_hash}")

        # Check if all hashes are identical
        unique_hashes = set(hashes)

        if len(unique_hashes) == 1:
            self.logger.info(
                f"✓ DETERMINISM VERIFIED: {self.name} produced identical output across {runs} runs"
            )
            return {
                "deterministic": True,
                "runs": runs,
                "hash": hashes[0],
                "message": "All outputs identical"
            }
        else:
            self.logger.error(
                f"✗ DETERMINISM VIOLATION: {self.name} produced {len(unique_hashes)} different outputs"
            )
            raise DeterminismError(
                f"Agent {self.name} failed determinism check: "
                f"produced {len(unique_hashes)} different outputs across {runs} runs. "
                f"Hashes: {unique_hashes}"
            )

    def get_execution_history(self) -> list:
        """Get agent execution history."""
        return self.execution_history

    def reset_history(self):
        """Reset execution history."""
        self.execution_history = []
        self.logger.info(f"Agent {self.name} history reset")
