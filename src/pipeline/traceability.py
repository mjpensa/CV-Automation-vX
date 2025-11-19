"""Traceability matrix generation and export."""
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd


class TraceabilityManager:
    """
    Manages complete traceability matrix.
    Implements REQ-EXP-01, REQ-EXP-02, REQ-EXP-03
    """

    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def add_stage(self, stage_name: str, result: Dict[str, Any]):
        """Add a pipeline stage result."""
        stage_entry = {
            "stage_name": stage_name,
            "timestamp": datetime.now().isoformat(),
            "status": result.get("status", "unknown"),
            "result": result.get("result", {}),
            "metadata": result.get("metadata", {}),
            "error": result.get("error") if result.get("status") == "failed" else None
        }
        self.stages.append(stage_entry)

    def generate_matrix(self) -> List[Dict[str, Any]]:
        """Generate complete traceability matrix."""
        matrix = []

        for idx, stage in enumerate(self.stages):
            entry = {
                "stage_number": idx + 1,
                "stage_name": stage["stage_name"],
                "timestamp": stage["timestamp"],
                "status": stage["status"],
                "duration_seconds": self._calculate_duration(idx),
                "inputs": self._extract_inputs(stage),
                "outputs": self._extract_outputs(stage),
                "metadata": stage["metadata"]
            }
            matrix.append(entry)

        return matrix

    def _calculate_duration(self, stage_idx: int) -> float:
        """Calculate stage duration."""
        if stage_idx == 0:
            start = self.start_time
        else:
            start = datetime.fromisoformat(self.stages[stage_idx - 1]["timestamp"])

        end = datetime.fromisoformat(self.stages[stage_idx]["timestamp"])

        return (end - start).total_seconds()

    def _extract_inputs(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input summary from stage."""
        # Simplified - extract key inputs
        return {
            "status": stage["status"]
        }

    def _extract_outputs(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Extract output summary from stage."""
        result = stage.get("result", {})

        # Extract key metrics
        outputs: Dict[str, Any] = {}

        if "cv_data" in result:
            outputs["cv_extracted"] = True

        if "mappings" in result:
            outputs["mappings_count"] = len(result["mappings"])

        if "transformed_content" in result:
            outputs["transformations_count"] = len(result["transformed_content"])

        return outputs

    def export_json(self, output_path: str):
        """Export traceability to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "pipeline_start": self.start_time.isoformat(),
            "pipeline_end": datetime.now().isoformat(),
            "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "stages": self.stages,
            "matrix": self.generate_matrix()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_excel(self, output_path: str):
        """Export traceability to Excel."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        matrix = self.generate_matrix()

        # Convert to DataFrame
        df = pd.DataFrame(matrix)

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Traceability', index=False)

            # Add summary sheet
            summary = {
                "Metric": [
                    "Pipeline Start",
                    "Pipeline End",
                    "Total Duration (seconds)",
                    "Total Stages",
                    "Successful Stages",
                    "Failed Stages"
                ],
                "Value": [
                    self.start_time.isoformat(),
                    datetime.now().isoformat(),
                    (datetime.now() - self.start_time).total_seconds(),
                    len(self.stages),
                    sum(1 for s in self.stages if s["status"] == "success"),
                    sum(1 for s in self.stages if s["status"] == "failed")
                ]
            }

            summary_df = pd.DataFrame(summary)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
