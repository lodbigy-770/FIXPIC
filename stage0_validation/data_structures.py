# data_structures.py
"""
Data structures for Stage 0 Model Validation.
Defines TensorInfo and ValidationReport classes.
"""

from dataclasses import dataclass, field
from typing import List, Union, Tuple, Any


@dataclass
class TensorInfo:
    """
    Represents metadata for a tensor (input or output).
    
    Attributes:
        name: Name of the tensor
        shape: Shape of the tensor, with dynamic dimensions as strings
        dtype: Data type of the tensor (e.g., "tensor(float)")
    """
    name: str
    shape: List[Union[int, str]]
    dtype: str


@dataclass
class ValidationReport:
    """
    Tracks the validation process with step-by-step results.
    
    Attributes:
        steps: List of (step_name, status, details) tuples
    """
    steps: List[Tuple[str, str, Any]] = field(default_factory=list)
    
    def add_step(self, name: str, status: str, details: Any = None) -> None:
        """
        Add a validation step to the report.
        
        Args:
            name: Name of the validation step
            status: Status of the step ("success" or "failed")
            details: Optional details about the step
        """
        self.steps.append((name, status, details))
    
    def is_success(self) -> bool:
        """
        Check if all validation steps succeeded.
        
        Returns:
            True if all steps have "success" status, False otherwise
        """
        return all(status == "success" for _, status, _ in self.steps)
