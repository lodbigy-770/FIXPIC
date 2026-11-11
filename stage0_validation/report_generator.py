# report_generator.py
"""
Report generation and logging utilities for Stage 0 Model Validation.
"""

from typing import Any
from .data_structures import ValidationReport


def log_step(step_name: str, status: str, details: Any = None) -> None:
    """
    Log the progress of a validation step to console.
    
    Args:
        step_name: Name of the validation step
        status: Status of the step (e.g., "success", "failed")
        details: Optional details about the step
    
    Format: [STEP_NAME] status
    """
    print(f"[{step_name.upper()}] {status}")
    if details:
        print(f"  Details: {details}")


def generate_summary(report: ValidationReport) -> str:
    """
    Create a detailed summary of the entire validation process.
    
    Args:
        report: ValidationReport containing all validation steps
    
    Returns:
        Formatted summary string with ✓/✗ symbols for each step
    """
    lines = ["=" * 60, "Stage 0 Validation Summary", "=" * 60, ""]
    
    for name, status, details in report.steps:
        symbol = "✓" if status == "success" else "✗"
        lines.append(f"{symbol} {name}: {status}")
        if details:
            lines.append(f"  → {details}")
        lines.append("")
    
    overall = "SUCCESS" if report.is_success() else "FAILED"
    lines.append(f"Overall: {overall}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_error(exception: Exception, context: str) -> str:
    """
    Format an error message with type, message, and context.
    
    Args:
        exception: The exception that was raised
        context: Context string describing where the error occurred
    
    Returns:
        Formatted error message string
    """
    error_type = type(exception).__name__
    error_message = str(exception)
    
    lines = [
        f"Error in {context}:",
        f"  Type: {error_type}",
        f"  Message: {error_message}"
    ]
    
    return "\n".join(lines)
