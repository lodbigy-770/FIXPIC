#!/usr/bin/env python3
"""
validate_stage0.py — Main entry point for Stage 0 Model Validation.

Runs the full validation workflow:
  1. File integrity checks (SHA256, size, LFS pointer)
  2. ONNX model loading and metadata extraction
  3. MODEL_CARD.md update with verified data
  4. Summary report to stdout

Usage:
    python validate_stage0.py

Exit codes:
    0  — all steps succeeded
    1  — SHA256 or file size check failed
    2  — LFS pointer check failed
    3  — session loading failed
    4  — input/output metadata extraction failed
    5  — tensor role identification failed
    6  — MODEL_CARD write failed

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import sys


def main() -> None:
    """Run Stage 0 validation and exit with an appropriate code."""
    # Handle optional --help flag without importing heavy dependencies
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    from stage0_validation.validation_orchestrator import validate_model, get_exit_code

    report = validate_model()
    exit_code = get_exit_code(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
