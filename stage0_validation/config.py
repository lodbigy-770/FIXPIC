# config.py
"""
Configuration constants for Stage 0 Model Validation.
Contains file paths, expected values, and other settings.
"""

from pathlib import Path

# Paths
WORKSPACE_ROOT = Path(__file__).parent.parent
MODEL_PATH = WORKSPACE_ROOT / "models" / "big-lama" / "inpainter.onnx"
MODEL_CARD_PATH = WORKSPACE_ROOT / "docs" / "architecture" / "cleanup-ai" / "MODEL_CARD.md"

# Expected values
EXPECTED_SHA256 = "ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff"
EXPECTED_FILE_SIZE = 208851664  # bytes

# Settings
LFS_MARKER = b"version https://git-lfs.github.com"
LFS_CHECK_BYTES = 200
SHA256_BLOCK_SIZE = 8192

# Status markers
MARKER_VERIFIED = "[אומת]"
MARKER_POINTER = "[pointer]"
MARKER_TO_VERIFY = "[README – לאמת ב-`probe`]"
MARKER_README = "[README]"
MARKER_UNKNOWN = "[לא ידוע]"
MARKER_GENERAL_KNOWLEDGE = "[ידע כללי]"
