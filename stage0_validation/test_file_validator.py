# test_file_validator.py
"""
Unit tests for file_validator module.
Tests SHA256 computation, file size checking, and LFS pointer detection.
"""

import hashlib
import tempfile
from pathlib import Path
import pytest
from .file_validator import compute_sha256, check_file_size, check_lfs_pointer
from .config import EXPECTED_SHA256, EXPECTED_FILE_SIZE, MODEL_PATH, LFS_MARKER


def test_compute_sha256_with_known_content():
    """Test SHA256 computation with known content."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_content = b"Hello, World!"
        f.write(test_content)
        temp_path = Path(f.name)
    
    try:
        # Compute expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        # Test our function
        result = compute_sha256(temp_path)
        
        assert result == expected_hash
        assert len(result) == 64  # SHA256 produces 64-character hex string
    finally:
        temp_path.unlink()


def test_compute_sha256_empty_file():
    """Test SHA256 computation with empty file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        result = compute_sha256(temp_path)
        # SHA256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected
    finally:
        temp_path.unlink()


def test_compute_sha256_large_file():
    """Test SHA256 computation with file larger than block size."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        # Write 20KB of data (more than the 8KB block size)
        test_content = b"A" * (20 * 1024)
        f.write(test_content)
        temp_path = Path(f.name)
    
    try:
        expected_hash = hashlib.sha256(test_content).hexdigest()
        result = compute_sha256(temp_path)
        assert result == expected_hash
    finally:
        temp_path.unlink()


def test_compute_sha256_nonexistent_file():
    """Test SHA256 computation with nonexistent file."""
    nonexistent_path = Path("/nonexistent/file/path.bin")
    
    with pytest.raises(FileNotFoundError):
        compute_sha256(nonexistent_path)


def test_compute_sha256_actual_model():
    """Test SHA256 computation on the actual model file (if it exists)."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    
    result = compute_sha256(MODEL_PATH)
    
    # Verify format
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)
    
    # Check against expected hash
    assert result == EXPECTED_SHA256, (
        f"Model hash mismatch!\n"
        f"Expected: {EXPECTED_SHA256}\n"
        f"Got:      {result}"
    )


def test_compute_sha256_actual_model():
    """Test SHA256 computation on the actual model file (if it exists)."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    
    result = compute_sha256(MODEL_PATH)
    
    # Verify format
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)
    
    # Check against expected hash
    assert result == EXPECTED_SHA256, (
        f"Model hash mismatch!\n"
        f"Expected: {EXPECTED_SHA256}\n"
        f"Got:      {result}"
    )


# ---------------------------------------------------------------------------
# check_file_size() tests (Task 2.2)
# ---------------------------------------------------------------------------

def test_check_file_size_matching_size():
    """Returns True when actual file size matches expected."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b"x" * 100)
        temp_path = Path(f.name)

    try:
        assert check_file_size(temp_path, expected=100) is True
    finally:
        temp_path.unlink()


def test_check_file_size_mismatched_size():
    """Returns False when actual file size does not match expected."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b"x" * 100)
        temp_path = Path(f.name)

    try:
        assert check_file_size(temp_path, expected=999) is False
    finally:
        temp_path.unlink()


def test_check_file_size_empty_file():
    """Returns True for an empty file when expected is 0."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = Path(f.name)

    try:
        assert check_file_size(temp_path, expected=0) is True
        assert check_file_size(temp_path, expected=1) is False
    finally:
        temp_path.unlink()


def test_check_file_size_nonexistent_file():
    """Raises FileNotFoundError for a missing file."""
    nonexistent = Path("/nonexistent/path/file.bin")
    with pytest.raises((FileNotFoundError, OSError)):
        check_file_size(nonexistent, expected=0)


def test_check_file_size_actual_model():
    """Check the actual model file size against the expected constant."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found at {MODEL_PATH}")

    assert check_file_size(MODEL_PATH) is True, (
        f"Model size mismatch!\n"
        f"Expected: {EXPECTED_FILE_SIZE} bytes\n"
        f"Got:      {MODEL_PATH.stat().st_size} bytes"
    )


# ---------------------------------------------------------------------------
# check_lfs_pointer() tests (Task 2.3)
# ---------------------------------------------------------------------------

def test_check_lfs_pointer_regular_binary():
    """Returns False for a regular binary file (not an LFS pointer)."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        # Write random binary content (not an LFS pointer)
        f.write(bytes(range(256)) * 10)
        temp_path = Path(f.name)

    try:
        assert check_lfs_pointer(temp_path) is False
    finally:
        temp_path.unlink()


def test_check_lfs_pointer_lfs_file():
    """Returns True when the file starts with the Git LFS marker."""
    lfs_content = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:abc123\n"
        b"size 12345\n"
    )
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(lfs_content)
        temp_path = Path(f.name)

    try:
        assert check_lfs_pointer(temp_path) is True
    finally:
        temp_path.unlink()


def test_check_lfs_pointer_marker_beyond_200_bytes():
    """Returns False when the LFS marker appears after the first 200 bytes."""
    # 200 bytes of zeros, then the LFS marker
    content = b"\x00" * 200 + LFS_MARKER
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        assert check_lfs_pointer(temp_path) is False
    finally:
        temp_path.unlink()


def test_check_lfs_pointer_empty_file():
    """Returns False for an empty file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = Path(f.name)

    try:
        assert check_lfs_pointer(temp_path) is False
    finally:
        temp_path.unlink()


def test_check_lfs_pointer_actual_model():
    """The actual model file must NOT be an LFS pointer."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found at {MODEL_PATH}")

    result = check_lfs_pointer(MODEL_PATH)
    assert result is False, (
        "The model file appears to be a Git LFS pointer. "
        "Run 'git lfs pull' to download the actual model."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
