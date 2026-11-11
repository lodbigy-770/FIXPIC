# file_validator.py
"""
File validation functions for Stage 0 Model Validation.
Provides SHA256 hashing, file size checking, and LFS pointer detection.
"""

import hashlib
from pathlib import Path
from .config import (
    SHA256_BLOCK_SIZE,
    EXPECTED_SHA256,
    EXPECTED_FILE_SIZE,
    LFS_MARKER,
    LFS_CHECK_BYTES,
)


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file in blocks to save memory.
    
    Reads the file in 8KB chunks to avoid loading large files (200MB+)
    entirely into memory.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Hexadecimal string representation of the SHA256 hash
        
    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read
    """
    sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(SHA256_BLOCK_SIZE):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def check_file_size(file_path: Path, expected: int = EXPECTED_FILE_SIZE) -> bool:
    """
    Check whether a file's size matches the expected size.

    Uses ``Path.stat().st_size`` to read the on-disk size without loading
    the file into memory.

    Args:
        file_path: Path to the file to check.
        expected:  Expected size in bytes (defaults to EXPECTED_FILE_SIZE =
                   208,851,664 bytes).

    Returns:
        True if the actual size equals *expected*, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError:   If the file metadata cannot be read.
    """
    actual_size = file_path.stat().st_size
    return actual_size == expected


def check_lfs_pointer(file_path: Path) -> bool:
    """
    Check if a file is a Git LFS pointer instead of actual binary content.
    
    Reads the first 200 bytes of the file and checks for the LFS marker
    string "version https://git-lfs.github.com". If found, the file is
    an LFS pointer and the actual model file needs to be downloaded with
    'git lfs pull'.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is an LFS pointer (error condition)
        False if the file is a regular binary file (expected)
        
    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read
    """
    with open(file_path, 'rb') as f:
        header = f.read(LFS_CHECK_BYTES)
    
    return LFS_MARKER in header
