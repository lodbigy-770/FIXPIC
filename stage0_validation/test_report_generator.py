# test_report_generator.py
"""
Quick test for report_generator.py functions.
"""

from stage0_validation.data_structures import ValidationReport
from stage0_validation.report_generator import log_step, generate_summary, format_error


def test_log_step():
    """Test log_step function."""
    print("\n=== Testing log_step ===")
    log_step("file_validation", "success", "SHA256 match")
    log_step("model_loading", "failed")
    print("✓ log_step test passed")


def test_generate_summary():
    """Test generate_summary function."""
    print("\n=== Testing generate_summary ===")
    
    # Create a sample report
    report = ValidationReport()
    report.add_step("File Validation", "success", "SHA256 verified")
    report.add_step("Model Loading", "success", "ONNX session created")
    report.add_step("Metadata Extraction", "success", "2 inputs, 1 output")
    
    summary = generate_summary(report)
    print(summary)
    
    # Verify summary contains expected elements
    assert "✓" in summary
    assert "SUCCESS" in summary
    assert "File Validation" in summary
    print("\n✓ generate_summary test passed")


def test_generate_summary_with_failure():
    """Test generate_summary with a failed step."""
    print("\n=== Testing generate_summary with failure ===")
    
    report = ValidationReport()
    report.add_step("File Validation", "success", "SHA256 verified")
    report.add_step("Model Loading", "failed", "File not found")
    
    summary = generate_summary(report)
    print(summary)
    
    # Verify summary shows failure
    assert "✗" in summary
    assert "FAILED" in summary
    assert "File not found" in summary
    print("\n✓ generate_summary with failure test passed")


def test_format_error():
    """Test format_error function."""
    print("\n=== Testing format_error ===")
    
    # Create a sample exception
    try:
        raise FileNotFoundError("inpainter.onnx not found")
    except FileNotFoundError as e:
        error_msg = format_error(e, "Model Loading")
        print(error_msg)
        
        # Verify error message contains expected elements
        assert "FileNotFoundError" in error_msg
        assert "inpainter.onnx not found" in error_msg
        assert "Model Loading" in error_msg
        print("\n✓ format_error test passed")


if __name__ == "__main__":
    test_log_step()
    test_generate_summary()
    test_generate_summary_with_failure()
    test_format_error()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
