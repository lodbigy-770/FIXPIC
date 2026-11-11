# test_model_inspector.py
"""
Unit tests for ModelInspector – load_session() function (task 4.1).

Covers Requirements 3.1, 3.2, 3.4:
  3.1 – create an ONNX InferenceSession from the model path
  3.2 – use CPU ExecutionProvider only
  3.4 – if session creation fails, report an error with exception details
"""

import struct
import pytest
from pathlib import Path

import onnxruntime as ort

from stage0_validation.model_inspector import load_session
from stage0_validation.config import MODEL_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_onnx_bytes() -> bytes:
    """
    Build the smallest possible valid ONNX model in memory.

    The model has one float input ``x`` with shape [1] and one float
    output ``y``, connected by a single Identity node.  The serialised
    proto is enough for onnxruntime to accept the model without loading a
    large file.
    """
    # Build a minimal ONNX ModelProto using protobuf-compatible raw bytes.
    # We rely on onnx (if installed) or construct the bytes directly.
    try:
        import onnx
        from onnx import helper, TensorProto

        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "test_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        # Pin IR version to 8 (max supported by common onnxruntime builds).
        # Newer onnx packages default to IR v9+ which some runtimes reject.
        model.ir_version = 8
        return model.SerializeToString()
    except ImportError:
        return None  # skip if onnx package is unavailable


# ---------------------------------------------------------------------------
# Tests for load_session() – Requirements 3.1, 3.2, 3.4
# ---------------------------------------------------------------------------

class TestLoadSessionWithRealModel:
    """Tests that use the actual inpainter.onnx model (Requirement 3.1, 3.2)."""

    def test_returns_inference_session(self):
        """Requirement 3.1 – load_session() must return an InferenceSession."""
        session = load_session(MODEL_PATH)
        assert isinstance(session, ort.InferenceSession)

    def test_uses_cpu_provider_only(self):
        """Requirement 3.2 – only CPUExecutionProvider must be active."""
        session = load_session(MODEL_PATH)
        providers = session.get_providers()
        assert providers == ["CPUExecutionProvider"], (
            f"Expected ['CPUExecutionProvider'], got {providers}"
        )

    def test_session_has_inputs(self):
        """The loaded session should expose input metadata (sanity check)."""
        session = load_session(MODEL_PATH)
        inputs = session.get_inputs()
        assert len(inputs) > 0, "Session should have at least one input"

    def test_session_has_outputs(self):
        """The loaded session should expose output metadata (sanity check)."""
        session = load_session(MODEL_PATH)
        outputs = session.get_outputs()
        assert len(outputs) > 0, "Session should have at least one output"

    def test_accepts_path_object(self):
        """load_session() should accept a pathlib.Path without errors."""
        assert isinstance(MODEL_PATH, Path)
        session = load_session(MODEL_PATH)
        assert session is not None

    def test_accepts_string_path(self):
        """load_session() should also accept a string path."""
        session = load_session(str(MODEL_PATH))
        assert isinstance(session, ort.InferenceSession)


class TestLoadSessionErrorHandling:
    """Tests for error handling in load_session() (Requirement 3.4)."""

    def test_raises_file_not_found_for_missing_model(self, tmp_path):
        """Requirement 3.4 – missing file must raise FileNotFoundError."""
        missing = tmp_path / "nonexistent.onnx"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_session(missing)
        # The error message should contain the path so callers know what failed
        assert str(missing) in str(exc_info.value)

    def test_raises_runtime_error_for_invalid_onnx(self, tmp_path):
        """Requirement 3.4 – invalid file content must raise RuntimeError
        (wrapping the onnxruntime exception with context)."""
        bad_model = tmp_path / "bad.onnx"
        bad_model.write_bytes(b"this is not a valid onnx model")

        with pytest.raises(RuntimeError) as exc_info:
            load_session(bad_model)

        error_msg = str(exc_info.value)
        assert str(bad_model) in error_msg, (
            "RuntimeError message should include the model path for diagnosis"
        )

    def test_error_message_contains_original_exception(self, tmp_path):
        """Requirement 3.4 – the RuntimeError should chain the original cause."""
        bad_model = tmp_path / "bad.onnx"
        bad_model.write_bytes(b"\x00\x01\x02\x03corrupt")

        with pytest.raises(RuntimeError) as exc_info:
            load_session(bad_model)

        # The original exception should be chained (__cause__)
        assert exc_info.value.__cause__ is not None


class TestLoadSessionWithMinimalModel:
    """Tests using a minimal in-memory ONNX model (requires onnx package)."""

    @pytest.fixture(autouse=True)
    def _require_onnx_package(self):
        pytest.importorskip("onnx", reason="onnx package required for this test")

    def test_loads_minimal_valid_model(self, tmp_path):
        """load_session() should succeed on any valid ONNX model."""
        model_bytes = _minimal_onnx_bytes()
        if model_bytes is None:
            pytest.skip("Could not build minimal ONNX model bytes")

        model_file = tmp_path / "minimal.onnx"
        model_file.write_bytes(model_bytes)

        session = load_session(model_file)
        assert isinstance(session, ort.InferenceSession)

    def test_minimal_model_uses_cpu_only(self, tmp_path):
        """CPU-only constraint must hold for any valid model, not just big-lama."""
        model_bytes = _minimal_onnx_bytes()
        if model_bytes is None:
            pytest.skip("Could not build minimal ONNX model bytes")

        model_file = tmp_path / "minimal.onnx"
        model_file.write_bytes(model_bytes)

        session = load_session(model_file)
        assert session.get_providers() == ["CPUExecutionProvider"]
