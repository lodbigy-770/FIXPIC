# model_inspector.py
"""
ModelInspector component for Stage 0 Model Validation.
Provides ONNX session loading and tensor metadata extraction.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import onnxruntime as ort

from .data_structures import TensorInfo


def load_session(model_path: Path) -> ort.InferenceSession:
    """
    Load an ONNX InferenceSession using CPU ExecutionProvider only.

    Creates SessionOptions configured for sequential execution with full
    graph optimisations, then loads the model from *model_path*.

    Args:
        model_path: Path to the ONNX model file (e.g. inpainter.onnx).

    Returns:
        A loaded ``onnxruntime.InferenceSession`` object ready for
        metadata inspection.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        ort.capi.onnxruntime_pybind11_state.InvalidGraph: If the model
            file is not a valid ONNX graph.
        Exception: Any other onnxruntime error is re-raised with an
            enriched message that includes *model_path* and the original
            exception details.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    try:
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        return session

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ONNX session from '{model_path}': {exc}"
        ) from exc


def extract_input_metadata(session: ort.InferenceSession) -> List[TensorInfo]:
    """
    Extract metadata for all input tensors of an ONNX session.

    Reads ``session.get_inputs()`` and converts each NodeArg into a
    :class:`TensorInfo` dataclass.  Dynamic dimensions are preserved as
    their original type (string symbol or ``None``).

    Args:
        session: A loaded ``onnxruntime.InferenceSession``.

    Returns:
        List of :class:`TensorInfo` objects, one per input tensor.
    """
    return [
        TensorInfo(name=inp.name, shape=inp.shape, dtype=inp.type)
        for inp in session.get_inputs()
    ]


def extract_output_metadata(session: ort.InferenceSession) -> List[TensorInfo]:
    """
    Extract metadata for all output tensors of an ONNX session.

    Reads ``session.get_outputs()`` and converts each NodeArg into a
    :class:`TensorInfo` dataclass.

    Args:
        session: A loaded ``onnxruntime.InferenceSession``.

    Returns:
        List of :class:`TensorInfo` objects, one per output tensor.
    """
    return [
        TensorInfo(name=out.name, shape=out.shape, dtype=out.type)
        for out in session.get_outputs()
    ]


def identify_tensor_roles(inputs: List[TensorInfo]) -> Dict[str, TensorInfo]:
    """
    Identify image and mask tensors from a list of input tensors.

    Uses the channel-dimension heuristic defined in the design spec:
    - Image tensor: shape ``[1, 3, H, W]`` (3 channels)
    - Mask tensor:  shape ``[1, 1, H, W]`` (1 channel)

    Falls back to name-based detection ("mask" in tensor name) when
    shape-based detection is inconclusive.

    Args:
        inputs: List of input :class:`TensorInfo` objects.

    Returns:
        Dict with keys ``"image"`` and ``"mask"``, each mapping to the
        corresponding :class:`TensorInfo`.

    Raises:
        ValueError: If the number of input tensors is not exactly 2, or
            if image / mask roles cannot be determined.
    """
    if len(inputs) != 2:
        raise ValueError(
            f"Expected exactly 2 input tensors, found {len(inputs)}: "
            f"{[t.name for t in inputs]}"
        )

    image_tensor: TensorInfo | None = None
    mask_tensor: TensorInfo | None = None

    for tensor in inputs:
        shape = tensor.shape
        if len(shape) == 4:
            channel_dim = shape[1]
            if channel_dim == 3:
                image_tensor = tensor
            elif channel_dim == 1:
                mask_tensor = tensor

    # Fallback: use name-based detection if shape detection was inconclusive
    if image_tensor is None or mask_tensor is None:
        for tensor in inputs:
            if "mask" in tensor.name.lower() and mask_tensor is None:
                mask_tensor = tensor
            elif image_tensor is None:
                image_tensor = tensor

    if image_tensor is None:
        raise ValueError(
            "Could not identify image tensor (expected shape [1, 3, H, W]). "
            f"Tensors found: {[(t.name, t.shape) for t in inputs]}"
        )
    if mask_tensor is None:
        raise ValueError(
            "Could not identify mask tensor (expected shape [1, 1, H, W]). "
            f"Tensors found: {[(t.name, t.shape) for t in inputs]}"
        )

    return {"image": image_tensor, "mask": mask_tensor}


# ---------------------------------------------------------------------------
# Dimension constraint helpers
# ---------------------------------------------------------------------------

# Regex that matches a "multiple-of-N" symbolic dimension such as "8*height"
# Captured groups: (factor, variable_name)
_MULTIPLE_PATTERN = re.compile(r'^(\d+)\*(\w+)$')


def _collect_all_shapes(session: ort.InferenceSession) -> List[List[Any]]:
    """Return all shape lists from both inputs and outputs."""
    shapes = []
    for node in session.get_inputs() + session.get_outputs():
        shapes.append(node.shape)
    return shapes


def verify_dimension_constraints(session: ort.InferenceSession) -> Dict[str, Any]:
    """
    Discover and verify dimension constraints for H and W from an ONNX session.

    The function probes two sources of information:

    1. **Tensor shape symbols** – dynamic dimensions expressed as strings like
       ``"8*height"`` and ``"8*width"`` directly encode a multiple-of-8
       constraint without needing extra model metadata.
    2. **Model custom metadata** (``session.get_modelmeta().custom_metadata_map``) –
       optional key/value pairs that may carry explicit min/max bounds for
       ``height``/``width`` (e.g. ``"height_min"``, ``"width_max"``).

    Args:
        session: A loaded ``onnxruntime.InferenceSession``.

    Returns:
        A dict describing the discovered constraints.  Two possible shapes:

        * When a multiple-of-8 pattern is found in shape symbols::

            {
                "constraint":   "multiple_of_8",
                "h_symbol":     "8*height",   # exact string from tensor shape
                "w_symbol":     "8*width",    # exact string from tensor shape
                "h_min":        None,         # or int if found in metadata
                "h_max":        None,
                "w_min":        None,
                "w_max":        None,
                "note":         "[אומת]",
            }

        * When no recognisable constraint is found::

            {
                "note": "[לא ידוע - יש לאמת בבדיקה]",
            }

    Requirements:
        - 6.1 Extract H/W min/max from metadata when available
        - 6.2 Verify dimensions are documented as multiples of 8
        - 6.3 Record discovered constraints
        - 6.4 Record "[לא ידוע - יש לאמת בבדיקה]" if not found
    """
    # ------------------------------------------------------------------
    # Step 1: scan tensor shape symbols for multiple-of-8 evidence
    # ------------------------------------------------------------------
    h_symbol: Optional[str] = None
    w_symbol: Optional[str] = None

    for shape in _collect_all_shapes(session):
        for dim in shape:
            if not isinstance(dim, str):
                continue
            m = _MULTIPLE_PATTERN.match(dim)
            if m is None:
                continue
            factor = int(m.group(1))
            var = m.group(2).lower()
            if factor % 8 == 0:
                # Heuristic: variable name containing 'h' / 'height' → H dim
                if h_symbol is None and ('height' in var or var == 'h'):
                    h_symbol = dim
                # variable name containing 'w' / 'width' → W dim
                elif w_symbol is None and ('width' in var or var == 'w'):
                    w_symbol = dim

    # ------------------------------------------------------------------
    # Step 2: check custom_metadata_map for explicit min/max bounds
    # ------------------------------------------------------------------
    metadata_map: Dict[str, str] = session.get_modelmeta().custom_metadata_map

    def _int_or_none(key: str) -> Optional[int]:
        val = metadata_map.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    h_min = _int_or_none("height_min") or _int_or_none("h_min") or _int_or_none("min_height")
    h_max = _int_or_none("height_max") or _int_or_none("h_max") or _int_or_none("max_height")
    w_min = _int_or_none("width_min") or _int_or_none("w_min") or _int_or_none("min_width")
    w_max = _int_or_none("width_max") or _int_or_none("w_max") or _int_or_none("max_width")

    # ------------------------------------------------------------------
    # Step 3: compose result
    # ------------------------------------------------------------------
    if h_symbol is not None or w_symbol is not None:
        return {
            "constraint": "multiple_of_8",
            "h_symbol": h_symbol,
            "w_symbol": w_symbol,
            "h_min": h_min,
            "h_max": h_max,
            "w_min": w_min,
            "w_max": w_max,
            "note": "[אומת]",
        }

    # No evidence found in shapes; check metadata for any numeric bounds
    if any(v is not None for v in (h_min, h_max, w_min, w_max)):
        return {
            "constraint": "explicit_bounds_only",
            "h_symbol": None,
            "w_symbol": None,
            "h_min": h_min,
            "h_max": h_max,
            "w_min": w_min,
            "w_max": w_max,
            "note": "[לא ידוע - יש לאמת בבדיקה]",
        }

    return {"note": "[לא ידוע - יש לאמת בבדיקה]"}
