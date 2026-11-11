# validation_orchestrator.py
"""
Orchestrator for Stage 0 Model Validation workflow.

Coordinates all validation steps: file validation, model inspection,
MODEL_CARD updates, and reporting. Implements the requirements 1.1-1.4,
2.1-2.4, and 3.1-3.4.
"""

from pathlib import Path
from typing import Dict, Any

from .config import MODEL_PATH, MODEL_CARD_PATH
from .data_structures import ValidationReport, TensorInfo
from .file_validator import compute_sha256, check_file_size, check_lfs_pointer
from .model_inspector import (
    load_session,
    extract_input_metadata,
    extract_output_metadata,
    identify_tensor_roles,
    verify_dimension_constraints,
)
from .model_card_updater import (
    parse_model_card,
    backup_model_card,
    update_section_1,
    update_section_3,
    update_task_list,
    add_verification_note,
    write_model_card,
)
from .report_generator import log_step, format_error, generate_summary


def validate_model() -> ValidationReport:
    """
    Orchestrate the entire Stage 0 model validation workflow.

    This function coordinates all validation steps in sequence:
    1. File validation (SHA256, size, LFS pointer check)
    2. Model session loading
    3. Metadata extraction (inputs, outputs, tensor roles, dimensions)
    4. MODEL_CARD updates (sections 1, 3, 8, verification note)
    5. Error handling (stops on any failure)

    The function uses constants from config.py (MODEL_PATH, MODEL_CARD_PATH)
    and logs each step using report_generator.log_step().

    Returns:
        ValidationReport containing all step results. Use report.is_success()
        to check if all steps passed.

    Requirements:
        - 1.1-1.4: File validation (SHA256, size, LFS check)
        - 2.1-2.4: Model inspection (load, metadata, roles, dimensions)
        - 3.1-3.4: MODEL_CARD updates (parse, update, write)
    """
    report = ValidationReport()

    try:
        # ---------------------------------------------------------------
        # Step 1: File Validation
        # ---------------------------------------------------------------
        log_step("File Validation", "started")

        # 1.1: Compute SHA256 hash
        try:
            sha256_hash = compute_sha256(MODEL_PATH)
            log_step("SHA256 Computation", "success", sha256_hash)
            report.add_step("sha256_computation", "success", sha256_hash)
        except Exception as exc:
            error_msg = format_error(exc, "SHA256 computation")
            log_step("SHA256 Computation", "failed", error_msg)
            report.add_step("sha256_computation", "failed", error_msg)
            return report

        # 1.2: Check file size
        try:
            size_valid = check_file_size(MODEL_PATH)
            if size_valid:
                actual_size = MODEL_PATH.stat().st_size
                log_step("File Size Check", "success", f"{actual_size} bytes")
                report.add_step("file_size_check", "success", actual_size)
            else:
                actual_size = MODEL_PATH.stat().st_size
                error_msg = f"Size mismatch: {actual_size} bytes"
                log_step("File Size Check", "failed", error_msg)
                report.add_step("file_size_check", "failed", error_msg)
                return report
        except Exception as exc:
            error_msg = format_error(exc, "file size check")
            log_step("File Size Check", "failed", error_msg)
            report.add_step("file_size_check", "failed", error_msg)
            return report

        # 1.3: Check for LFS pointer
        try:
            is_lfs_pointer = check_lfs_pointer(MODEL_PATH)
            if is_lfs_pointer:
                error_msg = "File is a Git LFS pointer, not actual binary"
                log_step("LFS Pointer Check", "failed", error_msg)
                report.add_step("lfs_pointer_check", "failed", error_msg)
                return report
            else:
                log_step("LFS Pointer Check", "success", "Not an LFS pointer")
                report.add_step("lfs_pointer_check", "success", "Not an LFS pointer")
        except Exception as exc:
            error_msg = format_error(exc, "LFS pointer check")
            log_step("LFS Pointer Check", "failed", error_msg)
            report.add_step("lfs_pointer_check", "failed", error_msg)
            return report

        # File validation complete
        file_info = {
            "sha256": sha256_hash,
            "size": MODEL_PATH.stat().st_size,
        }
        log_step("File Validation", "success")

        # ---------------------------------------------------------------
        # Step 2: Model Inspection - Load Session
        # ---------------------------------------------------------------
        log_step("Model Session Loading", "started")

        try:
            session = load_session(MODEL_PATH)
            log_step("Model Session Loading", "success")
            report.add_step("session_loading", "success", "Session loaded")
        except Exception as exc:
            error_msg = format_error(exc, "session loading")
            log_step("Model Session Loading", "failed", error_msg)
            report.add_step("session_loading", "failed", error_msg)
            return report

        # ---------------------------------------------------------------
        # Step 3: Model Inspection - Extract Metadata
        # ---------------------------------------------------------------
        log_step("Metadata Extraction", "started")

        # 2.1: Extract input metadata
        try:
            input_tensors = extract_input_metadata(session)
            log_step("Input Metadata Extraction", "success", f"{len(input_tensors)} inputs")
            report.add_step("input_metadata", "success", input_tensors)
        except Exception as exc:
            error_msg = format_error(exc, "input metadata extraction")
            log_step("Input Metadata Extraction", "failed", error_msg)
            report.add_step("input_metadata", "failed", error_msg)
            return report

        # 2.2: Extract output metadata
        try:
            output_tensors = extract_output_metadata(session)
            log_step("Output Metadata Extraction", "success", f"{len(output_tensors)} outputs")
            report.add_step("output_metadata", "success", output_tensors)
        except Exception as exc:
            error_msg = format_error(exc, "output metadata extraction")
            log_step("Output Metadata Extraction", "failed", error_msg)
            report.add_step("output_metadata", "failed", error_msg)
            return report

        # 2.3: Identify tensor roles (image, mask)
        try:
            tensor_roles = identify_tensor_roles(input_tensors)
            log_step(
                "Tensor Role Identification",
                "success",
                f"image={tensor_roles['image'].name}, mask={tensor_roles['mask'].name}",
            )
            report.add_step("tensor_roles", "success", tensor_roles)
        except Exception as exc:
            error_msg = format_error(exc, "tensor role identification")
            log_step("Tensor Role Identification", "failed", error_msg)
            report.add_step("tensor_roles", "failed", error_msg)
            return report

        # 2.4: Verify dimension constraints
        try:
            dimension_constraints = verify_dimension_constraints(session)
            log_step("Dimension Constraints", "success", dimension_constraints.get("note", ""))
            report.add_step("dimension_constraints", "success", dimension_constraints)
        except Exception as exc:
            error_msg = format_error(exc, "dimension constraint verification")
            log_step("Dimension Constraints", "failed", error_msg)
            report.add_step("dimension_constraints", "failed", error_msg)
            return report

        # Combine metadata for MODEL_CARD update
        metadata = {
            "image": tensor_roles["image"],
            "mask": tensor_roles["mask"],
            "output": output_tensors[0],  # Assuming single output
            "dimension_constraints": dimension_constraints,
        }

        log_step("Metadata Extraction", "success")

        # ---------------------------------------------------------------
        # Step 4: MODEL_CARD Updates
        # ---------------------------------------------------------------
        log_step("MODEL_CARD Update", "started")

        # 3.1: Parse MODEL_CARD
        try:
            model_card_content = parse_model_card(MODEL_CARD_PATH)
            log_step("MODEL_CARD Parsing", "success")
            report.add_step("model_card_parsing", "success", "Parsed")
        except Exception as exc:
            error_msg = format_error(exc, "MODEL_CARD parsing")
            log_step("MODEL_CARD Parsing", "failed", error_msg)
            report.add_step("model_card_parsing", "failed", error_msg)
            return report

        # 3.2: Backup MODEL_CARD
        try:
            backup_path = backup_model_card(MODEL_CARD_PATH)
            log_step("MODEL_CARD Backup", "success", str(backup_path))
            report.add_step("model_card_backup", "success", str(backup_path))
        except Exception as exc:
            error_msg = format_error(exc, "MODEL_CARD backup")
            log_step("MODEL_CARD Backup", "failed", error_msg)
            report.add_step("model_card_backup", "failed", error_msg)
            return report

        # 3.3: Update section 1 (SHA256, file size markers)
        try:
            model_card_content = update_section_1(model_card_content, file_info)
            log_step("Section 1 Update", "success")
            report.add_step("section_1_update", "success", "Markers updated")
        except Exception as exc:
            error_msg = format_error(exc, "section 1 update")
            log_step("Section 1 Update", "failed", error_msg)
            report.add_step("section_1_update", "failed", error_msg)
            return report

        # 3.4: Update section 3 (tensor names)
        try:
            model_card_content = update_section_3(model_card_content, metadata)
            log_step("Section 3 Update", "success")
            report.add_step("section_3_update", "success", "Tensor names updated")
        except Exception as exc:
            error_msg = format_error(exc, "section 3 update")
            log_step("Section 3 Update", "failed", error_msg)
            report.add_step("section_3_update", "failed", error_msg)
            return report

        # 3.5: Update task list (section 8)
        # Match substrings that actually appear in the MODEL_CARD section 8 task lines:
        #   - "להשיג את תוכן הקובץ ולאמת SHA256 מול ה-pointer."  → matched by "SHA256"
        #   - "`probe`: אימות שמות, טיפוסים ומידות דינמיות (סעיף 3)." → matched by "probe"
        try:
            completed_tasks = [
                "SHA256",
                "probe",
            ]
            model_card_content = update_task_list(model_card_content, completed_tasks)
            log_step("Task List Update", "success")
            report.add_step("task_list_update", "success", "Tasks marked complete")
        except Exception as exc:
            error_msg = format_error(exc, "task list update")
            log_step("Task List Update", "failed", error_msg)
            report.add_step("task_list_update", "failed", error_msg)
            return report

        # 3.6: Add verification note
        try:
            model_card_content = add_verification_note(model_card_content)
            log_step("Verification Note", "success")
            report.add_step("verification_note", "success", "Note added")
        except Exception as exc:
            error_msg = format_error(exc, "verification note")
            log_step("Verification Note", "failed", error_msg)
            report.add_step("verification_note", "failed", error_msg)
            return report

        # 3.7: Write MODEL_CARD
        try:
            write_model_card(MODEL_CARD_PATH, model_card_content)
            log_step("MODEL_CARD Write", "success")
            report.add_step("model_card_write", "success", "Written to disk")
        except Exception as exc:
            error_msg = format_error(exc, "MODEL_CARD write")
            log_step("MODEL_CARD Write", "failed", error_msg)
            report.add_step("model_card_write", "failed", error_msg)
            return report

        log_step("MODEL_CARD Update", "success")

        # ---------------------------------------------------------------
        # Validation Complete
        # ---------------------------------------------------------------
        log_step("Validation Complete", "success")
        report.add_step("validation_complete", "success", "All steps passed")

    except Exception as exc:
        # Catch-all for any unexpected errors
        error_msg = format_error(exc, "validation orchestration")
        log_step("Validation", "failed", error_msg)
        report.add_step("validation_orchestration", "failed", error_msg)

    # Generate and print summary report
    summary = generate_summary(report)
    print("\n" + summary)

    return report


def get_exit_code(report: ValidationReport) -> int:
    """
    Map validation report results to an appropriate exit code.

    Exit codes:
        0  — all steps succeeded
        1  — SHA256 or file size check failed (or any unmapped failure)
        2  — LFS pointer check failed
        3  — session loading failed
        4  — input or output metadata extraction failed
        5  — tensor role identification failed
        6  — MODEL_CARD write failed

    Args:
        report: ValidationReport containing all validation steps

    Returns:
        Integer exit code representing the first / most-specific failure.

    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """
    if report.is_success():
        return 0

    # Walk through steps in order and return the exit code for the first failure
    for name, status, _details in report.steps:
        if status != "failed":
            continue

        if name in ("sha256_computation", "file_size_check"):
            return 1
        if name == "lfs_pointer_check":
            return 2
        if name == "session_loading":
            return 3
        if name in ("input_metadata", "output_metadata"):
            return 4
        if name == "tensor_roles":
            return 5
        if name == "model_card_write":
            return 6

    # Fallback: some other step failed
    return 1
