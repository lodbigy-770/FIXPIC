# Requirements Document

## Introduction

שלב 0 (אימות ותיעוד מודל) הוא שלב קדם-תנאי ל-M1 במערכת Cleanup AI. מטרת השלב היא לאמת את קובץ המודל ONNX שהורד ולתעד את הפרמטרים המאומתים שלו ב-MODEL_CARD.md, תוך שמירה על פורמט הסימון הקיים ([אומת], [README], [לא ידוע]). השלב אינו כולל הרצת inference מלאה על תמונות, רק אימות טכני של הקובץ וקריאת metadata מהמודל.

השלב מבצע:
1. אימות SHA256 של `inpainter.onnx`
2. אימות שהקובץ אינו Git LFS pointer
3. טעינת המודל ב-onnxruntime לקריאת metadata (שמות tensors, צורות, טיפוסים)
4. עדכון `MODEL_CARD.md` עם נתונים מאומתים
5. ללא יצירת סקריפטים נוספים או כלי עזר חדשים

## Glossary

- **Model_File**: הקובץ `models/big-lama/inpainter.onnx`
- **MODEL_CARD**: הקובץ `docs/architecture/cleanup-ai/MODEL_CARD.md`
- **LFS_Pointer**: קובץ Git LFS pointer שמכיל מזהה במקום תוכן בינארי
- **ONNX_Session**: אובייקט `onnxruntime.InferenceSession` שנוצר מהמודל
- **Input_Tensors**: tensors הקלט של המודל (תמונה ומסכה)
- **Output_Tensors**: tensors הפלט של המודל (תמונה מעובדת)
- **Validation_System**: המערכת המבצעת את האימות והתיעוד בשלב 0

## Requirements

### Requirement 1

**User Story:** As a developer, I want to verify the Model_File integrity, so that I can trust the file content matches the expected model

#### Acceptance Criteria

1. THE Validation_System SHALL calculate the SHA256 hash of the Model_File
2. THE Validation_System SHALL compare the calculated hash against the expected value `ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff`
3. IF the calculated hash matches the expected hash, THEN THE Validation_System SHALL record the verification status as "[אומת]" in the MODEL_CARD
4. IF the calculated hash does not match the expected hash, THEN THE Validation_System SHALL report an error and halt further processing

### Requirement 2

**User Story:** As a developer, I want to ensure the Model_File is not a Git LFS pointer, so that I can load and process the actual model content

#### Acceptance Criteria

1. THE Validation_System SHALL read the first 200 bytes of the Model_File
2. THE Validation_System SHALL check for the presence of the string "version https://git-lfs.github.com" in the first 200 bytes
3. IF the LFS marker is found, THEN THE Validation_System SHALL report an error indicating the Model_File is an LFS_Pointer
4. IF the LFS marker is not found, THEN THE Validation_System SHALL proceed with model loading
5. THE Validation_System SHALL verify the Model_File size is 208851664 bytes

### Requirement 3

**User Story:** As a developer, I want to load the Model_File using onnxruntime, so that I can extract metadata without running full inference

#### Acceptance Criteria

1. THE Validation_System SHALL create an ONNX_Session from the Model_File using `onnxruntime.InferenceSession`
2. THE Validation_System SHALL use CPU ExecutionProvider only
3. IF the ONNX_Session creation fails, THEN THE Validation_System SHALL report an error with the exception details
4. IF the ONNX_Session creation succeeds, THEN THE Validation_System SHALL proceed to metadata extraction

### Requirement 4

**User Story:** As a developer, I want to extract Input_Tensors metadata from the model, so that I can document the exact input specifications

#### Acceptance Criteria

1. THE Validation_System SHALL read the Input_Tensors metadata from the ONNX_Session
2. THE Validation_System SHALL record the name of each input tensor
3. THE Validation_System SHALL record the shape of each input tensor including dynamic dimensions
4. THE Validation_System SHALL record the data type of each input tensor
5. THE Validation_System SHALL verify there are exactly 2 input tensors
6. THE Validation_System SHALL identify which tensor is the image input and which is the mask input based on shape pattern `[1, 3, H, W]` for image and `[1, 1, H, W]` for mask

### Requirement 5

**User Story:** As a developer, I want to extract Output_Tensors metadata from the model, so that I can document the exact output specifications

#### Acceptance Criteria

1. THE Validation_System SHALL read the Output_Tensors metadata from the ONNX_Session
2. THE Validation_System SHALL record the name of the output tensor
3. THE Validation_System SHALL record the shape of the output tensor including dynamic dimensions
4. THE Validation_System SHALL record the data type of the output tensor
5. THE Validation_System SHALL verify there is exactly 1 output tensor
6. THE Validation_System SHALL verify the output tensor shape matches the pattern `[1, 3, H, W]`

### Requirement 6

**User Story:** As a developer, I want to verify dynamic dimension constraints, so that I can document the supported input sizes

#### Acceptance Criteria

1. WHERE the model metadata includes dimension constraints, THE Validation_System SHALL extract the minimum and maximum values for H and W dimensions
2. THE Validation_System SHALL verify the dimensions are documented as multiples of 8
3. THE Validation_System SHALL record the discovered dimension ranges in the MODEL_CARD
4. IF no explicit constraints are found in metadata, THEN THE Validation_System SHALL record the constraint status as "[לא ידוע - יש לאמת בבדיקה]"

### Requirement 7

**User Story:** As a developer, I want to update MODEL_CARD.md with verified data, so that the documentation reflects actual model characteristics

#### Acceptance Criteria

1. THE Validation_System SHALL update section 3 (מבנה קלט ופלט) in the MODEL_CARD with verified tensor names
2. THE Validation_System SHALL replace "[README – לאמת ב-`probe`]" markers with "[אומת]" for verified information
3. THE Validation_System SHALL preserve all existing "[README]" and "[לא ידוע]" markers that were not verified in this stage
4. THE Validation_System SHALL update the SHA256 status in section 1 from "[pointer]" to "[אומת]"
5. THE Validation_System SHALL update the file size status from "[pointer]" to "[אומת]"
6. THE Validation_System SHALL add a note at the top indicating when the verification was performed
7. THE Validation_System SHALL update the task list in section 8 to mark completed items

### Requirement 8

**User Story:** As a developer, I want to preserve the existing MODEL_CARD format, so that the documentation style remains consistent

#### Acceptance Criteria

1. THE Validation_System SHALL maintain all existing section headers in the MODEL_CARD
2. THE Validation_System SHALL maintain all existing tables in the MODEL_CARD
3. THE Validation_System SHALL preserve the Hebrew language content in the MODEL_CARD
4. THE Validation_System SHALL preserve the existing marker format: [אומת], [README], [pointer], [לא ידוע], [ידע כללי]
5. THE Validation_System SHALL update only the data values and status markers, not the document structure

### Requirement 9

**User Story:** As a developer, I want to avoid creating additional scripts, so that the validation remains a one-time manual task

#### Acceptance Criteria

1. THE Validation_System SHALL perform all validation operations using existing Python capabilities
2. THE Validation_System SHALL NOT create new files in `tools/` directory
3. THE Validation_System SHALL NOT create new script files with `.py` extension for this validation
4. THE Validation_System SHALL use standard library and onnxruntime only
5. THE Validation_System SHALL output results directly to the MODEL_CARD file

### Requirement 10

**User Story:** As a developer, I want a clear validation report, so that I can verify all steps completed successfully

#### Acceptance Criteria

1. THE Validation_System SHALL report the success or failure of each validation step
2. THE Validation_System SHALL report the extracted tensor names, shapes, and types
3. THE Validation_System SHALL report the SHA256 verification result
4. THE Validation_System SHALL report the LFS pointer check result
5. THE Validation_System SHALL report the MODEL_CARD update status
6. IF any step fails, THEN THE Validation_System SHALL report which step failed and why
