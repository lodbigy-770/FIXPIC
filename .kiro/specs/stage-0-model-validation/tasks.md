# Implementation Plan: Stage 0 Model Validation

## Overview

שלב 0 הוא תהליך אימות חד-פעמי לאימות תקינות קובץ המודל ONNX ולעדכון MODEL_CARD.md עם נתונים מאומתים. המימוש יבנה מערכת validation מודולרית בפייתון עם 4 רכיבים עיקריים: FileValidator (אימות קובץ), ModelInspector (חילוץ metadata), ModelCardUpdater (עדכון תיעוד), ו-ReportGenerator (דיווחים). כל שלב מותנה בהצלחת הקודם, והתהליך יעצר במקרה של שגיאה.

## Tasks

- [x] 1. הגדרת תצורה ומבנה פרויקט
  - יצירת קובץ config.py עם קבועים: נתיבי קבצים, SHA256 צפוי, גודל קובץ צפוי, והגדרות אחרות
  - הגדרת מבנה הדאטה TensorInfo (dataclass) עם שדות: name, shape, dtype
  - הגדרת מבנה ValidationReport (dataclass) עם רשימת שלבים ושיטות add_step() ו-is_success()
  - _Requirements: 1.1, 1.2, 2.1, 2.5_

- [x] 2. מימוש FileValidator לאימות תקינות קובץ
  - [x] 2.1 מימוש compute_sha256()
    - חישוב SHA256 בבלוקים של 8KB לחיסכון בזיכרון
    - השוואת hash מחושב לערך הצפוי
    - החזרת מחרוזת hex של ה-hash
    - _Requirements: 1.1, 1.2_
  
  - [x] 2.2 מימוש check_file_size()
    - קריאת גודל הקובץ באמצעות Path.stat().st_size
    - השוואה לגודל הצפוי (208,851,664 bytes)
    - החזרת boolean לאימות התאמה
    - _Requirements: 2.5_
  
  - [x] 2.3 מימוש check_lfs_pointer()
    - קריאת 200 bytes ראשונים מהקובץ
    - בדיקת נוכחות המחרוזת "version https://git-lfs.github.com"
    - החזרת True אם זה LFS pointer (שגיאה), False אחרת
    - _Requirements: 2.1, 2.2, 2.3_

- [~] 3. Checkpoint - אימות FileValidator
  - הרצת בדיקות יחידה ל-FileValidator
  - אימוד פונקציות על קובץ המודל בפועל
  - ודא שכל הבדיקות עוברות, שאל את המשתמש אם מתעוררות שאלות

- [x] 4. מימוש ModelInspector לחילוץ metadata מהמודל
  - [x] 4.1 מימוש load_session()
    - יצירת SessionOptions עם CPU provider בלבד
    - טעינת InferenceSession מנתיב המודל
    - טיפול בחריגות והחזרת session object
    - _Requirements: 3.1, 3.2, 3.4_
  
  - [x] 4.2 מימוש extract_input_metadata()
    - קריאת session.get_inputs()
    - חילוץ name, shape, dtype לכל input tensor
    - החזרת רשימת TensorInfo objects
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 4.3 מימוש extract_output_metadata()
    - קריאת session.get_outputs()
    - חילוץ name, shape, dtype לכל output tensor
    - החזרת רשימת TensorInfo objects
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 4.4 מימוש identify_tensor_roles()
    - זיהוי tensor תמונה לפי צורה [1, 3, H, W]
    - זיהוי tensor מסכה לפי צורה [1, 1, H, W]
    - אימות שיש בדיוק 2 input tensors
    - החזרת מילון {"image": TensorInfo, "mask": TensorInfo}
    - _Requirements: 4.6_
  
  - [x] 4.5 מימוש verify_dimension_constraints()
    - חיפוש אילוצים במטא-דאטה של המודל (min/max לערכי H ו-W)
    - אימות שהמימדים הם כפולות של 8
    - רישום האילוצים שהתגלו או "[לא ידוע - יש לאמת בבדיקה]"
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [~] 5. Checkpoint - אימות ModelInspector
  - הרצת בדיקות יחידה ל-ModelInspector
  - אימות טעינת session וחילוץ metadata מהמודל
  - ודא שכל הבדיקות עוברות, שאל את המשתמש אם מתעוררות שאלות

- [ ] 6. מימוש ModelCardUpdater לעדכון תיעוד
  - [-] 6.1 מימוש parse_model_card()
    - קריאת MODEL_CARD.md כמחרוזת
    - שמירת כל התוכן בזיכרון לעדכון
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 6.2 מימוש backup_model_card()
    - יצירת קובץ גיבוי עם סיומת .md.bak
    - העתקת MODEL_CARD.md לגיבוי לפני עדכון
    - _Requirements: 7.4, 7.5_
  
  - [~] 6.3 מימוש update_section_1()
    - עדכון סטטוס SHA256 מ-"[pointer]" ל-"[אומת]"
    - עדכון סטטוס גודל קובץ מ-"[pointer]" ל-"[אומת]"
    - שמירה על מבנה הטבלה המקורי
    - _Requirements: 7.4, 7.5, 8.1, 8.2, 8.4_
  
  - [~] 6.4 מימוש update_section_3()
    - עדכון שמות tensors בטבלה (image, mask, output)
    - החלפת "[README – לאמת ב-`probe`]" ב-"[אומת]"
    - שמירה על כל הסימונים האחרים ([README], [לא ידוע], [ידע כללי])
    - _Requirements: 7.1, 7.2, 7.3, 8.1, 8.2, 8.4_
  
  - [~] 6.5 מימוש update_task_list()
    - מציאת סעיף 8 (משימות להשלמת הכרטיס)
    - החלפת "- [ ]" ב-"- [x]" למשימות שהושלמו (SHA256 ו-probe)
    - שמירה על משימות שטרם הושלמו
    - _Requirements: 7.7_
  
  - [~] 6.6 מימוש add_verification_note()
    - יצירת הערת אימות עם timestamp
    - הוספת ההערה בראש המסמך אחרי הכותרת
    - פורמט: "> אומת בשלב 0 ב-{timestamp}"
    - _Requirements: 7.6, 8.3_
  
  - [~] 6.7 מימוש write_model_card()
    - כתיבת התוכן המעודכן חזרה ל-MODEL_CARD.md
    - שימוש ב-UTF-8 encoding לשמירת תווים עבריים
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.3_

- [~] 7. Checkpoint - אימות ModelCardUpdater
  - הרצת בדיקות יחידה ל-ModelCardUpdater
  - אימות שמירת מבנה ועדכון סימונים נכונים
  - ודא שכל הבדיקות עוברות, שאל את המשתמש אם מתעוררות שאלות

- [ ] 8. מימוש ReportGenerator ליצירת דוחות
  - [~] 8.1 מימוש log_step()
    - הדפסת שם שלב וסטטוס לקונסול
    - פורמט: "[STEP_NAME] status"
    - הוספת פרטים אופציונלית
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [~] 8.2 מימוש generate_summary()
    - יצירת סיכום מפורט של כל התהליך
    - הצגת סטטוס כל שלב עם סימונים (✓/✗)
    - הצגת נתונים שהתגלו ושגיאות (אם היו)
    - _Requirements: 10.1, 10.6_
  
  - [~] 8.3 מימוש format_error()
    - עיצוב הודעת שגיאה קריאה עם הקשר
    - הוספת סוג השגיאה, הודעה, ו-stack trace
    - _Requirements: 10.6_

- [ ] 9. מימוש ValidationOrchestrator לתיאום התהליך
  - [~] 9.1 יצירת validate_model() ראשית
    - יצירת ValidationReport
    - קריאה ל-FileValidator.validate()
    - קריאה ל-ModelInspector.load_session() וextract_metadata()
    - קריאה ל-ModelCardUpdater.update()
    - טיפול בחריגות בכל שלב (הפסקה במקרה של כישלון)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_
  
  - [~] 9.2 הוספת לוגיקת דיווח ושגיאות
    - קריאה ל-ReportGenerator.log_step() בכל שלב
    - קריאה ל-ReportGenerator.generate_summary() בסיום
    - החזרת קוד יציאה מתאים (0 להצלחה, 1-6 לשגיאות שונות)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [~] 10. יצירת נקודת כניסה ראשית
  - יצירת validate_stage0.py כסקריפט ראשי
  - קריאה ל-ValidationOrchestrator.validate_model()
  - טיפול ב-command line arguments (אם נדרש)
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [~] 11. Checkpoint סופי - אינטגרציה ובדיקה מקיפה
  - הרצת התהליך המלא על קובץ המודל בפועל
  - אימות שכל השלבים עוברים בהצלחה
  - בדיקת MODEL_CARD.md המעודכן לוודא שמירת מבנה ועדכונים נכונים
  - ודא שכל הבדיקות עוברות, שאל את המשתמש אם מתעוררות שאלות

- [ ]* 12. כתיבת unit tests לרכיבים בודדים
  - [ ]* 12.1 כתיבת test_file_validator.py
    - בדיקות ל-compute_sha256() עם קובץ ידוע
    - בדיקות ל-check_file_size() עם גדלים שונים
    - בדיקות ל-check_lfs_pointer() עם LFS pointer ממשי ועם קובץ רגיל
  
  - [ ]* 12.2 כתיבת test_model_inspector.py
    - בדיקות ל-load_session() עם מודל תקין
    - בדיקות ל-extract_input/output_metadata()
    - בדיקות ל-identify_tensor_roles() עם צורות שונות
  
  - [ ]* 12.3 כתיבת test_model_card_updater.py
    - בדיקות ל-update_section() לוודא שמירת מבנה
    - בדיקות ל-update_markers() עם סימונים שונים
    - בדיקות ל-add_verification_note() לוודא פורמט נכון

- [ ]* 13. כתיבת property-based tests לתכונות אוניברסליות
  - [ ]* 13.1 Property test לזיהוי תפקידי טנסורים
    - **Property 1: זיהוי תפקידי טנסורים לפי צורה**
    - **Validates: Requirements 4.6**
    - יצירת tensors עם צורות שונות [1, 3, H, W] ו-[1, 1, H, W]
    - וידוא שהזיהוי נכון בכל המקרים
  
  - [ ]* 13.2 Property test לאימות תבנית פלט
    - **Property 2: אימות תבנית פלט**
    - **Validates: Requirements 5.6**
    - בדיקה שכל output tensor תואם לתבנית [1, 3, H, W]
  
  - [ ]* 13.3 Property test לשמירת מבנה MODEL_CARD
    - **Property 4: שמירת מבנה MODEL_CARD**
    - **Validates: Requirements 8.1, 8.2, 8.5**
    - וידוא שכל הכותרות, טבלאות ומבנה נשמרים אחרי עדכון
    - רק ערכי נתונים וסימוני סטטוס משתנים
  
  - [ ]* 13.4 Property test לניהול סימוני סטטוס
    - **Property 5: ניהול עקבי של סימוני סטטוס**
    - **Validates: Requirements 7.2, 7.3, 8.4**
    - וידוא שרק [pointer] ו-[README – לאמת ב-probe] משתנים ל-[אומת]
    - כל הסימונים האחרים נשמרים
  
  - [ ]* 13.5 Property test לשמירת תוכן עברית
    - **Property 6: שמירת תוכן בעברית**
    - **Validates: Requirements 8.3**
    - וידוא שכל התווים העבריים נשמרים ללא שיבוש
  
  - [ ]* 13.6 Property test לעדכון רשימת משימות
    - **Property 7: עדכון רשימת משימות**
    - **Validates: Requirements 7.7**
    - וידוא שרק משימות שהושלמו מסומנות [x]
    - משימות שלא הושלמו נשארות [ ]

- [~] 14. תיעוד השימוש והפעלה
  - הוספת הוראות הפעלה ל-README או למסמך נפרד
  - תיעוד דרישות (onnxruntime>=1.30.0)
  - דוגמאות שימוש והפלט הצפוי
  - _Requirements: 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

## Notes

- משימות המסומנות ב-`*` הן אופציונליות וניתן לדלג עליהן למען מהירות
- כל משימה מפנה לדרישות ספציפיות לעקיבות
- Checkpoints מבטיחים אימות הדרגתי לפני המשך
- Property tests מאמתים תכונות אוניברסליות מעבר לדוגמאות בודדות
- השפה: Python עם שימוש ב-pathlib, dataclasses, hashlib, ו-onnxruntime
- הקוד צריך לטפל בתווים עבריים (UTF-8 encoding)
- הסקריפט הוא חד-פעמי ולא צריך להישמר בתיקיית tools/

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "4.5"] },
    { "id": 4, "tasks": ["6.1", "6.2"] },
    { "id": 5, "tasks": ["6.3", "6.4", "6.5", "6.6"] },
    { "id": 6, "tasks": ["6.7"] },
    { "id": 7, "tasks": ["8.1", "8.2", "8.3"] },
    { "id": 8, "tasks": ["9.1"] },
    { "id": 9, "tasks": ["9.2"] },
    { "id": 10, "tasks": ["10"] },
    { "id": 11, "tasks": ["12.1", "12.2", "12.3"] },
    { "id": 12, "tasks": ["13.1", "13.2", "13.3", "13.4", "13.5", "13.6"] },
    { "id": 13, "tasks": ["14"] }
  ]
}
```
