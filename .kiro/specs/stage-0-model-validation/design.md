# Design Document – Stage 0: Model Validation

## 1. אסטרטגיה כללית

שלב 0 הוא תהליך אימות חד-פעמי שמטרתו לאמת את תקינות קובץ המודל ONNX ולעדכן את `MODEL_CARD.md` עם מידע מאומת. התכנון מתמקד בפשטות ובמהימנות, תוך שימוש בכלים קיימים ללא יצירת סקריפטים נוספים.

### 1.1 עקרונות מנחים

1. **אי-הרסנות**: כל תהליך אימות שנכשל לא ישנה קבצים קיימים
2. **אטומיות**: עדכון `MODEL_CARD.md` יתבצע רק לאחר הצלחת כל שלבי האימות
3. **שקיפות**: כל שלב ידווח על תוצאותיו באופן מפורש
4. **ללא תלות**: אין צורך בכלים חיצוניים מעבר לספריות Python הסטנדרטיות ו-onnxruntime

### 1.2 סדר ביצוע

התהליך מורכב מ-5 שלבים עיקריים שחייבים להתבצע לפי הסדר:

```
1. אימות SHA256 וגודל קובץ
   ↓
2. בדיקת Git LFS pointer
   ↓
3. טעינת ONNX session וקריאת metadata
   ↓
4. ניתוח מבנה הטנסורים
   ↓
5. עדכון MODEL_CARD.md
```

כל שלב מותנה בהצלחת הקודם. כישלון בכל שלב יעצור את התהליך ויפיק דיווח שגיאה מפורט.

## 2. ארכיטקטורה

### 2.1 רכיבים עיקריים

```
ValidationOrchestrator
├── FileValidator
│   ├── compute_sha256()
│   ├── check_file_size()
│   └── check_lfs_pointer()
├── ModelInspector
│   ├── load_session()
│   ├── extract_input_metadata()
│   ├── extract_output_metadata()
│   └── identify_tensor_roles()
├── ModelCardUpdater
│   ├── parse_model_card()
│   ├── update_section()
│   ├── update_markers()
│   └── write_model_card()
└── ReportGenerator
    ├── log_step()
    ├── generate_summary()
    └── format_error()
```

### 2.2 תזרים נתונים

```python
# פסאודו-קוד לתזרים הכללי
def validate_model():
    report = ValidationReport()
    
    # שלב 1: אימות קובץ
    file_info = FileValidator.validate(MODEL_PATH)
    report.add_step("file_validation", file_info)
    
    # שלב 2: טעינת מודל
    session = ModelInspector.load_session(MODEL_PATH)
    report.add_step("session_creation", session)
    
    # שלב 3: חילוץ metadata
    metadata = ModelInspector.extract_metadata(session)
    report.add_step("metadata_extraction", metadata)
    
    # שלב 4: עדכון MODEL_CARD
    ModelCardUpdater.update(metadata, file_info)
    report.add_step("model_card_update", "success")
    
    # שלב 5: דיווח
    ReportGenerator.print_summary(report)
    
    return report
```

## 3. מפרט רכיבים מפורט

### 3.1 FileValidator

**אחריות**: אימות תקינות הקובץ הפיזי

#### compute_sha256(file_path: Path) -> str
- קורא את הקובץ בבלוקים (8KB לכל פעם) ומחשב SHA256
- מטרה: לא לטעון 200MB+ לזיכרון בבת אחת
- מחזיר: מחרוזת hash hex

```python
import hashlib

def compute_sha256(file_path: Path) -> str:
    """חישוב SHA256 בבלוקים לחיסכון בזיכרון"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
```

#### check_file_size(file_path: Path, expected: int) -> bool
- קורא `file_path.stat().st_size`
- משווה לערך הצפוי: 208,851,664 bytes
- מחזיר: True אם תואם, False אחרת

#### check_lfs_pointer(file_path: Path) -> bool
- קורא 200 bytes ראשונים
- בודק נוכחות של `"version https://git-lfs.github.com"`
- מחזיר: True אם זה LFS pointer (שגיאה), False אם זה קובץ רגיל

```python
def check_lfs_pointer(file_path: Path) -> bool:
    """בדיקה אם הקובץ הוא Git LFS pointer"""
    with open(file_path, 'rb') as f:
        header = f.read(200)
    return b"version https://git-lfs.github.com" in header
```

### 3.2 ModelInspector

**אחריות**: טעינת המודל וחילוץ metadata

#### load_session(model_path: Path) -> ort.InferenceSession
- יוצר `SessionOptions` עם CPU provider בלבד
- קורא ל-`ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])`
- זורק חריגה במקרה של כישלון (תיתפס ע"י orchestrator)

```python
import onnxruntime as ort

def load_session(model_path: Path) -> ort.InferenceSession:
    """טעינת ONNX session עם CPU provider"""
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    return ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=["CPUExecutionProvider"]
    )
```

#### extract_input_metadata(session: ort.InferenceSession) -> List[TensorInfo]
- קורא `session.get_inputs()`
- עבור כל input: שם, shape, dtype
- מחזיר רשימת `TensorInfo` objects

```python
@dataclass
class TensorInfo:
    name: str
    shape: List[Union[int, str]]  # ערכים דינמיים כמחרוזות
    dtype: str

def extract_input_metadata(session: ort.InferenceSession) -> List[TensorInfo]:
    """חילוץ metadata של כל tensors הקלט"""
    inputs = []
    for inp in session.get_inputs():
        inputs.append(TensorInfo(
            name=inp.name,
            shape=inp.shape,
            dtype=inp.type
        ))
    return inputs
```

#### extract_output_metadata(session: ort.InferenceSession) -> List[TensorInfo]
- זהה ל-`extract_input_metadata` אבל ל-outputs
- קורא `session.get_outputs()`

#### identify_tensor_roles(inputs: List[TensorInfo]) -> Dict[str, TensorInfo]
- מזהה איזה tensor הוא תמונה ואיזה מסכה לפי הצורה:
  - תמונה: `[1, 3, H, W]` (3 ערוצים)
  - מסכה: `[1, 1, H, W]` (ערוץ אחד)
- מחזיר: `{"image": TensorInfo, "mask": TensorInfo}`
- זורק שגיאה אם לא נמצאו בדיוק 2 tensors או אם הזיהוי נכשל

```python
def identify_tensor_roles(inputs: List[TensorInfo]) -> Dict[str, TensorInfo]:
    """זיהוי תפקידי הטנסורים לפי הצורה"""
    if len(inputs) != 2:
        raise ValueError(f"Expected 2 input tensors, found {len(inputs)}")
    
    image_tensor = None
    mask_tensor = None
    
    for tensor in inputs:
        if len(tensor.shape) == 4 and tensor.shape[1] == 3:
            image_tensor = tensor
        elif len(tensor.shape) == 4 and tensor.shape[1] == 1:
            mask_tensor = tensor
    
    if not image_tensor or not mask_tensor:
        raise ValueError("Could not identify image and mask tensors")
    
    return {"image": image_tensor, "mask": mask_tensor}
```

### 3.3 ModelCardUpdater

**אחריות**: עדכון `MODEL_CARD.md` עם נתונים מאומתים

#### parse_model_card(path: Path) -> str
- קורא את `MODEL_CARD.md` כמחרוזת
- שומר את כל התוכן בזיכרון לעדכון

#### update_section(content: str, section: int, new_data: Dict) -> str
- מזהה את הסעיף המבוקש (לפי כותרת `## N.`)
- מעדכן טבלאות/שדות בתוך הסעיף
- שומר על המבנה המקורי (שורות, רווחים, כותרות)

**אסטרטגיה לעדכון סעיף 3** (מבנה קלט ופלט):
```python
def update_input_output_section(content: str, metadata: Dict) -> str:
    """עדכון סעיף 3 עם tensor names מאומתים"""
    # 1. מצא את הטבלה בסעיף 3
    # 2. עדכן שורות Input תמונה ו-Input מסכה עם השמות הנכונים
    # 3. עדכן שורת Output עם השם הנכון
    # 4. החלף "[README – לאמת ב-`probe`]" ב-"[אומת]"
    
    replacements = {
        f"Input תמונה | `rgb`": f"Input תמונה | `{metadata['image'].name}`",
        f"Input מסכה | `mask`": f"Input מסכה | `{metadata['mask'].name}`",
        f"Output | `inpainted`": f"Output | `{metadata['output'].name}`",
        "[README – לאמת ב-`probe`]": "[אומת]"
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content
```

#### update_markers(content: str, updates: Dict[str, str]) -> str
- מחליף סימוני סטטוס ספציפיים
- דוגמה: `"[pointer]"` → `"[אומת]"`
- שומר על סימונים אחרים: `[README]`, `[לא ידוע]`, `[ידע כללי]`

```python
def update_markers(content: str, marker_map: Dict[str, str]) -> str:
    """עדכון סימוני סטטוס מוגדרים בלבד"""
    for old_marker, new_marker in marker_map.items():
        # החלפה רק במקומות ספציפיים (לא כל המופעים)
        content = content.replace(old_marker, new_marker)
    
    return content
```

#### add_verification_note(content: str, timestamp: str) -> str
- מוסיף הערה בראש המסמך (אחרי הכותרת)
- פורמט: `> אומת ב-{timestamp}`

```python
from datetime import datetime

def add_verification_note(content: str) -> str:
    """הוספת הערת אימות בראש המסמך"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    note = f"> אומת בשלב 0 ב-{timestamp}\n\n"
    
    # מצא את סוף הכותרת הראשית והוסף אחריה
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('# MODEL_CARD'):
            # הוסף אחרי שורות ה-blockquote הקיימות
            insert_pos = i + 1
            while insert_pos < len(lines) and lines[insert_pos].startswith('>'):
                insert_pos += 1
            lines.insert(insert_pos, note.rstrip())
            break
    
    return '\n'.join(lines)
```

#### update_task_list(content: str, completed_tasks: List[str]) -> str
- מזהה את סעיף 8 (משימות להשלמת הכרטיס)
- משנה `- [ ]` ל-`- [x]` עבור משימות שהושלמו

```python
def update_task_list(content: str, completed_tasks: List[str]) -> str:
    """סימון משימות מושלמות ברשימה"""
    lines = content.split('\n')
    in_section_8 = False
    
    for i, line in enumerate(lines):
        if line.startswith('## 8.'):
            in_section_8 = True
        elif in_section_8 and line.startswith('##'):
            break
        elif in_section_8 and line.strip().startswith('- [ ]'):
            # בדוק אם המשימה ברשימת המושלמות
            for task in completed_tasks:
                if task in line:
                    lines[i] = line.replace('- [ ]', '- [x]')
    
    return '\n'.join(lines)
```

#### write_model_card(path: Path, content: str) -> None
- כותב את התוכן המעודכן חזרה לקובץ
- משתמש ב-UTF-8 encoding

```python
def write_model_card(path: Path, content: str) -> None:
    """כתיבת MODEL_CARD מעודכן"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
```

### 3.4 ReportGenerator

**אחריות**: יצירת דוחות ותיעוד התהליך

#### log_step(step_name: str, status: str, details: Any) -> None
- מדפיס לקונסול את התקדמות כל שלב
- פורמט: `[STEP] step_name: status`

```python
def log_step(step_name: str, status: str, details: Any = None) -> None:
    """תיעוד התקדמות שלב"""
    print(f"[{step_name.upper()}] {status}")
    if details:
        print(f"  Details: {details}")
```

#### generate_summary(report: ValidationReport) -> str
- יוצר סיכום מפורט של כל התהליך
- כולל: סטטוס כל שלב, נתונים שהתגלו, שגיאות (אם היו)

```python
@dataclass
class ValidationReport:
    steps: List[Tuple[str, str, Any]]
    
    def add_step(self, name: str, status: str, details: Any = None):
        self.steps.append((name, status, details))
    
    def is_success(self) -> bool:
        return all(status == "success" for _, status, _ in self.steps)

def generate_summary(report: ValidationReport) -> str:
    """יצירת סיכום התהליך"""
    lines = ["=" * 60, "Stage 0 Validation Summary", "=" * 60, ""]
    
    for name, status, details in report.steps:
        symbol = "✓" if status == "success" else "✗"
        lines.append(f"{symbol} {name}: {status}")
        if details:
            lines.append(f"  → {details}")
        lines.append("")
    
    overall = "SUCCESS" if report.is_success() else "FAILED"
    lines.append(f"Overall: {overall}")
    lines.append("=" * 60)
    
    return "\n".join(lines)
```

#### format_error(exception: Exception, context: str) -> str
- מעצב הודעת שגיאה קריאה עם הקשר
- כולל: סוג השגיאה, הודעה, stack trace (אם רלוונטי)

## 4. תרחישי זרימה מפורטים

### 4.1 זרימה מוצלחת (Happy Path)

```
1. התחלה
   ↓
2. חישוב SHA256 של models/big-lama/inpainter.onnx
   → תוצאה: ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff
   → לוג: [FILE_VALIDATION] SHA256 verified: match
   ↓
3. בדיקת גודל קובץ
   → תוצאה: 208,851,664 bytes
   → לוג: [FILE_VALIDATION] File size verified: 208851664 bytes
   ↓
4. בדיקת LFS pointer
   → תוצאה: false (זה קובץ בינארי רגיל)
   → לוג: [FILE_VALIDATION] Not an LFS pointer
   ↓
5. טעינת ONNX session
   → תוצאה: session object
   → לוג: [MODEL_LOADING] Session created successfully
   ↓
6. חילוץ metadata של inputs
   → תוצאה: 2 tensors
     - tensor 1: name="image", shape=[1,3,'height','width'], dtype="tensor(float)"
     - tensor 2: name="mask", shape=[1,1,'height','width'], dtype="tensor(float)"
   → לוג: [METADATA] Extracted 2 input tensors
   ↓
7. זיהוי תפקידים
   → תוצאה: {"image": tensor1, "mask": tensor2}
   → לוג: [METADATA] Identified tensor roles
   ↓
8. חילוץ metadata של outputs
   → תוצאה: 1 tensor
     - name="output", shape=[1,3,'height','width'], dtype="tensor(float)"
   → לוג: [METADATA] Extracted 1 output tensor
   ↓
9. אימות אילוצי מימדים
   → בדיקה: האם יש אילוצים במטא-דאטה?
   → תוצאה: לא נמצאו אילוצים מפורשים
   → לוג: [METADATA] No explicit dimension constraints found
   ↓
10. עדכון MODEL_CARD.md
    → פעולות:
      a. הוספת הערת אימות עם timestamp
      b. עדכון סעיף 1: "[pointer]" → "[אומת]" (SHA256 וגודל)
      c. עדכון סעיף 3: שמות הטנסורים המאומתים
      d. החלפת "[README – לאמת ב-`probe`]" → "[אומת]"
      e. עדכון סעיף 8: סימון משימות מושלמות
    → לוג: [MODEL_CARD] Updated successfully
   ↓
11. יצירת דוח סיכום
    → הדפסת כל השלבים ותוצאותיהם
   ↓
12. סיום מוצלח
```

### 4.2 זרימה עם שגיאת SHA256

```
1. התחלה
   ↓
2. חישוב SHA256
   → תוצאה: abc123... (לא תואם)
   → לוג: [FILE_VALIDATION] ERROR: SHA256 mismatch
   → פרטים:
     Expected: ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff
     Got: abc123...
   ↓
3. עצירת תהליך
   → דוח שגיאה: "SHA256 hash does not match. File may be corrupted or incorrect."
   → EXIT CODE: 1
```

### 4.3 זרימה עם LFS pointer

```
1-3. [שלבים מוצלחים]
   ↓
4. בדיקת LFS pointer
   → תוצאה: true (זוהה marker של LFS)
   → לוג: [FILE_VALIDATION] ERROR: File is a Git LFS pointer
   → פרטים: "Run 'git lfs pull' to download the actual model file"
   ↓
5. עצירת תהליך
   → EXIT CODE: 2
```

### 4.4 זרימה עם כישלון טעינת מודל

```
1-4. [שלבים מוצלחים]
   ↓
5. טעינת ONNX session
   → חריגה: ONNXRuntimeError
   → לוג: [MODEL_LOADING] ERROR: Failed to load ONNX session
   → פרטים: stack trace + הודעת השגיאה
   ↓
6. עצירת תהליך
   → EXIT CODE: 3
```

## 5. טיפול בשגיאות ואילוצים

### 5.1 מדיניות שגיאות

| שלב | סוג שגיאה | טיפול | קוד יציאה |
|------|-----------|--------|----------|
| SHA256 | אי-התאמה | דיווח + עצירה | 1 |
| File size | אי-התאמה | דיווח + עצירה | 1 |
| LFS check | זוהה pointer | דיווח + הנחיה (git lfs pull) | 2 |
| Session load | חריגה ONNX | דיווח + stack trace | 3 |
| Metadata extraction | חוסר נתונים | דיווח + פרטים | 4 |
| Tensor identification | לא נמצאו 2 inputs | דיווח + מה שנמצא | 5 |
| MODEL_CARD update | שגיאת כתיבה | דיווח + permissions | 6 |

### 5.2 אילוצי קלט

1. **נתיב המודל**: `models/big-lama/inpainter.onnx` (קבוע)
2. **נתיב MODEL_CARD**: `docs/architecture/cleanup-ai/MODEL_CARD.md` (קבוע)
3. **SHA256 צפוי**: `ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff`
4. **גודל צפוי**: 208,851,664 bytes

### 5.3 אילוצי פלט

1. **MODEL_CARD מעודכן**: חייב לשמר מבנה, שפה (עברית), וכותרות
2. **סימוני סטטוס**: רק `[pointer]` ו-`[README – לאמת ב-probe]` משתנים ל-`[אומת]`
3. **משימות בסעיף 8**: רק משימות שהושלמו בפועל מסומנות

## 6. שיקולי מימוש

### 6.1 ניהול זיכרון

- קריאת הקובץ לחישוב SHA256 בבלוקים של 8KB
- טעינת MODEL_CARD כולו לזיכרון (גודל: ~5KB, זניח)
- שחרור session אחרי חילוץ metadata

### 6.2 עמידות (Robustness)

- בדיקת קיום קבצים לפני פעולה
- validation של מבנה MODEL_CARD לפני עדכון
- גיבוי אוטומטי של MODEL_CARD לפני שינוי (`.bak`)

```python
def backup_model_card(path: Path) -> Path:
    """יצירת גיבוי לפני עדכון"""
    backup_path = path.with_suffix('.md.bak')
    shutil.copy2(path, backup_path)
    return backup_path
```

### 6.3 תאימות פלטפורמות

- שימוש ב-`Path` מ-pathlib לנתיבים
- encoding מפורש (UTF-8) לקבצים
- תמיכה בנתיבים עם עברית (Windows)

### 6.4 בדיקות (Testing Strategy)

ראו סעיף 7 (Correctness Properties) לפירוט מלא של תכונות הנבדקות.

**רמות בדיקה**:
1. **Unit tests**: פונקציות בודדות (SHA256, LFS check, marker replacement)
2. **Integration tests**: תהליך מלא עם מודל אמיתי
3. **Property tests**: תכונות אוניברסליות (שמירת מבנה, ניהול סימונים)

## 7. Correctness Properties

*תכונה (property) היא מאפיין או התנהגות שצריכים להתקיים בכל הרצות תקינות של המערכת - למעשה, הצהרה פורמלית על מה שהמערכת צריכה לעשות. תכונות משמשות כגשר בין מפרטים קריאים לבני אדם לבין ערובות תקינות הניתנות לאימות מכני.*

### Property 1: זיהוי תפקידי טנסורים לפי צורה

*עבור כל* זוג טנסורים קלט שאחד בצורה `[1, 3, H, W]` והשני בצורה `[1, 1, H, W]`, המערכת תזהה נכון איזה הוא tensor התמונה (3 ערוצים) ואיזה tensor המסכה (ערוץ אחד).

**Validates: Requirements 4.6**

### Property 2: אימות תבנית פלט

*עבור כל* tensor פלט מהמודל, המערכת תאמת שהוא תואם לתבנית `[1, 3, H, W]` כאשר H ו-W הם מימדים תקינים (מספרים שלמים חיוביים או סימונים דינמיים).

**Validates: Requirements 5.6**

### Property 3: אימות אילוצי מימדים ככפולות של 8

*עבור כל* ערכי מימדים (H, W) שמתגלים באילוצי המטא-דאטה, המערכת תאמת שהם מתועדים ככפולות של 8.

**Validates: Requirements 6.2**

### Property 4: שמירת מבנה MODEL_CARD

*עבור כל* עדכון של MODEL_CARD, המערכת תשמר את כל הכותרות, הטבלאות והמבנה המקורי של המסמך - רק ערכי הנתונים וסימוני הסטטוס ישתנו.

**Validates: Requirements 8.1, 8.2, 8.5**

### Property 5: ניהול עקבי של סימוני סטטוס

*עבור כל* עדכון של MODEL_CARD, המערכת תחליף רק את הסימונים `[pointer]` ו-`[README – לאמת ב-probe]` ב-`[אומת]`, ותשמר על כל הסימונים האחרים (`[README]`, `[לא ידוע]`, `[ידע כללי]`) ללא שינוי.

**Validates: Requirements 7.2, 7.3, 8.4**

### Property 6: שמירת תוכן בעברית

*עבור כל* MODEL_CARD המכיל תוכן בעברית, המערכת תשמר את כל התווים העבריים ללא שיבוש או מחיקה לאחר העדכון.

**Validates: Requirements 8.3**

### Property 7: עדכון רשימת משימות

*עבור כל* רשימת משימות בסעיף 8 של MODEL_CARD, המערכת תסמן נכון את המשימות שהושלמו (`- [ ]` → `- [x]`) בלבד, מבלי לשנות משימות שטרם הושלמו.

**Validates: Requirements 7.7**

### Property 8: דיווח מקיף על שלבי האימות

*עבור כל* הרצת תהליך האימות, המערכת תדווח על הצלחה או כישלון של כל שלב, וכאשר שלב נכשל, הדיווח יכלול את שם השלב ואת הסיבה המדויקת לכישלון.

**Validates: Requirements 10.1, 10.6**

## 8. דוגמת ביצוע (Example Run)

```bash
$ python validate_stage0.py

[FILE_VALIDATION] Calculating SHA256...
[FILE_VALIDATION] SHA256 verified: match
[FILE_VALIDATION] File size verified: 208851664 bytes
[FILE_VALIDATION] Checking for LFS pointer...
[FILE_VALIDATION] Not an LFS pointer

[MODEL_LOADING] Loading ONNX session...
[MODEL_LOADING] Session created successfully
[MODEL_LOADING] ONNX Runtime version: 1.30.0
[MODEL_LOADING] Provider: CPUExecutionProvider

[METADATA] Extracting input tensors...
[METADATA] Found 2 input tensors:
  - name: image, shape: [1, 3, 'height', 'width'], type: tensor(float)
  - name: mask, shape: [1, 1, 'height', 'width'], type: tensor(float)
[METADATA] Identified roles: image=image, mask=mask

[METADATA] Extracting output tensors...
[METADATA] Found 1 output tensor:
  - name: output, shape: [1, 3, 'height', 'width'], type: tensor(float)

[METADATA] Checking dimension constraints...
[METADATA] No explicit dimension constraints found in metadata
[METADATA] Recording as "[לא ידוע - יש לאמת בבדיקה]"

[MODEL_CARD] Creating backup...
[MODEL_CARD] Backup saved: MODEL_CARD.md.bak
[MODEL_CARD] Updating section 1 (source and license)...
[MODEL_CARD] Updating section 3 (input/output structure)...
[MODEL_CARD] Updating section 8 (task list)...
[MODEL_CARD] Adding verification note...
[MODEL_CARD] Writing updated MODEL_CARD...
[MODEL_CARD] Update complete

============================================================
Stage 0 Validation Summary
============================================================

✓ file_validation: success
  → SHA256: ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff
  → Size: 208851664 bytes
  → LFS: No

✓ session_creation: success
  → Runtime: 1.30.0
  → Provider: CPUExecutionProvider

✓ metadata_extraction: success
  → Inputs: 2 (image, mask)
  → Outputs: 1 (output)
  → Dimension constraints: None found

✓ model_card_update: success
  → Updated sections: 1, 3, 8
  → Markers replaced: 5
  → Tasks completed: 2/8

Overall: SUCCESS
============================================================
```

## 9. נספחים

### 9.1 קבועים ופרמטרים

```python
# config.py
from pathlib import Path

# נתיבים
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = WORKSPACE_ROOT / "models" / "big-lama" / "inpainter.onnx"
MODEL_CARD_PATH = WORKSPACE_ROOT / "docs" / "architecture" / "cleanup-ai" / "MODEL_CARD.md"

# ערכים צפויים
EXPECTED_SHA256 = "ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff"
EXPECTED_FILE_SIZE = 208851664  # bytes

# הגדרות
LFS_MARKER = b"version https://git-lfs.github.com"
LFS_CHECK_BYTES = 200
SHA256_BLOCK_SIZE = 8192

# סימוני סטטוס
MARKER_VERIFIED = "[אומת]"
MARKER_POINTER = "[pointer]"
MARKER_TO_VERIFY = "[README – לאמת ב-`probe`]"
MARKER_README = "[README]"
MARKER_UNKNOWN = "[לא ידוע]"
```

### 9.2 תלויות (Dependencies)

```txt
# requirements-stage0.txt
onnxruntime>=1.30.0
```

הערה: Python standard library מספקת את כל השאר (hashlib, pathlib, dataclasses, datetime).

### 9.3 מבנה הקבצים המוצע

```
.
├── validate_stage0.py          # נקודת הכניסה הראשית
├── config.py                   # קבועים והגדרות
├── validators/
│   ├── __init__.py
│   ├── file_validator.py      # FileValidator
│   ├── model_inspector.py     # ModelInspector
│   └── model_card_updater.py  # ModelCardUpdater
├── utils/
│   ├── __init__.py
│   └── report.py              # ReportGenerator + ValidationReport
└── tests/
    ├── test_file_validator.py
    ├── test_model_inspector.py
    ├── test_model_card_updater.py
    └── test_integration.py
```

### 9.4 עדכונים צפויים ב-MODEL_CARD.md

**לפני**:
```markdown
# MODEL_CARD – Big LaMa (ONNX)

> עודכן: 2026-11-10.
> **קובץ ה-ONNX עצמו טרם נבדק**...

## 1. מקור ורישיון
| Revision / Commit ב-HF | — | [לא ידוע] |

## 3. מבנה קלט ופלט [README – לאמת ב-`probe`]
| Input תמונה | `rgb`, `[1, 3, H, W]` ... | [README – לאמת ב-`probe`] |

## 8. משימות להשלמת הכרטיס
- [ ] להשיג את תוכן הקובץ ולאמת SHA256...
- [ ] `probe`: אימות שמות, טיפוסים ומידות...
```

**אחרי**:
```markdown
# MODEL_CARD – Big LaMa (ONNX)

> עודכן: 2026-11-10.
> אומת בשלב 0 ב-2026-11-15 14:30
> **טרם בוצע inference מלא על תמונות**...

## 1. מקור ורישיון
| גודל קובץ | 208,851,664 בתים | [אומת] |
| SHA256 | ef85edd... | [אומת] |

## 3. מבנה קלט ופלט [אומת]
| Input תמונה | `image`, `[1, 3, H, W]` ... | [אומת] |
| Input מסכה | `mask`, `[1, 1, H, W]` ... | [אומת] |

## 8. משימות להשלמת הכרטיס
- [x] להשיג את תוכן הקובץ ולאמת SHA256...
- [x] `probe`: אימות שמות, טיפוסים ומידות...
```

## 10. נקודות החלטה והנחות

### 10.1 נקודות שתועדו כ-"לא ידוע"

אם המטא-דאטה של המודל לא מכילה אילוצים מפורשים על H ו-W (min/max), הם יתועדו כ-`[לא ידוע - יש לאמת בבדיקה]`. זה תקין - אימות המימדים יתבצע בשלבים מאוחרים יותר עם inference בפועל.

### 10.2 העדפות טכנולוגיות

- **Python standard library** מועדפת על ספריות חיצוניות
- **onnxruntime CPU** בלבד (ללא GPU providers)
- **עדכון in-place** של MODEL_CARD (לא יצירת קובץ חדש)

### 10.3 החלטות עיצוב

1. **אין סקריפט חדש בתיקיית tools/**: המימוש יהיה בתיקייה נפרדת או כחלק מסקריפט זמני שלא יישמר
2. **גיבוי אוטומטי**: לפני כל עדכון של MODEL_CARD, המערכת תיצור `.bak`
3. **קודי יציאה**: כל סוג שגיאה מקבל קוד יציאה ייחודי לאבחון קל

## 11. הערות מימוש

### 11.1 שפה לבדיקות

הבדיקות (unit tests, property tests) יכתבו באנגלית לעקביות עם הקוד, אבל הערות והמסמכים יכולים להיות בעברית כאן.

### 11.2 סדר עדיפות בפיתוח

1. **שלב 1**: מימוש FileValidator (SHA256, LFS, size)
2. **שלב 2**: מימוש ModelInspector (session, metadata)
3. **שלב 3**: מימוש ModelCardUpdater (parsing, updates)
4. **שלב 4**: אינטגרציה ודוחות
5. **שלב 5**: בדיקות property-based

### 11.3 שימוש חוזר

חלקים מ-`tools/poc/probe_and_run.py` יכולים לשמש כבסיס:
- `load_session()` - כבסיס ל-ModelInspector.load_session()
- `describe()` - כהשראה לחילוץ metadata
- לוגיקת זיהוי tensors מתוך `pick_inputs()`

אבל יש לבצע refactoring משמעותי להפרדת אחריות ולהוספת validation מוקפד.
