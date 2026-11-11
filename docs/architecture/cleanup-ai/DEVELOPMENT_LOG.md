# DEVELOPMENT_LOG – יומן פיתוח

> יומן כרונולוגי. אין למחוק רשומות קודמות; מוסיפים רשומות חדשות בסוף.

---

## 2026-11-10 – שלב 0/3/4/5 (תיעוד ותכנון; קליטת המודל נחסמה)

**שלב:** תכנון, ארכיטקטורה, מפת דרכים, שלד PoC. שלבים 1–2 (קליטת מודל ומחקר) טרם בוצעו.

**מה בוצע**
- שוכפל הריפו `lodbigy-770/FIXPIC` – היה ריק.
- נוצרו מסמכי התכנון בתיקייה `docs/architecture/cleanup-ai/`.
- נכתב סקריפט PoC גנרי שקורא את מבנה ה-Input/Output מקובץ ה-ONNX עצמו.
- נבדק שהסקריפט עובד (הדפסת מטא-דאטה, הרצה על CPU, מיזוג ובדיקת פיקסלים מחוץ למסכה) על מודל דמה פשוט.

**קבצים שנוספו**
- `docs/architecture/cleanup-ai/PLAN.md`
- `docs/architecture/cleanup-ai/ARCHITECTURE.md`
- `docs/architecture/cleanup-ai/ROADMAP.md`
- `docs/architecture/cleanup-ai/MODEL_CARD.md` (שלד; רוב השדות מסומנים "לא ידוע")
- `docs/architecture/cleanup-ai/DEVELOPMENT_LOG.md`
- `tools/poc/probe_and_run.py`

**החלטות טכניות**
- ADR-001: ONNX Runtime CPU כמריץ ראשי. ADR-002: C#/WPF לממשק, מנוע לפי מדידות ב-M1.5 (זמני). ADR-003: עיבוד ROI. ADR-004: Undo כ-Patch. ADR-005: שמות Tensors ופרמטרי קלט נקראים מהמודל ולא מקודדים בקוד.
- נוסף שלב ביניים M1.5 להכרעת Runtime/Stack על בסיס מדידות.

**בדיקות ותוצאות**
- `py_compile` על `probe_and_run.py`: עבר.
- הרצה על מודל דמה (`image*(1-mask)+0.5*mask`, נוצר בנפרד ולא נשמר בריפו): נטען, רץ ב-CPU, פיקסלים מחוץ למסכה זהים למקור. **זו בדיקת צנרת בלבד ואינה בדיקה של Big LaMa.**
- סביבת הבדיקה: onnxruntime 1.31.0, Python 3.12, ליבה אחת, ~3GB RAM.

**בעיות שהתגלו**
- אין גישה ל-`huggingface.co` ול-`cdn-lfs.huggingface.co` מסביבת העבודה (חסימת רשת). לא ניתן להוריד את המודל, לחשב SHA256 או להריץ PoC אמיתי.
- `git lfs` אינו מותקן בסביבה (ייבדק אם נדרש כשהמודל יוכנס).
- הטוקן של GitHub נחשף בצ'אט; מומלץ לבטל וליצור חדש.

**מה לא בוצע (במכוון)**
- `models/big-lama/` ו-README שלו – ייווצרו אחרי קבלת הקבצים האמיתיים.
- אפיון Tensors, Padding, נרמול, ביצועים – ממתינים לקובץ.

**משימות להמשך**
1. המשתמש מוריד את המודל ומעלה אותו (או פותח את הדומיינים של HF ברשת).
2. שלב 1: זיהוי קובץ ה-ONNX, Revision, גדלים, SHA256, רישיון, בדיקת LFS pointer.
3. שלב 2: `probe` על הקובץ האמיתי ועדכון `MODEL_CARD.md` (אומת/הנחה).
4. הרצת PoC על תמונות אמיתיות ומדידת זמן/זיכרון.

---

## 2026-11-10 – שיפור התוכנית: פריטיות מול Cleanup.pictures

**שלב:** תכנון (ללא קוד מוצר חדש).

**הרקע:** המשתמש הבהיר שהמוצר צריך להיות העתק של cleanup.pictures במצב Offline גמור, ולכן נבחר Big LaMa.

**מה בוצע**
- נבדק דף הבית הרשמי של האתר וסקירות צד ג'. הדף הרשמי שיווקי בלבד (חתימת 2022) ואינו מתאר את העורך; התנהגות העורך נלקחה מסקירות ומסומנת [דווח]. העורך עצמו לא נבדק.
- נכתב `PARITY-SPEC.md`: מטריצת פריטיות P0/P1/P2/OUT, לולאת החוויה, יתרונות ופערים, נושאי מחקר, שאלות פתוחות, עניינים משפטיים.
- `PLAN.md`: חזון מעודכן, דרישות עם עדיפויות, דרישות Latency/i18n/Offline, סיכונים חדשים, קריטריוני הצלחה מבוססי פריטיות.
- `ARCHITECTURE.md`: נוספו סעיפים 4א (לולאת עריכה, Job Scheduler, Warm-up) ו-4ב (רזולוציה מקורית תחילה); ADR-006 עד ADR-009; הרחבת מסכה כברירת מחדל.
- `ROADMAP.md`: M1.5 הורחב ל-Latency Spike; M2/M3/M5 עודכנו (תור משימות, לולאת משיכה, בדיקת Offline, ביקורת רישוי).

**קבצים שהשתנו:** `PLAN.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `DEVELOPMENT_LOG.md`; נוסף `PARITY-SPEC.md`.

**החלטות טכניות:** ADR-006 (עריכה מצטברת לפי משיכה), ADR-007 (WPF ולא WebView2), ADR-008 (Dilate כברירת מחדל), ADR-009 (Session חם ויחיד).

**בדיקות ותוצאות:** אין בדיקות קוד בשלב זה. נבדק שכל ההחלפות בקבצי התיעוד הוחלו (patch עם assert).

**בעיות שהתגלו:** הפרמטר הקריטי לחוויה (Latency למשיכה ב-CPU) עדיין לא ידוע; תלוי בהורדת המודל.

**משימות להמשך:** אימות ידני של פריטי [דווח] ע"י המשתמש; הכרעת שאלות §6 ב-`PARITY-SPEC.md`; הורדת המודל; ביצוע שלבים 1–2 ואז Latency Spike.

---

## 2026-11-10 – הכרעות המשתמש: עברית בלבד, כלי עצמאי

**שלב:** תכנון (ללא קוד).

**הכרעות:** (1) ממשק בעברית בלבד, סומן "אולי" ולכן ראשוני. (2) כלי Windows עצמאי, לא דרך הדפדפן.

**מה בוצע:** ADR-007 נסגר סופית (WPF, ללא WebView2/Electron, פריסה Self-contained). נוספה דרישת RTL. נוספה דרישה ובדיקה לנתיבי קבצים בעברית (סיכון ידוע עם `cv2.imread` ב-Windows; לאמת). `PARITY-SPEC.md` §6 עודכן; נשארו פתוחות: מצב השוואה, תקציב Latency, Batch.

**קבצים שהשתנו:** `PARITY-SPEC.md`, `PLAN.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `DEVELOPMENT_LOG.md`.

**בדיקות:** אין בדיקות קוד. כל ההחלפות בקבצי התיעוד הוחלו (patch עם assert).

**משימות להמשך:** הורדת המודל; שלבים 1–2; Latency Spike.

---

## 2026-11-10 – מעבר ל-Python (ADR-002/007 עודכנו)

**שלב:** תכנון (ללא קוד מוצר).

**הכרעות המשתמש:** השוואה לפני/אחרי גם Hold וגם Toggle; Batch לא נדרש; העדפה ל-Python או Node אם הביצועים לא נפגעים.

**החלטה טכנית:** Python 3.12 + PySide6 + onnxruntime + OpenCV/NumPy במקום C#/WPF. הנימוק: זמן ה-Inference נקבע ב-ONNX Runtime ולא בשפה; ה-PoC כבר ב-Python. Node נדחה (Electron = דפדפן ארוז, ואין חלופה בוגרת ל-OpenCV). ADR-002 ו-ADR-007 נכתבו מחדש. ההחלטה בכפוף למדידות M1.5 (זמן Pre/Post, זמן עלייה, שחרור GIL, Zoom/Pan על 24MP).

**מחיר שתועד:** אריזה גדולה יותר, זמן עלייה איטי, חשד אנטי-וירוס ל-exe ארוז, רישוי LGPL של PySide6 (לבדוק משפטית).

**קבצים שהשתנו:** `ARCHITECTURE.md` (כולל סעיף 13: ספריות ומבנה קוד), `PLAN.md`, `ROADMAP.md`, `PARITY-SPEC.md`, `MODEL_CARD.md` (שם הממשק `InpaintModel`), `DEVELOPMENT_LOG.md`.

**בדיקות:** אין בדיקות קוד. כל ההחלפות בקבצי התיעוד הוחלו (patch עם assert).

**נשאר פתוח:** תקציב Latency (אחרי מדידה). **הבא:** הורדת המודל, שלבים 1–2, Latency Spike.

---

## 2026-11-10 – קבלת המודל לריפו (כ-LFS pointer בלבד; התוכן עדיין לא נגיש)

**שלב:** שלב 1 (קליטת מודל) – בוצע חלקית.

**מה נמצא:** המשתמש דחף לריפו את הקומיט `8cccc36` ("Add Big LaMa ONNX model") עם הקובץ `models/big-lama/inpainter.onnx` ו-`.gitattributes` (`models/big-lama/*.onnx` דרך Git LFS).

**נתונים מה-pointer (לא אומתו מול הקובץ עצמו):**
- `oid sha256:ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff`
- `size 208851664` בתים (כ-199MB)

**בעיה:** בסביבת העבודה הורד רק ה-pointer (134 בתים). התקנתי `git-lfs` (3.4.1), אך `git lfs pull` נכשל: הרשת חוסמת את `github-cloud.githubusercontent.com` (שם מאוחסן אובייקט ה-LFS), וגם את `media.githubusercontent.com`. `raw.githubusercontent.com` מחזיר את ה-pointer ולא את התוכן. לכן לא בוצעו: אימות SHA256 של הקובץ האמיתי, טעינה ב-ONNX Runtime, `probe`, ו-PoC.

**לא בוצע במכוון:** `models/big-lama/README.md` ו-`MODEL_CARD.md` לא עודכנו, כי אין עדיין נתונים מאומתים (Revision של HF, שמות Tensors, רישיון).

**משימות להמשך:**
1. לספק את תוכן הקובץ לסביבה (פתיחת הדומיינים `github-cloud.githubusercontent.com` ו-`media.githubusercontent.com`, או העלאה כ-GitHub Release asset).
2. לאמת SHA256 מול ה-pointer ואת הגודל.
3. `probe_and_run.py probe` על הקובץ, ואז עדכון `MODEL_CARD.md`.
4. תיעוד Revision/רישיון מ-Hugging Face (דורש גישה לריפו ב-HF, או שהמשתמש יספק).

---

## 2026-11-10 – קליטת מסמכי המודל (README ו-LICENSE מ-HF)

**שלב:** שלבים 1–2 – חלקי; תוכן קובץ ה-ONNX עדיין לא נגיש.

**מה בוצע**
- התקבלו `LICENSE` ו-`README` מריפו ה-HF. נבדק: `LICENSE` הוא Apache-2.0 עם "Copyright [2021] Samsung Research". הקבצים הועתקו ל-`models/big-lama/` (`LICENSE`, `UPSTREAM_README.md`).
- נכתב `models/big-lama/README.md` (מקור, קבצים, גדלים, SHA256, רישיון, הפעלה). ערכי `inpainter.onnx` (גודל ו-SHA256) הם מה-pointer בלבד ומסומנים כך.
- `MODEL_CARD.md` נכתב מחדש על סמך ה-README, עם מקרא ברור: [README]/[pointer]/[אומת]. שום נתון עדיין אינו [אומת] מול הקובץ.
- ממצאים מרכזיים מה-README: קלט `rgb` ו-`mask` (1 = להסרה), פלט `inpainted`; **צורה דינמית** (H,W כפולות של 8, בין 64 ל-4096); ריפוד symmetric; סף מסכה `> 0`; ONNX Runtime 1.30+; FFT כמכפלות מטריצות; דיווח ביצועים: 4.6 שניות ל-1536² על 16 threads.

**החלטות טכניות:** ADR-010 (קלט דינמי: Padding לכפולות של 8, ROI בגודל מקורי עד תקרה, הקשר מתכוונן). זמני עד אימות.

**שינויים ב-PoC (`tools/poc/probe_and_run.py`):** ריפוד השתנה מ-`BORDER_REFLECT_101` ל-`BORDER_REFLECT` (symmetric, כפי ב-LaMa); סף המסכה הפך לפרמטר עם ברירת מחדל `> 0` (במקום `> 127`). נבדק מחדש על מודל הדמה בלבד: עבר.

**בעיות שנותרו:** אין גישה לתוכן ה-LFS. Revision ב-HF לא ידוע. תנאי נתוני האימון לא נבדקו.

**קבצים שהשתנו:** `models/big-lama/README.md`, `LICENSE`, `UPSTREAM_README.md` (חדשים); `MODEL_CARD.md`, `ARCHITECTURE.md`, `PLAN.md`, `DEVELOPMENT_LOG.md`, `tools/poc/probe_and_run.py`.

**משימות להמשך:** להשיג את תוכן הקובץ; SHA256; `probe`; PoC; Latency Spike.

---

## 2026-11-10 – M1: כלי בדיקה להרצה מקומית (המודל לא נגיש לסביבת העבודה)

**שלב:** Milestone 1 (Model Integration) – הכנת כלי; ההרצה על המודל האמיתי נעשית אצל המשתמש.

**הכרעת משתמש:** הפרויקט לשימוש אישי ולא מסחרי. נרשם ב-`PLAN.md` §8; בדיקה משפטית מסחרית אינה נדרשת כעת.

**מה בוצע**
- נבדק שוב: `git lfs pull` לא אפשרי מסביבת העבודה (הרשת חוסמת את שרת ה-LFS), לכן קובץ ה-ONNX נשאר pointer בריפו וההרצה עוברת למחשב המשתמש.
- נוסף `tools/m1/m1_check.py` עם `verify` (גודל ו-SHA256 מול ערכי ה-pointer, זיהוי pointer), `probe` (שמות, טיפוסים וצורות Tensors), `run` (קדם-עיבוד כמו predict של LaMa: symmetric padding לכפולות של 8, מסכה `> 0`, חיתוך הריפוד, המרה ל-8 ביט בקליפ וקטימה), ו-`all`. הבדיקות האוטומטיות: צורת פלט, NaN, טווח, פיקסלים מחוץ למסכה, שינוי בתוך המסכה. נמדדים זמן Cold/Warm וזיכרון שיא, ונכתב `m1_results.json` עם מפרט המחשב.
- נוסף `tools/m1/requirements.txt`, `M1-RUNBOOK.md` (שלבי ביצוע) והשורה `m1_out/` ב-`.gitignore`.

**בדיקות:** הכלי רץ מקצה לקצה על מודל דמה (קלט `rgb`+`mask`, פלט `inpainted`, צורה דינמית): 3 מקרים סינתטיים ותמונה בגודל לא כפולה של 8 (157x203) עברו. על קובץ ה-pointer האמיתי `verify` נכשל כמצופה. **לא נבדק על Big LaMa.**

**הערות:** הקבצים `PROGRESS.md`, `UNIFY-PLAN.md`, `CHANGELOG.md` ו-`LAYOUT-SPEC.md` שהוזכרו בנוהל העבודה אינם קיימים בריפו. עודכנו במקומם `DEVELOPMENT_LOG.md`, `PLAN.md` §8 ו-`ROADMAP.md`.

**קבצים שהשתנו:** `tools/m1/m1_check.py`, `tools/m1/requirements.txt`, `docs/architecture/cleanup-ai/M1-RUNBOOK.md` (חדשים); `PLAN.md`, `ROADMAP.md`, `DEVELOPMENT_LOG.md`, `.gitignore`.

**הבא:** המשתמש מריץ לפי `M1-RUNBOOK.md` ושולח את התוצאות; לאחר מכן השלמת `MODEL_CARD.md` וסימון M1, ואז M1.5 (Latency Spike).

---

## 2026-11-11 – סקירת קומיטים של KIRO (שלב 0) ועדכון תיעוד

**שלב:** שלב 0 (קליטת מודל ואפיון) – סקירה ותיעוד. לא נוסף קוד, והמודל לא הועתק לסביבת העבודה (בהנחיית המשתמש, שבודק אצלו).

**מה KIRO עשה**
- `9923eb3` (KIRO1): מפרט ב-`.kiro/specs/stage-0-model-validation/` (דרישות, עיצוב, משימות) והחבילה `stage0_validation/`: `FileValidator` (SHA256, גודל, זיהוי LFS pointer), `ModelInspector` (טעינת session, שמות/טיפוסים/צורות Tensors, זיהוי תפקידים), `ModelCardUpdater` (עדכון `MODEL_CARD.md`). נוספו גם `LICENSE (2)` ו-`README (3).md` בתיקיית המודל.
- `94d3064` (KIRO2): `ReportGenerator`, `ValidationOrchestrator`, נקודת הכניסה `validate_stage0.py` (קודי יציאה 0–6), `stage0_validation/README.md`, עדכון `MODEL_CARD.md` (וגיבוי `MODEL_CARD.md.bak`), וסימון כל משימות הליבה ב-`tasks.md` (1–11, 14).

**מה נבדק בסביבת Claude (על עותק זמני, בלי לשנות את הריפו)**
- 21 בדיקות עוברות. 9 נכשלות: 3 ב-`test_file_validator.py` ו-6 ב-`TestLoadSessionWithRealModel`. כולן דורשות את קובץ ה-ONNX האמיתי, והכישלון מצופה כי כאן הקובץ הוא pointer של 134 בתים.
- `python validate_stage0.py` על ה-pointer מסתיים בקוד 1 (אי-התאמת גודל), לא נוגע ב-`MODEL_CARD.md`, והדוח קריא.
- `_tmp_test_model_card.py` הוא סקריפט ולא קובץ pytest: הוא קורא ל-`sys.exit` בזמן import, ולכן `pytest` קורס עליו ב-INTERNALERROR.

**ממצאים**
1. **אין ראיה שמורה להרצה המקומית.** `MODEL_CARD.md.bak` מול `MODEL_CARD.md` מראים שהכלי כתב לכרטיס (סימון [אומת] בסעיפים 2 ו-3 ושתי משימות שסומנו `[x]`), והכלי כותב רק אחרי הצלחת כל שלבי האימות. אבל לא נשמר פלט הרצה, ולכן אי אפשר לשחזר ממנו את הנתונים (טיפוסים, מידות). לא שוחזר כאן.
2. **`update_section_3` החליף בפועל רק סימון, ושמות Tensors זהים לאלה שה-README טען** (`rgb`, `mask`, `inpainted`). זה עקבי עם README, אך לא מוכיח שמידות דינמיות נבדקו מעבר ל-`verify_dimension_constraints`.
3. **`MODEL_CARD.md` היה סותר את עצמו**: כותרת "אומת בשלב 0" לצד "טרם נבדק" ו-"לא אומתו". תוקן בשני מקומות, בלי למחוק את הסתייגות שסעיפים 4–6 עדיין [README].
4. **ל-KIRO לא נכתבה `test_model_card_updater.py`** (משימה 12.3, אופציונלית), וגם לא בדיקות property-based (משימה 13). במקומה קיים הסקריפט הזמני.
5. **קבצים מיותרים בריפו (לא נמחקו, ממתין לאישור):** `_tmp_test_model_card.py`, `docs/.../MODEL_CARD.md.bak`, `models/big-lama/LICENSE (2)` (שונה מ-`LICENSE` בגודל: 11,348 מול 11,549 בתים, כנראה סיומות שורה), `models/big-lama/README (3).md` (גרסה גולמית של README עם YAML, שונה מ-`UPSTREAM_README.md`).

**קבצים שהשתנו:** `ROADMAP.md` (סטטוס שלב 0 וסימון `[x]` בחמש משימות, פיצול משימת האפיון), `PLAN.md` §5, `MODEL_CARD.md` (שתי הערות מיושנות), `DEVELOPMENT_LOG.md`, וקובץ חדש `PROGRESS.md` (מסמך המשך).

**הערה:** `PROGRESS.md` נוצר עכשיו כי נוהל העבודה מפנה אליו. `UNIFY-PLAN.md` ו-`CHANGELOG.md` עדיין לא קיימים, והם לא נוצרו כי לא ידוע תוכנם המצופה.

**הבא:** המשתמש מאשר ששלב 0 עבר אצלו ומריץ M1 לפי `M1-RUNBOOK.md`; אחר כך משלימים `MODEL_CARD.md` סעיפים 4–6 ועוברים ל-M1.5.

---

## 2026-11-11 – אישור שלב 0 מפלט המשתמש; דחיית דוגמאות M1

**שלב:** שלב 0 – אושר. M1 – ממתין (לא הורץ).

**פלט שהמשתמש הדביק** (`.venv\Scripts\python.exe validate_stage0.py`, הורץ פעמיים, קוד יציאה 0 בשתיהן):
- SHA256 `ef85edd8…ba2ff` ו-208,851,664 בתים: תואמים. הקובץ אינו LFS pointer. ה-session נטען.
- Inputs: `rgb` `[1, 3, '8*height', '8*width']` ו-`mask` `[1, 1, '8*height', '8*width']`, שניהם `tensor(float)`. Output: `inpainted` `[1, 3, '8*height', '8*width']`, `tensor(float)`.
- אילוץ המידות: `multiple_of_8`; `h_min`, `h_max`, `w_min`, `w_max` חזרו `None`. **הטווח 64–4096 מה-README לא אומת.**
- `_tmp_test_model_card.py`: 29/31 בהרצה ראשונה (כשלו "Contains section 3 marker" ו-"[pointer] replaced", שניהם בגלל הבדיקה ולא הקוד), ו-31/31 אחרי תיקון הבדיקה אצל המשתמש. המקרא עדיין מכיל `[pointer]` במכוון.

**הערות על הכלי:** כל הרצה כותבת ל-`MODEL_CARD.md` מחדש ודורסת את `MODEL_CARD.md.bak`, ולכן אין להריץ שוב ללא צורך. בסיכום של ההרצה השנייה לא הופיעה שורת `model_card_write`, אף ש-`[MODEL_CARD WRITE] success` הופיע בלוג (ייתכן פגם קוסמטי בדוח; לא נבדק).

**החלטת משתמש:** המשתמש לא הצליח ליצור 3 תמונות דוגמה עם מסכות, ולכן ממשיכים בלעדיהן. הדוגמאות האמיתיות והדירוג הוויזואלי של M1 נדחים; ההרצה הסינתטית (`run --out m1_out`) אינה דורשת קבצי קלט.

**קבצים שהשתנו:** `MODEL_CARD.md` (סעיף 3: מה אומת ומה לא), `ROADMAP.md` (סטטוס שלב 0 ו-M1), `PROGRESS.md`, `DEVELOPMENT_LOG.md`.

**הבא:** המשתמש מריץ `verify`, `probe`, `run --out m1_out` ושולח פלט ו-`m1_results.json`.

---

## 2026-11-11 – M1 סינתטי על Big LaMa (אצל המשתמש) וכלי ציור מסכות

**שלב:** Milestone 1 – חלקי. נשארו דוגמאות אמיתיות, גודל שאינו כפולה של 8, ודירוג ויזואלי.

**פלט שהמשתמש הדביק** (Windows, `.venv`, onnxruntime 1.31.1, opencv 5.0.0.93, numpy 2.5.4):
- `verify`: גודל ו-SHA256 תואמים. `probe`: כל הבדיקות PASS, טעינה 2.51 שניות, `CPUExecutionProvider` בלבד. סדר ה-inputs ב-session הוא `['mask', 'rgb']`, והכלי משווה כקבוצה (שמות נקראים מהקובץ, בהתאם ל-ADR-005).
- `run` סינתטי (`synthetic_blob`, `synthetic_scratch`, `synthetic_text`, כולם 640×480): כל הבדיקות PASS בשלושתם: צורת פלט, ללא NaN/Inf, טווח 0..1, פלט גולמי מחוץ למסכה זהה למקור ב-0 רמות, ופיקסלים מחוץ למסכה זהים אחרי מיזוג. בתוך המסכה שונו 11287/11289, 4686/4691 ו-9060/9063 פיקסלים.
- זמנים: Cold 5.16 / 5.69 / 5.39 שניות; Warm 5.82 / 5.24 / 5.12 שניות. שיא זיכרון 946 / 961 / 961MB.

**הערות**
1. **הריפוד לא נבדק על המודל האמיתי.** 640 ו-480 הם כפולות של 8, ולכן נתיב ה-symmetric padding לא הופעל. הבדיקה על גודל כמו 157×203 נעשתה רק על מודל דמה.
2. **ה-Latency גבוה:** כ-5 שניות ל-640×480 (כ-0.3 מגה-פיקסל), בעוד שמחבר המודל דיווח 4.6 שניות ל-1536² (כ-2.4 מגה-פיקסל) על 16 threads. כנראה המחשב או מספר ה-threads שונים מהמדווח, אבל **מפרט המחשב לא ידוע** כי `m1_results.json` לא נשלח. זה קריטי לתקציב ה-Latency (ראו PLAN §3) ויידון ב-M1.5.
3. Warm לא מהיר מ-Cold (5.1–5.8 מול 5.2–5.7), כלומר אין עלות חימום משמעותית. הזמן הנמדד הוא `infer()` כולו (עיבוד מקדים, הרצה ועיבוד לאחר), ו-"Cold" הוא הקריאה הראשונה לכל דוגמה, ולכן רק בדוגמה הראשונה ה-session באמת קר. ב-M1.5 כדאי למדוד Inference בלבד בנפרד.
4. שיא זיכרון כ-950MB לתמונה קטנה: יש להביא בחשבון בתקציב הזיכרון.
5. ל-Git המקומי של המשתמש חסר קומיט אחד (`git pull`).

**מה נוסף:** `tools/m1/make_mask.py`, כלי ממשק (tkinter, בלי תלויות חדשות מעבר ל-`requirements.txt`) לציור מסכה על תמונה ושמירת זוג `NAME.png` + `NAME_mask.png` תואם ל-`m1_check.py run --dir`. נבדק כאן: בדיקות יחידה לליבה (נתיבים בעברית, ציור, מחיקה, הרחבה, שמירה) והפעלת הממשק מול Xvfb עם אירועי עכבר מדומים (ציור, מחיקה בימני, ביטול, שמירה, אזהרה על מסכה ריקה). **לא נבדק על Windows ועם עכבר אמיתי.**

**קבצים שהשתנו:** `tools/m1/make_mask.py` (חדש), `M1-RUNBOOK.md` (שלב 6), `ROADMAP.md`, `MODEL_CARD.md` (§6), `PROGRESS.md`, `DEVELOPMENT_LOG.md`.

**הבא:** `git pull`; לשלוח את `m1_out\m1_results.json`; ליצור 3 דוגמאות עם הכלי ולהריץ `run --dir samples`, כולל תמונה שמידותיה אינן כפולה של 8.
