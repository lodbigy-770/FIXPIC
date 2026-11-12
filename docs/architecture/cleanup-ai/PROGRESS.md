# PROGRESS – מצב הפרויקט והמשך עבודה

> עודכן: 2026-11-12. מסמך קצר להעברת הקשר לשיחה הבאה. הפירוט המלא ב-`DEVELOPMENT_LOG.md`, השלבים ב-`ROADMAP.md`.

## מצב נוכחי

| שלב | מצב |
|-----|-----|
| תיעוד ותכנון (PLAN, ARCHITECTURE, PARITY-SPEC) | בוצע |
| שלב 0 – קליטת מודל ואימות | **בוצע ואושר** ע"י פלט ההרצה של המשתמש (קוד יציאה 0). הטווח 64–4096 עדיין לא אומת |
| M1 – הרצת Big LaMa על CPU | **בוצע (2026-11-11).** 3 דוגמאות אמיתיות ו-3 סינתטיות עברו, הריפוד נבדק, הפלט מחוץ למסכה זהה למקור. איכות: אדם טוב, טקסט טוב מאוד, שריטה טובה. ממצא: 6.6–9.7 שניות ל-0.43MP על 4 ליבות, בלי תלות בגודל המסכה |
| M1.5 – Latency Spike | **מדידות בוצעו ונותחו; תקציב Latency ו-Python 3.14 הוכרעו (2026-11-12).** OpenVINO ו-3 threads נמדדו: OpenVINO CPU ב-3 threads 0.83/1.68/3.12 שניות (256/384/512) מול ORT ב-4 threads 0.84/1.97/3.90; ה-iGPU נפסל. צורה דינמית נמדדה (compile אחד 5.1 שניות לכל הגדלים, Latency דומה לסטטי). מטמון `CACHE_DIR` נמדד (compile חם כ-3 שניות לצורה במקום 6, כ-210MB דיסק לצורה). GIL וזמן עלייה נמדדו (ה-GIL משוחרר; אין צורך ברכיב נייטיב). נשאר: הכרעה על מטמון/דינמי, אריזה על 3.14, UI Spike, GIL וזמן עלייה, אריזה על 3.14. רץ בשיחה נפרדת (A) |
| M2 – Core Inpainting Engine | **M2.1–M2.5 בוצעו (2026-11-12):** (M2.4: קיבוץ מסכות, Tiling, Feathering; M2.5: תקרת זיכרון, התאוששות מ-OOM, דיווח התקדמות) (M2.3: `ModelManager` ו-`models/big-lama/model.json` עם sha256 מה-LFS pointer) `InpaintModel`, `OrtModel`, `OvModel` (OpenVINO ראשי, compile לכל bucket), `AutoModel` (fallback אוטומטי ל-ORT בכשל טעינה/compile/הרצה), `Pipeline.warmup`, ו-Pipeline עם ROI ל-buckets. 71 בדיקות עוברות עם מודל ONNX סינתטי (כולל תיקון אימות file ב-Windows, M2.5.1) (פריטיות OV/ORT ≤3). **לא נבדק מול Big LaMa האמיתי** (LFS pointer בסביבת Claude): `tests/test_real_model.py` רץ אצל המשתמש. נשאר: M2.6–M2.9 (פירוק ב-DEVELOPMENT_LOG) |
| M3–M5 | טרם התחיל |

## הגבלות סביבה (חשוב)

- Claude לא מצליח להוריד את `models/big-lama/inpainter.onnx`: הוא LFS pointer (134 בתים), והרשת חוסמת את `github-cloud.githubusercontent.com` ואת `media.githubusercontent.com`. כל מה שדורש את המודל האמיתי רץ אצל המשתמש.
- פתיחת הדומיין הראשון בהגדרות הרשת של Claude הייתה מאפשרת להריץ M1 בתוך השיחה, אך המשתמש בחר לבדוק מקומית.
- הטוקן לריפו נחשף בשתי שיחות (גם ב-2026-11-12). מומלץ להחליף אותו בדחיפות; הוא לא נשמר באף קובץ בריפו.

## החלטות שכבר התקבלו

- שפת ממשק: עברית בלבד (RTL). כלי עצמאי לשולחן העבודה, Python ו-PySide6.
- שימוש אישי ולא מסחרי.
- ONNX Runtime CPU בלבד, גרסה 1.30 ומעלה. שמות Tensors נקראים מהקובץ ולא מוכתבים (ADR-005).
- **Python 3.14** (הוכרע 2026-11-12; ADR-002 עודכן). PySide6 6.12 תומך (>=3.10,<3.16). אריזה על 3.14 טרם נבדקה.
- **תקציב Latency** (הוכרע 2026-11-12, ב-4 threads על i5-6500T): אינטראקטיבי עד קלט 256 (~0.8s), רגיל עד 384 (~2s, ברירת מחדל), איכות עד 512 (~4s, ידני), 768+ ידני בלבד. ROI גדול מוקטן ל-512 לכל היותר. פירוט ב-`DEVELOPMENT_LOG.md`.
- **OpenVINO CPU כ-backend ראשי (3 threads), ORT כ-fallback** (ADR-011, **מאושר במפורש** ע"י המשתמש 2026-11-12).
- **ללא GPU, כולל ה-iGPU של Intel** (הוכרע 2026-11-12): ה-iGPU נמדד (HD 530) ולא נתן יתרון משמעותי על OpenVINO CPU, והמשתמש בחר לא להרחיב אליו.
- מיזוג: `orig*(1-m) + result*m`, כך שמחוץ למסכה הפיקסלים זהים למקור.

## מה KIRO הוסיף (שלב 0)

- חבילת `stage0_validation/`: `FileValidator`, `ModelInspector`, `ModelCardUpdater`, `ReportGenerator`, `ValidationOrchestrator`. נקודת כניסה `python validate_stage0.py` (קודי יציאה 0–6). הוראות ב-`stage0_validation/README.md`.
- מפרט ב-`.kiro/specs/stage-0-model-validation/`. משימות 12–13 (בדיקות אופציונליות) לא בוצעו.
- בסביבת Claude: 21 בדיקות עוברות, 9 נכשלות רק כי הן דורשות את הקובץ האמיתי.

## שאלות פתוחות

1. ~~האם `validate_stage0.py` הסתיים בקוד 0?~~ נענה: כן, פעמיים. (ייתכן שאצל המשתמש נשארו שינויים מקומיים ב-`model_card_updater.py` ו-`_tmp_test_model_card.py` שטרם נדחפו; לבדוק עם `git status`.)
2. מחיקת קבצים מיותרים: `_tmp_test_model_card.py`, `MODEL_CARD.md.bak`, `models/big-lama/LICENSE (2)`, `models/big-lama/README (3).md`. נשארו כי לא אושרה מחיקה.
3. האם ליצור `CHANGELOG.md` ו-`UNIFY-PLAN.md`? הם מוזכרים בנוהל העבודה אך לא קיימים, ותוכנם המצופה לא ידוע.
4. האם לכתוב `test_model_card_updater.py` אמיתי (משימה 12.3) במקום הסקריפט הזמני?

## הצעד הבא – שתי שיחות במקביל

כללי עבודה משותפים כדי למנוע התנגשויות: לפני כל פוש `git pull --rebase`; כל שיחה נוגעת רק בתחומה; ב-`PROGRESS.md` כל שיחה מעדכנת רק את השורה שלה בטבלה; ב-`DEVELOPMENT_LOG.md` כל שיחה מוסיפה ערך חדש בסוף בלבד.

- **שיחה A – שאר M1.5.** תחום: `tools/m1/`, סעיף M1.5 ב-`ROADMAP.md`, §6 ב-`MODEL_CARD.md`. נשאר: (1) ~~אישור מפורש ל-ADR-011~~ ניתן (2026-11-12);  (2) עלות compile: להריץ `--dynamic` ולמדוד מטמון (`CACHE_DIR`); (3) UI Spike של Zoom/Pan על 24MP; (4) GIL וזמן עלייה; (5) אריזה על 3.14.
- **שיחה B – M2.** תחום: קוד המנוע (ספרייה חדשה ללא UI), בדיקות, סעיף M2 ב-`ROADMAP.md`. נקודות מחייבות מ-M1.5: **ברירת מחדל 3 threads** (פרמטר מוגדר; עלות 3–8% ב-ORT מול 4, ומשאיר ליבה ל-UI); מצבי הקלט (256/384/512) כפרמטר של ה-Pipeline; ה-Pipeline מחזיר ROI **לגדלים קבועים (buckets)** ולא לצורות שרירותיות, כי OpenVINO דורש compile לכל צורה; Backend מאחורי `InpaintModel`: **OpenVINO ראשי ו-ORT fallback לפי ADR-011** (זיהוי כשל וחזרה אוטומטית) (ה-API של שניהם נמדד: `run` על `rgb`/`mask`, פלט `inpainted`); ROI גדול מוקטן ל-512 לכל היותר ב-INTER_AREA (~2ms). לבדוק ב-M2 את השפעת ההקטנה על האיכות.

## פרומפט A (M1.5 – המשך)

```
אני ממשיך פרויקט להסרת אלמנטים מתמונה, ריפו: https://github.com/lodbigy-770/FIXPIC
(הטוקן אצלי; אספק אותו בשיחה.)
אופן עבודה: כמו תמיד (שכפול/pull, קריאת PROGRESS.md וסעיף השלב ב-ROADMAP.md בלבד, שלב אחד, עדכון PROGRESS/ROADMAP/DEVELOPMENT_LOG, קומיט ופוש עם git pull --rebase לפני, מחבר Claude <noreply@anthropic.com>, חיפוש רק עם grep ו-sed, סיכום בקצרה).
הקשר: אני שיחה A. שלב 0, M1 והמדידות של M1.5 בוצעו; התקציב ו-Python 3.14 הוכרעו. שיחה B מבצעת M2 במקביל ולא אגע בתחומה.
משימה: להשלים את M1.5 – סקריפט OpenVINO ו-3 threads, ואחר כך UI Spike, GIL וזמן עלייה, לפי הסדר ותוך כדי תיאום איתי על מה להריץ.
```

## פרומפט B (M2)

```
אני ממשיך פרויקט להסרת אלמנטים מתמונה, ריפו: https://github.com/lodbigy-770/FIXPIC
(הטוקן אצלי; אספק אותו בשיחה.)
אופן עבודה: כמו תמיד (שכפול/pull, קריאת PROGRESS.md וסעיף השלב ב-ROADMAP.md, ARCHITECTURE.md לפי הצורך, שלב אחד, עדכון PROGRESS/ROADMAP/DEVELOPMENT_LOG, קומיט ופוש עם git pull --rebase לפני, מחבר Claude <noreply@anthropic.com>, חיפוש רק עם grep ו-sed, סיכום בקצרה).
הקשר: אני שיחה B. שלבים 0, M1 ו-M1.5 (מדידות) בוצעו; תקציב Latency ו-Python 3.14 הוכרעו (ראו PROGRESS.md). שיחה A משלימה את M1.5 במקביל ולא אגע בתחומה (tools/m1).
משימה: M2 – Core Inpainting Engine. להתחיל ב-InpaintModel + מימוש ONNX + Pipeline (ROI, Pad/Resize, Pre/Post, מיזוג), ולהציע פירוק של שאר M2 לשלבים.
```
