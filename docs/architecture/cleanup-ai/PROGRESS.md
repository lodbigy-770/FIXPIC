# PROGRESS – מצב הפרויקט והמשך עבודה

> עודכן: 2026-11-11. מסמך קצר להעברת הקשר לשיחה הבאה. הפירוט המלא ב-`DEVELOPMENT_LOG.md`, השלבים ב-`ROADMAP.md`.

## מצב נוכחי

| שלב | מצב |
|-----|-----|
| תיעוד ותכנון (PLAN, ARCHITECTURE, PARITY-SPEC) | בוצע |
| שלב 0 – קליטת מודל ואימות | **בוצע ואושר** ע"י פלט ההרצה של המשתמש (קוד יציאה 0). הטווח 64–4096 עדיין לא אומת |
| M1 – הרצת Big LaMa על CPU | **בוצע (2026-11-11).** 3 דוגמאות אמיתיות ו-3 סינתטיות עברו, הריפוד נבדק, הפלט מחוץ למסכה זהה למקור. איכות: אדם טוב, טקסט טוב מאוד, שריטה טובה. ממצא: 6.6–9.7 שניות ל-0.43MP על 4 ליבות, בלי תלות בגודל המסכה |
| M1.5 – Latency Spike | כלי מדידה מוכן ב-`tools/m1/latency_spike.py`; ממתין להרצה אצל המשתמש |
| M2–M5 | טרם התחיל |

## הגבלות סביבה (חשוב)

- Claude לא מצליח להוריד את `models/big-lama/inpainter.onnx`: הוא LFS pointer (134 בתים), והרשת חוסמת את `github-cloud.githubusercontent.com` ואת `media.githubusercontent.com`. כל מה שדורש את המודל האמיתי רץ אצל המשתמש.
- פתיחת הדומיין הראשון בהגדרות הרשת של Claude הייתה מאפשרת להריץ M1 בתוך השיחה, אך המשתמש בחר לבדוק מקומית.
- הטוקן לריפו נחשף בשיחה קודמת. מומלץ להחליף אותו; הוא לא נשמר באף קובץ.

## החלטות שכבר התקבלו

- שפת ממשק: עברית בלבד (RTL). כלי עצמאי לשולחן העבודה, Python ו-PySide6.
- שימוש אישי ולא מסחרי.
- ONNX Runtime CPU בלבד, גרסה 1.30 ומעלה. שמות Tensors נקראים מהקובץ ולא מוכתבים (ADR-005).
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

## הצעד הבא

M1.5 – הכלי `tools/m1/latency_spike.py` מוכן. המשתמש מריץ אותו ושולח את `latency_results.json`:

```
git pull
.venv\Scripts\python.exe tools\m1\latency_spike.py --quick    (כ-5 דקות; מומלץ להתחיל כך)
.venv\Scripts\python.exe tools\m1\latency_spike.py            (כ-15 דקות, מלא)
```

לפני ההרצה: חיבור לחשמל, סגירת תוכנות כבדות, ולא להשתמש במחשב בזמן ההרצה (הרעש במדידות הקודמות היה כ-45%). אחר כך Claude מנתח, מעדכן `ROADMAP.md`, `MODEL_CARD.md` §6 ו-`DEVELOPMENT_LOG.md`, ומציע תקציב Latency ו-ROI. נשארים אחרי זה: OpenVINO (אם המעבד Intel), UI Spike ו-ADR-002.

## פרומפט להדבקה בשיחה הבאה

```
אני ממשיך פרויקט להסרת אלמנטים מתמונה, ריפו: https://github.com/lodbigy-770/FIXPIC
(הטוקן אצלי; אספק אותו בשיחה.)

אופן עבודה:
1. שכפל את הריפו (או git pull אם כבר קיים).
2. קרא רק docs/architecture/cleanup-ai/PROGRESS.md ואת סעיף השלב הבא ב-ROADMAP.md
   (grep -n '^## ' ואז sed -n). ARCHITECTURE.md רק אם השלב דורש.
3. בצע שלב אחד בלבד, אלא אם כתבתי אחרת.
4. בסיום עדכן PROGRESS.md, ROADMAP.md ו-DEVELOPMENT_LOG.md.
5. קומיט ופוש אחרי כל שלב. מחבר: Claude <noreply@anthropic.com>.
6. חיפוש בקבצים רק עם grep ו-sed. בלי שינויים מחוץ לתחום המשימה.
7. בסיום סכם במשפטים בודדים: מה בוצע, מה הבא, שאלות פתוחות. אל תדביק קבצים שלמים.

הקשר: שלב 0 ו-M1 בוצעו. השלב הבא הוא M1.5; אני מריץ את המדידות אצלי ואשלח את הפלט כאן.
```
