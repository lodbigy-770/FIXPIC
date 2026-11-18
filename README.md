<p align="center"><img src="src/fixpic/ui/assets/fixpic.png" width="96" alt="FixPic"></p>

<h1 align="center">FixPic</h1>
<p align="center">הסרת אלמנטים ורקע מתמונה — מקומית, על המעבד (CPU), בלי רשת ובלי ענן.</p>

---

## מה זה

FixPic היא אפליקציית שולחן עבודה ל-Windows למחיקת אובייקטים, אנשים או רקע מתוך תמונות, ומילוי האזור שנמחק
בתוכן טבעי (Inpainting). כל העיבוד קורה **מקומית על המחשב בלבד** — שום תמונה לא נשלחת לשום שרת. מנוע ההסרה
מבוסס על מודל [Big LaMa](https://github.com/advimman/lama) (ONNX), ורץ דרך OpenVINO או ONNX Runtime, לפי מה
שזמין על המחשב.

הפרויקט **לשימוש אישי ולא מסחרי**. הממשק בעברית בלבד (RTL).

## יכולות עיקריים

- **פתיחת תמונה**: גרירה לחלון, `Ctrl+O`, או הדבקה מהלוח (`Ctrl+V`).
- **הסרה במברשת**: ציור חופשי על מה שרוצים להסיר, עם עיבוד מיידי בכל משיכה.
- **בחירה חכמה**: מלבן, לאסו או שרביט (לפי דמיון צבע) שמציעים בחירה אוטומטית של אובייקט לפני ההסרה.
- **כלי מסכה**: הרחבה/כיווץ, מחיקת כתמים קטנים, מחיקת רכיב בלחיצה, לאסו להוספה/הסרה מהמסכה.
- **מצב שחזור**: החזרת אזור מהתמונה המקורית, לתיקון הסרה שלא הצליחה.
- **הסרת רקע**: הפיכת הרקע לשקוף תוך שמירה על האובייקט שנבחר.
- **מצב שקיפות**: הפיכת אזור נבחר לשקוף ישירות.
- **השוואת לפני/אחרי**: מתג קבוע או החזקה זמנית.
- **Undo/Redo** מלא, כולל לאחר שמירה.
- **ייצוא**: PNG / JPEG / WebP, עם בקרת איכות לפורמטים דחוסים ושמירה על רזולוציה מלאה.
- **קיצורי מקלדת** ניתנים להתאמה אישית (תפריט עזרה ← קיצורי מקלדת).
- **דיווח קריסות מקומי**: יומן והודעת שגיאה ברורה; שום דבר לא נשלח החוצה.
- **ממשק בעברית מלא, RTL**, כולל גרירה, הדבקה ותמיכה בנתיבי קבצים בעברית.

## הורדה והתקנה

הדרך הפשוטה ביותר: בעמוד ה-[Releases](https://github.com/lodbigy-770/FIXPIC/releases) של הריפו יש מתקין יחיד
בשם `FixPicSetup-X.Y.Z.exe` (נבנה אוטומטית מ-`.github/workflows/release.yml` בכל תג גרסה, כולל המודל — אין
צורך להוריד או להתקין שום דבר בנפרד). הרצתו מתקינה את FixPic ל-`%LOCALAPPDATA%\Programs\FixPic`, בלי צורך
בהרשאות מנהל.

## הרצה מהמקור (לפיתוח)

```powershell
pip install -e .[dev]
fixpic-app                      # פותח את הממשק
python -m fixpic תמונה מסכה -o פלט   # CLI לבדיקת המנוע בלי ממשק
```

להרצת הבדיקות (על Linux אפשר גם בלי מסך):

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

### אריזה למתקין EXE (Windows)

```powershell
python tools\m5\third_party_notices.py
pyinstaller fixpic-app.spec --noconfirm --distpath dist --workpath build\pyi
python tools\m5\check_dist.py dist\FixPic
iscc /DMyAppVersion=<גרסה> packaging\fixpic.iss     # דורש התקנת Inno Setup
```

הפלט: `dist\installer\FixPicSetup-<גרסה>.exe`. לשחרור רשמי כ-Release עם המתקין מצורף, מספיק לתייג:

```bash
git tag -a v<גרסה> -m "..." && git push origin v<גרסה>
```

ו-GitHub Actions (`.github/workflows/release.yml`) בונה ומשחרר לבד; `tools/m5/make_release.py` קיים כנתיב
גיבוי ידני אם ה-workflow נכשל.

## המודל והרישיון שלו

| | |
|---|---|
| מודל | [Big LaMa](https://github.com/advimman/lama) (ONNX), גרסת ההמרה: [smile-forge/big-lama-onnx](https://huggingface.co/smile-forge/big-lama-onnx) |
| רישיון המודל | Apache-2.0, Samsung Research |
| רישיון הפרויקט | **טרם סופי** — כרגע לשימוש אישי ולא מסחרי בלבד; ראו הפריט הפתוח ב-`docs/architecture/cleanup-ai/ROADMAP.md` |
| רישיונות צד שלישי | נוצרים בזמן האריזה (`tools/m5/third_party_notices.py`) ומוצגים בתפריט עזרה ← אודות |

## תיעוד נוסף (לפיתוח)

כל תכנון, החלטות ארכיטקטורה והתקדמות מתועדים ב-`docs/architecture/cleanup-ai/`:

- [`ARCHITECTURE.md`](docs/architecture/cleanup-ai/ARCHITECTURE.md) — מבנה המערכת והחלטות (ADR).
- [`PLAN.md`](docs/architecture/cleanup-ai/PLAN.md) ו-[`ROADMAP.md`](docs/architecture/cleanup-ai/ROADMAP.md) — תכנון ושלבים.
- [`PROGRESS.md`](docs/architecture/cleanup-ai/PROGRESS.md) — מצב נוכחי, לפי שלב.
- [`PARITY-SPEC.md`](docs/architecture/cleanup-ai/PARITY-SPEC.md) — מפרט היכולות ועניינים משפטיים.
- [`MODEL_CARD.md`](docs/architecture/cleanup-ai/MODEL_CARD.md) — פרטי המודל.
- [`DEVELOPMENT_LOG.md`](docs/architecture/cleanup-ai/DEVELOPMENT_LOG.md) — יומן פיתוח מפורט, צעד אחר צעד.
