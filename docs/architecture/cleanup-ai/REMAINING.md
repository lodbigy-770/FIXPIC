# מה נשאר, ומה להריץ אצל המשתמש

> עודכן: 2026-11-16 (אחרי M3.7.11 שלב 1). מקור האמת לסימונים הוא `ROADMAP.md`; כאן סיכום תפעולי. כשפריט מסתיים מוחקים אותו מכאן ומעדכנים ב-ROADMAP/PROGRESS.

## א. שלבים שנשארו (חוץ מ-M5)

| שלב | מה נשאר | תלוי ב |
|---|---|---|
| **M3.7.11 – קבלה** | סט נוסף של 3-5 מסכות גדולות; **החלטה על `boy_large`** (נקי משרידים אך מרוח ובלי מבנה מול Cleanup.pictures): (1) להמשיך לסט נוסף, (2) לחקור מודל/תהליך חלופי לחורים גדולים, (3) לקבל את המגבלה ולתעד | המשתמש (הרצה והחלטה) |
| **M3.7 – חלקיים** | 3.7.5-3.7.8 נבדקו חלקית על Big LaMa; **3.7.12** (הרחבת מסכה לפי קנה המידה של המודל) הוא כלי שלא הורץ; כיול 768 מול 1024 | המשתמש (הרצה) |
| **M3 – נתיבים ו-RTL** | בדיקת נתיבים בעברית, מקלדת עברית ושמירה מדיאלוג אמיתי ב-Windows; שורת "ממשק בעברית (RTL)" סומנה בוצעה (2026-11-16); נשאר רק מראה RTL בעין ב-Windows | המשתמש (Windows) |
| **M4 – עריכה מתקדמת** | שירותי המסכה והתכנון בוצעו, אבל **נשאר כל הממשק:** ציור המסכה על הקנבס, לחיצה על רכיב, לאסו, סליידר Feathering, קיצורי מקלדת וסדר Ctrl+Z, בחירה אוטומטית (AUTO-SELECT §7 שלבים 2-3), ו-**Spike מודל "לחיצה-חכמה"** (שלבים 4-6) | המשתמש (מועמד ל-Spike) |
| **שלב 0** | Revision של המודל ב-Hugging Face ותיעוד מלא ב-README של המודל; טווח 64-4096 לא אומת | גישה ל-HF |

## ב. מה להריץ אצלך (לפי סדר מומלץ)

לפני כל הרצה: `git pull`, חיבור לחשמל, לסגור תוכנות כבדות, לא לגעת במחשב בזמן המדידה. אחרי: `git add` לתיקיות הפלט, קומיט ודחיפה.

### 1. M3.7.11: הסט הנוסף (הכי דחוף)
לכל תמונה (12-24MP, כדאי: אדם שלם, אובייקט על רקע מורכב, אזור בקצה, כמה אזורים) שניקית באתר Cleanup.pictures:
```
python tools\m3\make_case.py "C:\נתיב\photo1.jpg" "C:\נתיב\photo1_cleanup.png" -o cases\photo1 --reference "C:\נתיב\photo1_cleanup.png" --ref-original
```
(`photo1` = שם תיקייה באנגלית בלי רווחים; התוצאה של האתר יכולה להיות קטנה יותר, אך באותו יחס גובה-רוחב.) לבדוק ב-`cases\photo1\mask_preview.jpg` שהאדום הוא האזור שהוסר. אחרי כל המקרים:
```
python tools\m3\acceptance.py cases\photo1 cases\photo2 cases\photo3 --out m3_accept_new
```
**להחזיר:** `m3_accept_new` ו-`cases\photo*`. אחר כך ממלאים "ממצא בעין" ב-`ACCEPT.md`. פירוט: `M3-ACCEPTANCE.md` (סעיף M3.7.11).

### 2. M3.7.12: הרחבת מסכה (אם רוצים לחקור את `boy_large`)
```
python tools\m3\quality_bench.py cases\boy_large --out m3_bench --sizes 384 --strategies auto --min-scales 0.5 --blends 32 --coarses 1 --coarse-sizes 768 --belt-px 0 --dilates auto,60,100,m8,m12
```
**להחזיר:** `m3_bench`. הכרעה בעין (המדדים לא אמינים בין ערכי הרחבה שונים).

### 3. M3: בדיקה ידנית ב-Windows
לפי `M3-ACCEPTANCE.md` §3 (סקריפט בדיקה ידנית), עם `python -m fixpic.app`: נתיבים בעברית, מקלדת עברית, שמירה בשם, Ctrl+S, הודעת "עיבוד ארוך" (M3.7.10) אחרי כ-3 שניות על אזור גדול, ו-Ctrl+Z בזמן עיבוד. **להחזיר:** רשימת סטיות, או צילומי מסך.

### 4. M4: Spike מודל "לחיצה-חכמה" (דורש החלטה ומועמד)
צריך להוריד זוג ONNX של מועמד משפחת SAM (מקודד + מפענח); הסקריפט לא מוריד ולא מכריע. אחרי שיש מועמד:
```
python tools\m4\model_spike.py --candidate mobilesam C:\models\enc.onnx C:\models\dec.onnx --threads 4 1 --contention
```
(דורש `onnxruntime`, `opencv-python-headless`, `numpy`, `psutil`.) **להחזיר:** `model_spike_results.json` ואת תמונות הפלט. הסף: `propose` עד 0.5s, `prepare` עד 5s ברקע. **שים לב:** חוזה הקלט/פלט של משפחת SAM לא אומת מול מודל אמיתי; אם הוא שונה הסקריפט עוצר ומדפיס שמות קלט/פלט.

### 5. M5 (במקביל)
- זיכרון בתמונות גדולות: `python tools\m5\memory_stress.py --model big-lama --threads 3 --json mem.json --md mem.md` (אפשר להגביל עם `--budget-mb`; ברירת מחדל 70% מהזיכרון). להחזיר `mem.json`/`mem.md`.
- רישיונות על Windows: `python tools\m5\license_audit.py --check --md lic_win.md`. להחזיר `lic_win.md`.
- **ביצועים על חומרת היעד (תקציב Latency): `python tools\m5\latency_budget.py --model big-lama --threads 4 --json latency_budget.json --md latency_budget.md`** (אחר כך גם `--threads 3`, ברירת המחדל של המוצר); להחזיר `latency_budget.json/md`. קוד יציאה 1 = חריגה מהתקציב. בדיקת Offline עם Firewall אמיתי, ואריזה (`fixpic.spec`, PyInstaller, Python 3.14 עם PySide6/OpenVINO/ORT/OpenCV): **אין עדיין סקריפט הרצה מוכן אצלך**; יוגדרו בשלבי M5 הבאים.

## ג. הערות
- הטוקן לריפו נחשף בשיחות רבות (גם בזו); מומלץ להחליף אותו, והוא אינו שמור בקובץ בריפו.
- Claude לא מצליח להוריד את `models/big-lama/inpainter.onnx` (קובץ LFS חסום ברשת הפיתוח), ולכן כל מה שדורש את המודל האמיתי נעשה אצלך.
