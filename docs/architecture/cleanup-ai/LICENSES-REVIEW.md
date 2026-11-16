# סקירת רישיונות – FIXPIC (M5)

> נוצר: 2026-11-16. **זו סקירה טכנית ראשונה, לא ייעוץ משפטי.** שימוש אישי ולא מסחרי (PROGRESS); לפני הפצה לאחרים יש לאמת מול גורם מוסמך (`PARITY-SPEC.md` §7).
> הטבלה נוצרה ב-`tools/m5/license_audit.py` על venv נקי (Linux, Python 3.12.3) עם התלויות מ-`pyproject.toml`. לא נבדקה סביבת Windows/3.14 שבה תיארזו התלויות בפועל, ולכן הסגור שם עלול להיות שונה. להרצה חוזרת (כולל אחרי שינוי תלויות): `python tools/m5/license_audit.py --md out.md --check`.

## 1. טבלת הרכיבים (מטא-דאטה של החבילות)

| חבילה | גרסה | רישיון (מטא-דאטה) | קטגוריה | חומר נלווה | נדרשת ע"י |
|---|---|---|---|---|---|
| PySide6 | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | (שורש) |
| PySide6_Addons | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | PySide6, PySide6_WebEngine |
| PySide6_Essentials | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | PySide6, PySide6_Addons, PySide6_Pdf |
| PySide6_Pdf | 6.12.0.140 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | PySide6 |
| PySide6_WebEngine | 6.12.0.140 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | PySide6 |
| shiboken6 | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (LGPL-3.0-only)) | Copyleft חלש | אין קובץ רישיון | PySide6_Essentials |
| flatbuffers | 25.12.19 | Apache 2.0 | מתירני | אין קובץ רישיון | onnxruntime |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (כל הרישיונות חלים (AND)) | מתירני | אזכור LGPL בקובצי הרישיון; אזכור GPL בקובצי הרישיון; 20 קובצי רישיון | (שורש), opencv-python-headless, onnxruntime, openvino |
| onnxruntime | 1.32.0 | MIT License | מתירני | אזכור LGPL בקובצי הרישיון; אזכור GPL בקובצי הרישיון; 2 קובצי רישיון | (שורש) |
| opencv-python-headless | 5.0.0.93 | Apache 2.0 | מתירני | אזכור LGPL בקובצי הרישיון; אזכור GPL בקובצי הרישיון; 4 קובצי רישיון | (שורש) |
| openvino | 2026.4.1 | Apache-2.0 | מתירני | אזכור LGPL בקובצי הרישיון; אזכור GPL בקובצי הרישיון; 4 קובצי רישיון | (שורש) |
| openvino-telemetry | 2025.2.0 | OSI Approved :: Apache Software License | מתירני | 1 קובצי רישיון | openvino |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause (רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה (Apache-2.0)) | מתירני | 3 קובצי רישיון | onnxruntime |
| pillow | 13.0.0 | MIT-CMU | מתירני | אזכור LGPL בקובצי הרישיון; אזכור GPL בקובצי הרישיון; 1 קובצי רישיון | (שורש) |
| protobuf | 7.36.2 | 3-Clause BSD License | מתירני | 1 קובצי רישיון | onnxruntime |

## 2. ממצאים

1. **אין תלות ב-Copyleft חזק (GPL/AGPL) כשלעצמה.** הטבלה מראה רק מתירניים ו-PySide6/shiboken6, שהם LGPL-3.0 בבחירה (הרישיון כפול: LGPL-3.0, GPL-2.0 או GPL-3.0; בוחרים LGPL). הכלי מסמן GPL/LGPL **בתוך קובצי הרישיון** של חמש חבילות; להלן מה שנמצא שם (ממצא ראשוני, לא כל קובץ נקרא עד הסוף):
   - **numpy:** הודעת `libgfortran`/GCC Runtime Library: GPL-3.0 **עם חריג ה-GCC Runtime Library**, שמתיר הפצה בתוכנה שאינה GPL. חל על הבינארים הנלווים של הגלגל.
   - **opencv-python-headless:** `LICENSE-3RD-PARTY.txt` כולל את הטקסט המלא של LGPL (נראה ש-OpenCV ארוז עם רכיבים בבינארי, למשל FFmpeg; **לא אומת איזה רכיב בדיוק**). זה דורש את אותן חובות LGPL כמו PySide6 לגבי הרכיב הזה.
   - **openvino:** `onetbb_third-party-programs.txt` (GPL-3.0 עם GCC Runtime Exception, בקובץ הרישיון של oneTBB) ו-`runtime-third-party-programs.txt` (liburing: LGPL-2.1 **או** MIT; OpenVINO משתמש בו תחת MIT לפי הקובץ).
   - **onnxruntime:** אזכור יחיד של LGPL/AGPL ב-`ThirdPartyNotices.txt` (במסגרת טקסט של רישיון צד שלישי; לא נבדק לאיזה רכיב).
   - **pillow:** `LICENSE` מפרט את הרישיונות של ספריות נארזות (FreeType FTL/GPLv2, ועוד); נאמר שם שקבצים תחת GPL אינם נארזים בגלגל.
2. **PySide6 (LGPL-3.0):** חבילות PySide6 **אינן כוללות קובץ רישיון בתוך ה-dist-info** (הכלי לא מצא אף אחד), ולכן את הודעות הרישיון והטקסט המלא של LGPL-3.0 וה-GPL-3.0 (ש-LGPL-3.0 מפנה אליו) צריך לצרף ידנית לאריזה. הסעיף "עמידה ברישיון LGPL של PySide6" ב-ROADMAP נשאר פתוח; אריזה ב-`onedir` (ספריות דינמיות נפרדות, ניתנות להחלפה) היא הדרך שנבחרה ב-ARCHITECTURE והיא מתיישבת עם דרישת הקישור הדינמי.
3. **`pip install PySide6` מתקין גם PySide6_Addons, PySide6_WebEngine ו-PySide6_Pdf** (Meta-package). הקוד משתמש רק ב-`QtCore`, `QtGui`, `QtWidgets` (בבדיקות גם `QtTest`; ו-`QtNetwork` מופיע בבדיקת ה-Offline) וכולם נמצאים ב-**PySide6_Essentials** (אומת על ה-RECORD של החבילה). ב-venv הנקי תיקיית `PySide6` היא כ-673MB, ו-`libQt6WebEngineCore` לבדו כ-195MB. WebEngine הוא Chromium עם עשרות רישיונות צד שלישי משלו ועם דפדפן שלם באריזה של כלי "ללא רשת". **המלצה (לא בוצעה, דורשת אישור):** להחליף ב-`pyproject.toml` את `PySide6` ב-`PySide6-Essentials` ולוודא שהבדיקות עוברות. ב-PyInstaller לא כדאי לסמוך על כך שהאריזה תשמיט את WebEngine לבד.
4. **`openvino-telemetry`** מותקנת כתלות של openvino, ו-`import openvino` טוען את מודוליה (כולל `backend_ga4` ו-`sender`). בבדיקה הקיימת `tests/test_m5_offline.py` (חוסמת `socket` בתוך התהליך) לא נרשם ניסיון רשת, ובקוד הטלמטריה הדיאלוג לקבלת הסכמה (opt-in) דורש טרמינל; כלומר אין ראיה שהיא שולחת משהו מהאפליקציה. **עם זאת:** טענת "ללא רשת" כדאי שלא תיסמך על כך; **המלצה:** להחריג את `openvino_telemetry` (ואת `openvino.tools.ovc`) באריזה, ולבדוק שאין כתיבת קובץ הסכמה בתיקיית המשתמש בהרצה הארוזה (לא נבדק).
5. **המודל (Big LaMa):** Apache-2.0, Copyright 2021 Samsung Research (`MODEL_CARD.md` §1; קובץ `LICENSE` נבדק בשלב 0). חובות הפצה: לצרף את הרישיון ולשמור הודעות שיוך (קיימים ב-`models/big-lama/`). `MODEL_CARD.md` מציין שנתוני האימון לא נבדקו משפטית; זה פתוח.
6. **אין מידע על:** רישיון הפונט/אייקונים של הממשק (לא נבדקו), `onnx` (תלות פיתוח בלבד, לא נארזת), וכלי האריזה/מתקין (PyInstaller: GPL עם חריג לתוכנה שנארזת; Inno Setup: רישיון משלו). אלה לא נסרקו.

## 3. מה בוצע אחרי הסקירה (2026-11-16) ומה נשאר

**בוצע (אישור המשתמש):** (א) `pyproject.toml` ו-`requirements-dev.txt`: `PySide6` הוחלף ב-`PySide6-Essentials`; ב-venv נקי עם Essentials בלבד כל החבילה עוברת (1024 עוברות, 3 מדולגות; Linux/3.12), והסגור ירד מ-15 ל-10-11 חבילות. (ב) `fixpic.spec`: `excludes` ל-`openvino_telemetry`, ל-`openvino.tools` ולרכיבי PySide6 שאינם בשימוש (WebEngine, Pdf, Quick, Qml, 3D, Multimedia). נבדק רק ברמת הייבוא: עם `openvino_telemetry` ו-`openvino.tools` חסומים `import openvino`, `openvino.Core()` ו-`tests/test_backends.py` + `tests/test_m5_offline.py` (12) עוברים. **לא נבדק:** שהאריזה בפועל (PyInstaller) מכבדת את ה-excludes, ושהבינארי הארוז עובד. (ג) `tools/m5/third_party_notices.py` + `licenses/LGPL-3.0.txt` ו-`licenses/GPL-3.0.txt` (הטקסטים הקנוניים של GNU מ-Debian `common-licenses`): מייצר `build/THIRD_PARTY_NOTICES.txt` עם קובצי הרישיון המלאים של הסגור, רישיון המודל ו-LGPL/GPL של PySide6 (ב-Linux: כ-935KB, 10 חבילות).

**נשאר:**
- הרצה חוזרת של הכלי **על סביבת Windows/Python 3.14 שבה תיארז האפליקציה** (ועל תיקיית ה-dist הארוזה, כדי לראות מה באמת נארז).
- הכללת `THIRD_PARTY_NOTICES.txt` בפועל בתיקיית האריזה והצגתו באפליקציה ("אודות"); המחולל אינו מתפרס על חבילות ללא קובץ רישיון (PySide6_Essentials, shiboken6, flatbuffers מדווחות באזהרה; הרישיון שלהן נכלל מהטקסטים ב-`licenses/` או במטא-דאטה בלבד).
- בירור איזה רכיב LGPL נארז ב-OpenCV (קריאה של `LICENSE-3RD-PARTY.txt` המלא) והאם אפשר להסתפק בהודעה.
- קריאת תנאי השימוש של האתר ובדיקה משפטית אם ההפצה תורחב מעבר לשימוש אישי.
