# models/big-lama – Big LaMa (ONNX)

> סטטוס: **קובץ המודל נמצא בריפו כ-Git LFS pointer בלבד בסביבת העבודה של Claude** (הרשת חוסמת את אחסון ה-LFS). אימות SHA256 של התוכן, טעינה והרצה טרם בוצעו.

## מקור

| פריט | ערך |
|------|-----|
| ריפו Hugging Face | https://huggingface.co/smile-forge/big-lama-onnx |
| Revision / Commit ב-HF | **לא ידוע** (ה-README אינו מציין; יש להשלים מדף ההיסטוריה ב-HF) |
| קוד מקור של המודל | advimman/lama בקומיט `786f593` (לפי ה-README של הריפו) |
| משקלים | smartywu/big-lama (לפי ה-README של הריפו) |
| מחבר ההמרה ל-ONNX | smile-forge (לפי ה-README של הריפו) |
| מאמר | Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions*, WACV 2022 |

## קבצים

| קובץ | גודל (בתים) | SHA256 | סטטוס אימות |
|------|-------------|--------|--------------|
| `inpainter.onnx` | 208,851,664 | `ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff` | **ערכים מה-LFS pointer בלבד; לא אומתו מול התוכן** |
| `LICENSE` | 11,549 | `99820be3ed3693f7726e7cda8baddbee9961137b4a402b3c4b2d539a3e234161` | חושב מהקובץ שהועלה |
| `UPSTREAM_README.md` | 4,025 | `ef7b86ad607d84ac8f768defd498197a3311e667a058f2a0729bc8744f08e78d` | חושב מהקובץ שהועלה; עותק של ה-README מ-HF |

לפי ה-README של הריפו, המודל הוא "גרף ONNX אחד" (fp32, כ-209MB). הגודל ב-pointer תואם לכך. לא נמצאו עדויות לקובצי External Data, אך זה **לא אומת** מול הקובץ.

## רישיון

Apache-2.0 (קוד ומשקלים של big-lama, וגם מתכון ההמרה), לפי ה-README. קובץ `LICENSE` שבתיקייה הוא Apache-2.0 עם הערת זכויות "Copyright [2021] Samsung Research".

חובות הפצה (Apache-2.0): לצרף עותק של הרישיון, ולשמור הודעות זכויות יוצרים ושיוך. לא נבדקו תנאי נתוני האימון. לבדיקה משפטית לפני הפצה מסחרית.

## Git LFS

`.gitattributes` מגדיר `models/big-lama/*.onnx` כ-LFS. אחרי clone: `git lfs install && git lfs pull`. ודאו שהקובץ אינו pointer (הגודל צריך להיות 208,851,664 בתים, לא 134).

## אימות והפעלה

```bash
# אימות שלמות
sha256sum models/big-lama/inpainter.onnx   # צפוי: ef85edd8...ba2ff
# בדיקת מבנה Input/Output (ללא הנחות)
python tools/poc/probe_and_run.py probe --model models/big-lama/inpainter.onnx
# בדיקת Inpainting על CPU (תמונה סינתטית, או --image/--mask)
python tools/poc/probe_and_run.py run --model models/big-lama/inpainter.onnx
```

דרישת הריפו: ONNX Runtime 1.30 ומעלה. פרטי קלט/פלט ב-`docs/architecture/cleanup-ai/MODEL_CARD.md`.
