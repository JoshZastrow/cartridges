# Data quality check: 383 labeled pages (2026-10-12)

Per-page metrics: `data/train/quality/pages.csv` (local, not committed): image size, ink
fraction, label chars/words/tokens (model tokenizer), placeholder labels, flags. Every page was
also reviewed visually on contact sheets.

## Summary
- 383 pages (Morning_Pages 285, There_He_Is 98); all labels `manual_gpt_5-5_visual`; 73 without a date.
- Median image 1360 × 1815 px; median label 252 tokens; longest **There_He_Is/83, 2,179 tokens**.
- No duplicate labels.

## Findings
| Page | Split | Issue |
|---|---|---|
| Morning_Pages/87 | test | **Typed page**, not handwriting (same as the removed There_He_Is p87) |
| Morning_Pages/21 | test | Blank page, label is empty `""` |
| Morning_Pages/214 | train | Drawing, label `No handwritten text.` |
| Morning_Pages/250 | train | Drawing, label `No handwritten text. Page contains an abstract line drawing.` |
| Morning_Pages/35 | train | Sketches, label `[No written text. The page contains sketches of a person and a flower.]` |
| 13 There_He_Is pages (58, 63, 74, 76, 78, 79, 81, 82, 83, 90, 93, …) | mostly train | Tall handwritten pages; labels over 1,024 tokens (cap raised to 2,560) |

"No text" pages use four different label conventions, so the model gets inconsistent targets
for the same situation.

Other flags in the CSV (tall images, text-vs-ink outliers) were checked visually: genuine long
handwritten pages, diagram pages with handwritten notes (Mo56–64), title pages (Mo1, Th1).
