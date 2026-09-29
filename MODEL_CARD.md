# Model Card: LiLT fine-tuned on FUNSD (segment-level boxes)

## Model description

`LiltForTokenClassification` initialized from [`SCUT-DLVCLab/lilt-roberta-en-base`](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) and fine-tuned for semantic entity recognition (SER) on FUNSD. Predicts one of 7 BIO-scheme labels per token: `O`, `B-HEADER`, `I-HEADER`, `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`.

This is a reproduction of the downstream fine-tuning result from Wang, Jin & Ding, *"LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding"* (ACL 2022). It does **not** reproduce the paper's pretraining stage.

## Intended use

- Extracting header/question/answer fields from scanned forms structurally similar to FUNSD (English, single-column business-form layouts with OCR-derived words and bounding boxes already available).
- Research and educational use, consistent with FUNSD's own license terms (non-commercial).
- **Not** intended for production deployment on arbitrary document types, non-English text, or documents with significantly different layout conventions than FUNSD's training distribution (see Limitations).

## Training data

- **Dataset:** [`nielsr/funsd-layoutlmv3`](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3) — 149 train / 50 test forms.
- **Box representation:** segment-level — every word within one annotated text segment shares a single bounding box, rather than each word having its own tight box. This is a deliberate, tested choice (see the project README's "box granularity" section) — the alternative per-word-box dataset (`nielsr/funsd`) produces a meaningfully different, lower-scoring model.
- FUNSD's own annotations derive from real scanned business forms; specific document sources/provenance follow the original FUNSD release, not this project.

## Training procedure

| Hyperparameter | Value |
|---|---|
| Base checkpoint | `SCUT-DLVCLab/lilt-roberta-en-base` |
| Learning rate | 5e-5, linear decay, 10% warmup |
| Max steps | 2500 |
| Per-device batch size | 8 |
| Gradient accumulation | 1 (effective batch size 8) |
| Precision | fp16 |
| Seeds | 42, 123, 2024 |
| Model selection | best checkpoint by eval F1 during training (`load_best_model_at_end`) |

## Metrics

Entity-level precision/recall/F1 via `seqeval`, matching the paper's evaluation protocol.

**Overall (3 seeds):** F1 = 0.8825 ± 0.0039 — paper reports 0.8841.

**Per-class (seed 42):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ANSWER | 0.8651 | 0.9081 | 0.8861 | 805 |
| HEADER | 0.5865 | 0.5126 | 0.5471 | 119 |
| QUESTION | 0.8943 | 0.9199 | 0.9070 | 1049 |

Full methodology, ablations, and the robustness sweep are in the project [README](README.md).

## Limitations and known failure modes

- **HEADER is the weakest class** (F1 ≈ 0.55), driven by its small support (119 of 1973 test entities) — expect materially worse performance on header-style fields than on questions or answers.
- **Degrades sharply under layout corruption.** Once roughly 20% of a document's boxes are shuffled or mismatched (or ~45% dropped, or jittered by ~29 units on a 0-1000 scale), F1 falls below a model trained with *no* layout information at all — see the corruption sweep in the README. This model should not be assumed robust to noisy or unreliable OCR-derived box coordinates.
- **Trained and evaluated only on FUNSD's specific segment-level box convention.** Feeding it per-word boxes (or boxes from a different OCR pipeline's segmentation) is an out-of-distribution shift this model was not tested against — the README documents a ~8-point F1 drop when the same architecture is trained on per-word boxes instead.
- **English only.** LiLT's architecture supports swapping the text encoder for other languages, but this specific checkpoint was fine-tuned only on English FUNSD text.
- **Small fine-tuning set (149 documents)** with near-zero training loss by step 2000 against a validation loss of ~1.8–2.2 — indicates meaningful overfitting to FUNSD's specific document population, not a general-purpose form-understanding model.
- **4 of 50 test documents exceed the 512-token limit** and are silently truncated during evaluation; on real-world documents longer than this, expect entity loss past the cutoff (concentrated in whichever entity type appears later in the document, empirically QUESTION-heavy in this test set).

## How to use

```python
from transformers import AutoTokenizer, LiltForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
model = LiltForTokenClassification.from_pretrained("<path-to-saved-checkpoint>")

encoding = tokenizer(
    words,            # list[str], pre-tokenized
    boxes=boxes,       # list[list[int]], 0-1000 normalized [x0,y0,x1,y1] per word
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)
outputs = model(**encoding)
predictions = outputs.logits.argmax(-1)
```

See `app.py` for a full inference + visualization example.
