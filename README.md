# LiLT Reproduction and Robustness Evaluation for Multilingual Document AI

Reproduction of LiLT (Wang, Jin & Ding, ACL 2022) semantic entity recognition on FUNSD, using the released `SCUT-DLVCLab/lilt-roberta-en-base` checkpoint, plus layout ablations and an OCR/layout-corruption robustness study.

**This reproduces the paper's downstream fine-tuning result. It does not reproduce LiLT's pretraining** (IIT-CDIP, 5 epochs, 4×A40 GPUs) — that's outside the scope of what free-tier compute can do, and isn't attempted here.

## Headline result

| Metric | This work | Paper (Table 2) |
|---|---|---|
| FUNSD SER, entity-level F1 | **0.8825 ± 0.0039** (3 seeds) | 0.8841 |

Config: `nielsr/funsd-layoutlmv3` (segment-level boxes), `SCUT-DLVCLab/lilt-roberta-en-base`, `lr=5e-5`, `max_steps=2500`, effective batch size 8, `transformers==5.16.1`.

### Per-class breakdown (seed 42)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ANSWER | 0.8651 | 0.9081 | 0.8861 | 805 |
| HEADER | 0.5865 | 0.5126 | 0.5471 | 119 |
| QUESTION | 0.8943 | 0.9199 | 0.9070 | 1049 |
| **Micro avg** | 0.8664 | 0.8905 | **0.8783** | 1973 |

HEADER is the weakest class, consistent with its small support (119 of 1973 entities) — this matches the pattern seen across independent public reproductions of this checkpoint, not an anomaly specific to this run.

## A note on box granularity (the main methodological finding)

FUNSD is available on the Hugging Face Hub in two forms:

- **`nielsr/funsd`** — per-word bounding boxes.
- **`nielsr/funsd-layoutlmv3`** — segment-level boxes (every word in a line/segment shares one bounding box), plus the page image.

These are **not interchangeable**, and the paper's text doesn't specify which one it used. This project tested both:

| Dataset variant | Mean F1 (3 seeds) | Std |
|---|---|---|
| `nielsr/funsd` (per-word boxes) | 0.7993 | 0.0032 |
| `nielsr/funsd-layoutlmv3` (segment boxes) | **0.8825** | 0.0039 |

Before settling on this explanation, two other hypotheses were tested and ruled out:
1. **Manual vs. tokenizer-native subword/label alignment** — switching from a hand-rolled `word_ids()`-based alignment to the tokenizer's built-in `word_labels=` argument produced no measurable change in any per-class score.
2. **Truncation at 512 tokens** — 4 of 50 test documents exceed the 512-token limit, but the dropped entities are overwhelmingly `QUESTION` (182 label positions) rather than `ANSWER` (14), which is the opposite of what would explain `ANSWER`'s underperformance on the per-word dataset.

Switching to segment-level boxes closed the gap almost entirely and moved every per-class score into the range seen in independent public reproductions of this checkpoint (e.g. [philschmid's LiLT tutorial](https://www.philschmid.de/fine-tuning-lilt), which reports ~0.89 F1 using this same segment-level dataset). **Both numbers are reported here deliberately** — this is a real, interesting property of the box representation the model is trained on, not a bug to be hidden by picking the more favorable result.

## Layout ablations

Isolates whether the model actually uses layout information, using the segment-box dataset, same architecture and training budget throughout — only the `bbox` input differs.

| Condition | Mean F1 | Seeds |
|---|---|---|
| LiLT, true boxes | **0.8825** | 3 |
| LiLT, zero boxes | 0.7185 | 1 |
| LiLT, shuffled boxes | 0.7203 | 1 |
| RoBERTa, text-only (no layout stream) | 0.7235 | 1 |

True boxes outperform all three layout-degraded conditions by roughly 16 points — more than 40× the observed seed-to-seed noise on the true-box run, so a single seed per ablation is sufficient to establish the effect. Zero-box and shuffled-box scoring almost identically indicates the model isn't merely benefiting from "having some numbers present" — the *specific geometry* matters, which is the paper's central claim.

**Caveat:** the text-only baseline uses the LiLT checkpoint's own RoBERTa tokenizer (not a separately-initialized `roberta-base` tokenizer), so its `input_ids`/`labels` are identical to the LiLT runs — only the `bbox` input differs across all four conditions. This was a deliberate control, not a shortcut, forced by a tokenizer environment issue during development (see Limitations).

## Corruption robustness sweep

Evaluates the trained true-box model (seed 42) under four controlled corruptions of the *test-set* boxes, at 6 severity levels each, averaged over 3 corruption-randomness seeds (model held fixed). No retraining — evaluation-only, which is why this extension is cheap.


| Corruption | F1 falls below the no-layout baseline (~0.72) at roughly |
|---|---|
| Reading-order shuffle | ~21% of boxes shuffled |
| Token-box mismatch | ~20% of token-box pairs swapped |
| Box dropout | ~45% of boxes zeroed |
| Coordinate jitter | ~29 units of noise (0-1000 scale) |

Notable findings:
- **Shuffle and token-box-mismatch curves are near-identical** at every severity level — both corruptions are effectively the same perturbation (randomizing which box goes with which token) at different granularities.
- **Full box dropout (all-zero) is *worse* than full box shuffle** (F1 0.154 vs. 0.276), despite shuffle preserving no correct geometry either. A plausible explanation is that all-zero boxes are out-of-distribution for a model that never saw that input during training, while shuffled boxes — though wrong — are still in-distribution values.
- **A layout-trained model degrades below a model trained with no layout at all** once roughly a fifth of its boxes are corrupted. Layout awareness is a net positive only within a moderate corruption regime — this is a real limitation of the trained model, not just an ablation curiosity, worth surfacing to anyone considering deploying it on noisy real-world OCR.

Error bars reflect variability across 3 corruption-randomness seeds on the **same** seed-42 model — they do not capture model-to-model (training-seed) variance, since only one trained model's weights were saved for this sweep (see Limitations).

## Success criteria (from the original project plan)

- **Reproduction success** — mean F1 within 1-2 points of the paper: **met** (0.8825 vs. 0.8841, using segment-level boxes).
- **Scientific success** — corruption produces measurable, interpretable degradation with variability reported: **met**.
- **Engineering success** — a fresh runtime can install dependencies, train/evaluate, and regenerate results from one notebook: **met**, with the caching layer below.
- **Portfolio success** — paper-reported and reproduced numbers kept separate, no overclaiming: **met** — this README does not claim full-paper reproduction, and reports both dataset variants honestly rather than only the closer one.

## Reproducing this

Everything lives in one notebook (`LiLT_full_pipeline.ipynb`), designed to be re-run without cost once results are cached:

- Every expensive step (dataset encoding, model training, evaluation) checks a `CACHE_ROOT` directory before doing any work, and skips straight to loading saved results on a cache hit.
- Trained model weights are only persisted for the one run the corruption sweep depends on (`lilt_true_segment`, seed 42) — every other run keeps only its metrics/logits/labels (a few MB), not full weights, to stay within free-tier storage limits.
- On Colab, `CACHE_ROOT` should point at a mounted Google Drive folder for persistence across sessions. On Kaggle, point it at `/kaggle/working` and commit ("Save Version") to persist past the session — Kaggle has no Drive-equivalent silent persistence.

Pinned dependencies (confirmed working):
```
transformers==5.16.1
datasets==4.8.5
tokenizers==0.23.1
seqeval==1.2.2
accelerate==1.14.0
torch==2.11.0+cu128
```
`transformers==5.0.0` was observed to silently break this pipeline — `AutoTokenizer` resolves to a generic backend that accepts `boxes=`/`word_labels=` without error but produces no `bbox`/`labels` columns at all, and downstream training fails on an empty result rather than a clear error. Pin the version above rather than an unbounded `>=`.

## Limitations

- **HEADER F1 (~0.55) is the weakest class** in both this work and in every public reproduction of this checkpoint checked during development — a known effect of its small support (119 of 1973 test entities), not specific to this run.
- **Ablation and text-only baselines are single-seed** (F1 differences are ~16 points, roughly 40× the observed seed-to-seed std on the true-box run, so this doesn't affect the ablation's conclusion, but no confidence interval is reported for these specific numbers).
- **Corruption sweep error bars reflect corruption-randomness variance only**, not training-seed variance — only one trained model (seed 42) had its weights saved.
- **4 of 50 test documents exceed the 512-token limit** and are silently truncated; entity loss from this was checked and is concentrated in QUESTION, not ANSWER (see the box-granularity discussion above).
- **Text-only baseline reuses the LiLT checkpoint's tokenizer** rather than a `roberta-base`-native one, due to a `RobertaTokenizerFast` availability issue encountered during development. Text content is identical across all baseline conditions either way, so this doesn't compromise the comparison, but it's a deviation from the most literal reading of "text-only RoBERTa."
- **Full-paper pretraining (IIT-CDIP) is not attempted or claimed.**
