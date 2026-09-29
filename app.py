"""
LiLT / FUNSD inference demo.

Loads the fine-tuned checkpoint and lets you pick a FUNSD test-set document
to run inference on, drawing predicted entity boxes over the page image.

Scope note: this uses FUNSD's own OCR-derived words/boxes (already present
in the dataset), not a fresh OCR pass on an arbitrary uploaded image. The
model was trained on FUNSD's segment-level box convention specifically
(see MODEL_CARD.md) -- boxes from a different OCR pipeline's word/line
segmentation would be a distribution shift the model wasn't evaluated
against. Wiring in pytesseract for arbitrary uploads is a reasonable next
step, not included here to keep this demo's predictions strictly within
the setting the model card's numbers describe.

Run:
    pip install gradio transformers torch datasets pillow
    CHECKPOINT_DIR=/path/to/saved/model python app.py
"""

import os

import gradio as gr
import torch
from datasets import load_dataset
from PIL import ImageDraw, ImageFont
from transformers import AutoTokenizer, LiltForTokenClassification

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./model")
BASE_TOKENIZER = "SCUT-DLVCLab/lilt-roberta-en-base"
DATASET_ID = "nielsr/funsd-layoutlmv3"
MAX_SEQ_LENGTH = 512

LABEL_COLORS = {
    "HEADER": "blue",
    "QUESTION": "red",
    "ANSWER": "green",
}

print(f"Loading tokenizer ({BASE_TOKENIZER}) and model ({CHECKPOINT_DIR}) ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
model = LiltForTokenClassification.from_pretrained(CHECKPOINT_DIR)
model.eval()

id2label = model.config.id2label

print(f"Loading {DATASET_ID} test split ...")
test_dataset = load_dataset(DATASET_ID, split="test")
NUM_EXAMPLES = len(test_dataset)


def unnormalize_box(box, width, height):
    """FUNSD boxes are normalized to a 0-1000 scale; convert back to pixels
    for drawing onto the actual page image."""
    return [
        width * (box[0] / 1000),
        height * (box[1] / 1000),
        width * (box[2] / 1000),
        height * (box[3] / 1000),
    ]


def draw_predictions(image, boxes, word_predictions):
    image = image.convert("RGB").copy()
    width, height = image.size
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, label in zip(boxes, word_predictions):
        if label == "O":
            continue
        bare_label = label.split("-", 1)[-1]  # "B-QUESTION" / "I-QUESTION" -> "QUESTION"
        color = LABEL_COLORS.get(bare_label, "black")
        px_box = unnormalize_box(box, width, height)
        draw.rectangle(px_box, outline=color, width=2)
        draw.text((px_box[0], max(0, px_box[1] - 10)), bare_label, fill=color, font=font)

    return image


@torch.no_grad()
def run_inference(example_index):
    example = test_dataset[int(example_index)]
    words = example["tokens"]
    boxes = example["bboxes"]
    image = example["image"]

    encoding = tokenizer(
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    # Keep the tokenizer's own word_ids() to map subword predictions back to
    # whole words -- only the FIRST subword's prediction is used per word,
    # matching how the model was trained (only_label_first_subword).
    word_ids = encoding.word_ids(batch_index=0)

    outputs = model(**{k: v for k, v in encoding.items() if k != "overflow_to_sample_mapping"})
    predicted_ids = outputs.logits.argmax(-1).squeeze().tolist()

    word_predictions = []
    seen_words = set()
    for token_idx, wid in enumerate(word_ids):
        if wid is None or wid in seen_words:
            continue
        seen_words.add(wid)
        word_predictions.append(id2label[predicted_ids[token_idx]])

    # Documents longer than 512 tokens get truncated by the tokenizer above;
    # words beyond the truncation point have no prediction. Pad with "O"
    # rather than crash, and surface the truncation in the returned table.
    n_missing = len(words) - len(word_predictions)
    if n_missing > 0:
        word_predictions += ["O"] * n_missing

    annotated_image = draw_predictions(image, boxes, word_predictions)

    rows = [
        {"word": w, "predicted_label": p}
        for w, p in zip(words, word_predictions)
        if p != "O"
    ]
    truncation_note = (
        f"⚠️ Document has {len(words)} words; {n_missing} were beyond the "
        f"{MAX_SEQ_LENGTH}-token limit and have no prediction."
        if n_missing > 0 else ""
    )

    return annotated_image, rows, truncation_note


with gr.Blocks(title="LiLT / FUNSD entity extraction") as demo:
    gr.Markdown(
        "# LiLT fine-tuned on FUNSD — entity extraction demo\n"
        "Pick a FUNSD test document to see predicted HEADER / QUESTION / ANSWER "
        "entities overlaid on the page. See `MODEL_CARD.md` for what this model "
        "does and doesn't generalize to."
    )
    with gr.Row():
        index_slider = gr.Slider(
            minimum=0, maximum=NUM_EXAMPLES - 1, step=1, value=0,
            label=f"Test document index (0-{NUM_EXAMPLES - 1})",
        )
    run_button = gr.Button("Run inference")
    with gr.Row():
        output_image = gr.Image(label="Predicted entities", type="pil")
        output_table = gr.Dataframe(
            headers=["word", "predicted_label"], label="Extracted entities (non-'O' only)"
        )
    warning_box = gr.Markdown()

    run_button.click(
        fn=run_inference,
        inputs=[index_slider],
        outputs=[output_image, output_table, warning_box],
    )
    demo.load(fn=run_inference, inputs=[index_slider], outputs=[output_image, output_table, warning_box])


if __name__ == "__main__":
    demo.launch()
