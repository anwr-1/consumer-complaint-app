import gradio as gr
from transformers import pipeline

MODEL_DIR = "consumer-complaint-model"  # folder with your fine-tuned HuggingFace model + tokenizer

classifier = pipeline("text-classification", model=MODEL_DIR, top_k=None)


def classify_complaint(text):
    if not text or not text.strip():
        return {}
    results = classifier(text)[0]  # list of {"label": ..., "score": ...} for every class
    return {r["label"]: float(r["score"]) for r in results}


demo = gr.Interface(
    fn=classify_complaint,
    inputs=gr.Textbox(lines=6, placeholder="Paste a customer complaint here...", label="Complaint text"),
    outputs=gr.Label(num_top_classes=5, label="Predicted category"),
    title="📋 Consumer Complaint Classifier",
    description=(
        "Enter a customer complaint narrative and this app predicts the correct "
        "complaint category using a fine-tuned transformer, along with a confidence score."
    ),
    examples=[
        ["I was charged twice for the same transaction and customer service won't refund me."],
        ["My credit report shows an account that isn't mine and I can't get it removed."],
    ],
)

if __name__ == "__main__":
    demo.launch()
