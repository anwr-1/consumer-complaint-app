# 📋 Consumer Complaint Classification

Predicts the correct category for a customer complaint narrative. Compares
SimpleRNN, LSTM, and GRU models built from scratch against a fine-tuned
HuggingFace transformer, and deploys the best-performing model.

## 🔗 Links

- **Training notebook (Kaggle):** [consumer-complaint](https://www.kaggle.com/code/anwernasr/consumer-complaint)

## Running the app

```bash
pip install -r requirements.txt
python app.py
```

Place your fine-tuned model + tokenizer in a folder named
`consumer-complaint-model/` next to `app.py` (the format HuggingFace's
`save_pretrained()` produces) before running.

## Project structure

```
consumer-complaint-app/
├── app.py
├── requirements.txt
└── README.md
```

## License

MIT
