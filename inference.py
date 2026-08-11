import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from train import TransformerClassifier, preprocess_text, MAX_LENGTH, DEVICE, MODEL_SAVE_PATH, TOKENIZER_NAME, LABEL_MAP


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = TransformerClassifier(TOKENIZER_NAME, len(LABEL_MAP))
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(texts):
    """Predict sentiment for a list of texts."""
    if isinstance(texts, str):
        texts = [texts]

    preprocessed = [preprocess_text(t) for t in texts]
    encodings = tokenizer(preprocessed, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    return [LABEL_MAP[p] for p in preds]

def predict_from_csv(csv_path, text_column="text", output_path=None):
    """Run batch prediction on a CSV file."""
    df = pd.read_csv(csv_path)
    assert text_column in df.columns, f"'{text_column}' column not found in CSV."

    predictions = predict(df[text_column].tolist())
    df['predicted_sentiment'] = predictions

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment Inference")
    parser.add_argument("--text", type=str, help="Single text for prediction")
    parser.add_argument("--csv", type=str, help="Path to CSV file for batch prediction")
    parser.add_argument("--column", type=str, default="text", help="Text column in CSV (default: text)")
    parser.add_argument("--out", type=str, help="Optional output CSV file path")

    args = parser.parse_args()

    if args.text:
        result = predict(args.text)
        print(f"Prediction: {result[0]}")
    elif args.csv:
        df_result = predict_from_csv(args.csv, text_column=args.column, output_path=args.out)
        print(df_result.head())
    else:
        print("Provide either --text or --csv input.")
