import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import re
import string
from torch.cuda.amp import GradScaler, autocast

DATASET_PATH = './combined_sentiment_data.csv'  
PREPROCESSED_DATA_PATH = './preprocessed_data.arrow'
MODEL_SAVE_PATH = './transformer_sentiment_model.pth'
TOKENIZER_NAME = 'xlm-roberta-base'
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.001
EARLY_STOPPING_PATIENCE = 2
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "punkt" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def plot_confusion_matrix(y_true, y_pred, classes, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

def plot_metrics(history, out_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics.png'))
    plt.close()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len,
                             padding='max_length', truncation=True,
                             return_attention_mask=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}

class TransformerClassifier(nn.Module):
    def __init__(self, pretrained, n_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    scaler = torch.amp.GradScaler()
    losses = []
    correct = 0
    for batch in tqdm(loader, desc='Train'):
        ids, masks, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(input_ids=ids, attention_mask=masks)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        correct += torch.sum(preds == labels)
    return correct.double() / len(loader.dataset), np.mean(losses)

def eval_model(model, loader, loss_fn):
    model.eval()
    losses = []
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            ids, masks, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            outputs = model(input_ids=ids, attention_mask=masks)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return correct.double() / len(loader.dataset), np.mean(losses), all_labels, all_preds

def main():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {DEVICE}")

    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

 
    if os.path.exists(PREPROCESSED_DATA_PATH):
        print("Loading preprocessed data...")
        df = pd.read_feather(PREPROCESSED_DATA_PATH)
    else:
        print("Preprocessing raw data...")
        df = pd.read_csv(DATASET_PATH)
        df.dropna(subset=['text', 'label'], inplace=True)
        label_remap = {-1: 0, 0: 1, 1: 2}
        df['label'] = df['label'].map(label_remap)
        df['text'] = df['text'].apply(preprocess_text)
        df = df[df['text'].str.strip() != '']
        df = df[df['label'].isin(label_remap.values())] # Ensure labels are valid
        df.reset_index(drop=True, inplace=True)
        df.to_feather(PREPROCESSED_DATA_PATH)
        print("Preprocessed data saved to:", PREPROCESSED_DATA_PATH)

 
    cw = compute_class_weight('balanced', classes=np.arange(NUM_LABELS), y=df['label'])
    class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)

    train_ds = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LENGTH)
    val_ds = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TransformerClassifier(TOKENIZER_NAME, NUM_LABELS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * NUM_EPOCHS)

    best_acc = 0
    no_improve = 0
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

 
    for epoch in range(1, NUM_EPOCHS + 1):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_acc, val_loss, val_labels, val_preds = eval_model(model, val_loader, loss_fn)

        history['train_acc'].append(train_acc.cpu())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.cpu())
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break

    plot_metrics(history, OUTPUT_DIR)
    plot_confusion_matrix(val_labels, val_preds, classes=list(LABEL_MAP.values()), out_dir=OUTPUT_DIR)

    print("\nFinal Report:\n", classification_report(val_labels, val_preds, target_names=list(LABEL_MAP.values()), digits=4))
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(val_labels, val_preds, target_names=list(LABEL_MAP.values()), digits=4))
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
        f.write(f"Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Validation Loss: {history['val_loss'][-1]:.4f}\n")

if __name__ == '__main__':
    main()
    print("Training complete. Model saved to:", MODEL_SAVE_PATH)
    print("Metrics and confusion matrix saved to:", OUTPUT_DIR)
