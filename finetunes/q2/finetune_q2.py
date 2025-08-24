import os
import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Binary Membrane Data
# -----------------------------

DATA_PATH = "../datasets/membrane_soulable"
TRAIN_FILE = "Swissprot_Train_Validation_dataset.csv"
TEST_FILE = "hpa_testset.csv"

def load_binary_data(file_path):
    """
    Loads protein sequences and binary membrane labels from a CSV.
    Filters out short/long sequences and replaces rare amino acids with 'X'.
    """
    df = pd.read_csv(file_path).dropna(subset=["Sequence", "Cell membrane"])
    df["Sequence"] = df["Sequence"].apply(lambda x: re.sub(r"[UZOB]", "X", x))
    df = df[(df["Sequence"].str.len() >= 20) & (df["Sequence"].str.len() <= 512)]

    sequences = df["Sequence"].tolist()
    labels = df["Cell membrane"].astype(int).tolist()
    return sequences, labels

# Load and split training/validation data
print("🔹 Loading binary classification dataset...")
combine_seqs, combine_labels = load_binary_data(os.path.join(DATA_PATH, TRAIN_FILE))
train_seqs = combine_seqs[:int(len(combine_seqs) * 0.8)]
train_labels = combine_labels[:int(len(combine_labels) * 0.8)]
valid_seqs = combine_seqs[int(len(combine_seqs) * 0.8):]
valid_labels = combine_labels[int(len(combine_labels) * 0.8):]
print(f"🔹 Loaded {len(train_seqs)} training samples and {len(valid_seqs)} validation samples.")

# Load test data
test_seqs, test_labels = load_binary_data(os.path.join(DATA_PATH, TEST_FILE))
print(f"Loaded {len(train_seqs)} training samples, {len(test_seqs)} test samples")

# -----------------------------
# 2. Tokenization + Dataset
# -----------------------------

MODEL_NAME = "Rostlab/prot_bert"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

class BinaryCLSProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Define training arguments
training_args = TrainingArguments(
    output_dir="../savings/results/q2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../savings/logs/q2",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=BinaryCLSProteinDataset(train_seqs, train_labels, TOKENIZER),
    eval_dataset=BinaryCLSProteinDataset(valid_seqs, valid_labels, TOKENIZER),
    tokenizer=TOKENIZER,
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))
    }
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(BinaryCLSProteinDataset(test_seqs, test_labels, TOKENIZER))
print("Test Results:", test_results)
