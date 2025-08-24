import os
import torch
import re
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Data
# -----------------------------
DATA_PATH = "../datasets/secondary_structure"
TRAIN_FILE = "training_secondary_structure_train.csv"
VALID_FILE = "validation_secondary_structure_valid.csv"
TEST_FILES = [
    "test_secondary_structure_casp12.csv",
    "test_secondary_structure_cb513.csv",
    "test_secondary_structure_ts115.csv"
]

# Label mappings for secondary structure classification
LABEL_MAP = {"H": 0, "E": 1, "C": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_data(file_path):
    """Load sequences and secondary structure labels from a CSV file."""
    df = pd.read_csv(file_path).dropna()
    df["seq"] = df["seq"].apply(lambda x: re.sub(r"[UZOB]", "X", x))  # Replace uncommon amino acids with 'X'
    df = df[(df["seq"].str.len() >= 20) & (df["seq"].str.len() <= 512)]  # Filter extreme lengths
    labels = [[LABEL_MAP[c] for c in s] for s in df["sst3"].astype(str)]  # Convert structure chars to int labels
    sequences = df["seq"].tolist()
    return sequences, labels

# Load datasets
train_seqs, train_labels = load_data(os.path.join(DATA_PATH, TRAIN_FILE))
valid_seqs, valid_labels = load_data(os.path.join(DATA_PATH, VALID_FILE))
test_seqs, test_labels = [], []
for test_file in TEST_FILES:
    seqs, labels = load_data(os.path.join(DATA_PATH, test_file))
    test_seqs.extend(seqs)
    test_labels.extend(labels)

print(f"🔹 Loaded {len(train_seqs)} training samples, {len(valid_seqs)} validation samples, and {len(test_seqs)} test samples.")

# -----------------------------
# 2. Tokenization & Dataset Class
# -----------------------------
TOKENIZER = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

class SecondaryStructureDataset(Dataset):
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
    output_dir="../savings/results/q3",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../savings/logs/q3",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Initialize model
model = AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert", num_labels=3)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=SecondaryStructureDataset(train_seqs, train_labels, TOKENIZER),
    eval_dataset=SecondaryStructureDataset(valid_seqs, valid_labels, TOKENIZER),
    tokenizer=TOKENIZER,
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))
    }
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(SecondaryStructureDataset(test_seqs, test_labels, TOKENIZER))
print("Test Results:", test_results)
