import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold
import os

# -----------------------------------
# 1. Dataset Class
# -----------------------------------

class DNABindingDataset(Dataset):
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
        spaced_seq = " ".join(seq)  # Add spaces between amino acids

        encoded = self.tokenizer(
            spaced_seq,
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

# -----------------------------------
# 2. Cross-Validation Training
# -----------------------------------

def cross_validate_model(data_dir, output_dir, model_name="yarongef/DistilProtBert", num_folds=5):
    df = pd.read_csv(os.path.join(data_dir, "combined.csv"))
    sequences = df["Sequence"].tolist()
    labels = df["binding"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold = 1
    for train_idx, val_idx in kf.split(sequences):
        print(f"🔹 Starting fold {fold}/{num_folds}")

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = DNABindingDataset(train_seqs, train_labels, tokenizer)
        val_dataset = DNABindingDataset(val_seqs, val_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"fold_{fold}"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, f"logs/fold_{fold}"),
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))
            }
        )

        trainer.train()
        fold += 1

# -----------------------------------
# 3. Main Execution
# -----------------------------------

if __name__ == "__main__":
    DATA_DIR = "../datasets/train_DNA_binding"
    OUTPUT_DIR = "../savings/results/dna_binding"

    cross_validate_model(DATA_DIR, OUTPUT_DIR)
