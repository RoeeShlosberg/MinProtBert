import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import pandas as pd
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
# 2. Load Data
# -----------------------------------

def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, val_df, test_df

# -----------------------------------
# 3. Fine-Tuning Function
# -----------------------------------

def fine_tune_model(data_dir, output_dir, model_name="yarongef/DistilProtBert"):
    train_df, val_df, test_df = load_data(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = DNABindingDataset(
        train_df["Sequence"].tolist(),
        train_df["binding"].astype(int).tolist(),
        tokenizer
    )
    val_dataset = DNABindingDataset(
        val_df["Sequence"].tolist(),
        val_df["binding"].astype(int).tolist(),
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
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
            "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
        }
    )

    trainer.train()

# -----------------------------------
# 4. Main Execution
# -----------------------------------

if __name__ == "__main__":
    DATA_DIR = "../datasets/original_dna_binding"
    OUTPUT_DIR = "../savings/results/dna_binding"

    fine_tune_model(DATA_DIR, OUTPUT_DIR)
