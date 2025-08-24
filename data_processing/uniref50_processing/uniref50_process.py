import os
import re
import subprocess
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# Constants
UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
UNIREF50_GZ = "../datasets/uniref50/uniref50.fasta.gz"
UNIREF50_FASTA = "../datasets/uniref50/uniref50.fasta"
UNIREF50_PROCESSED = "../datasets/uniref50/uniref50_20_512_oneliner_noheader.fasta"
SAVE_PATH = "../datasets/uniref50/train/"

# Functions
def download_and_prepare_uniref50():
    # Download UniRef50
    if not os.path.exists(UNIREF50_GZ):
        subprocess.run(["wget", UNIREF50_URL, "-O", UNIREF50_GZ], check=True)

    # Extract the .gz file
    if not os.path.exists(UNIREF50_FASTA):
        subprocess.run(["gzip", "-dk", UNIREF50_GZ], check=True)

    # Filter sequences by length using seqkit
    subprocess.run(["seqkit", "seq", "-M", "512", UNIREF50_FASTA, "-o", "../datasets/uniref50/uniref50_512.fasta"], check=True)
    subprocess.run(["seqkit", "seq", "-m", "20", "../datasets/uniref50/uniref50_512.fasta", "-o", "../datasets/uniref50/uniref50_20_512.fasta"], check=True)
    subprocess.run(["seqkit", "seq", "../datasets/uniref50/uniref50_20_512.fasta", "-w", "0", "-o", "../datasets/uniref50/uniref50_20_512_oneliner.fasta"], check=True)
    subprocess.run(["grep", "-v", ">", "../datasets/uniref50/uniref50_20_512_oneliner.fasta", "-o", UNIREF50_PROCESSED], check=True)

def preprocess_dataset(dataset_path):
    uniref50 = load_dataset("text", data_files=[dataset_path])
    uniref50 = uniref50.shuffle(seed=42)
    uniref50_processed_ds = uniref50.map(lambda example: {
        'Seqs': re.sub(r"[UZOB]", "X", " ".join(example['text'])),
        'length': len(example['text'])
    })
    return uniref50_processed_ds

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    result = tokenizer(examples["Seqs"], add_special_tokens=True, return_special_tokens_mask=True)
    return result

# Main Script
if __name__ == "__main__":
    # Step 1: Download and prepare UniRef50
    download_and_prepare_uniref50()

    # Step 2: Preprocess dataset
    if not os.path.exists(UNIREF50_PROCESSED):
        raise FileNotFoundError(f"Processed dataset not found at {UNIREF50_PROCESSED}.")

    uniref50_processed = preprocess_dataset(UNIREF50_PROCESSED)
    uniref50_tokenized = uniref50_processed.map(tokenize_function, batched=True, remove_columns=["text", "Seqs"])

    # Step 3: Save tokenized dataset
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    uniref50_tokenized.save_to_disk(SAVE_PATH)

    # Step 4: Load and print dataset
    uniref50_ds = load_from_disk(SAVE_PATH)
    print(uniref50_ds)