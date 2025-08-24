import os
import time
import gc
import torch
import argparse
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)
from torch.nn.functional import cross_entropy, cosine_similarity
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, concatenate_datasets
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------------------------------
#  Distributed-training setup
# -------------------------------------------------

if "LOCAL_RANK" not in os.environ:
    raise ValueError("LOCAL_RANK is not set. Please run using torchrun --nproc_per_node=X ...")

local_rank = int(os.environ["LOCAL_RANK"])

# Initialise NCCL backend and pin each process to its GPU
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# -------------------------------------------------
# 1. Cache directories for HuggingFace datasets/models
# -------------------------------------------------
cache_dir = "../datasets_cache"
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
os.makedirs(cache_dir, exist_ok=True)

scaler = GradScaler()  

if local_rank == 0:
    print(f" Cache directory set to: {cache_dir}")
    assert torch.cuda.is_available(), "CUDA GPU is not available."

# Load ProtBERT tokenizer (shared by student & teacher)
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

# -------------------------------------------------
# 2. Utility: timing helper
# -------------------------------------------------
def log_time(label, t0):
    t1 = time.time()
    if local_rank == 0:
        print(f"⏱️ {label}: {t1 - t0:.3f} sec")
    return t1

# -------------------------------------------------
# 3. Dataset loading from .arrow shards
# -------------------------------------------------
def load_arrow_files(directory, max_files=0, sample_size=50000000):
    arrow_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('171.arrow')],
        key=lambda f: os.path.getsize(f)
    )

    if local_rank == 0:
        print(f"🔹 Found {len(arrow_files)} .arrow files.")

    # Concatenate all shards into one HF Dataset
    dataset = Dataset.from_file(arrow_files[0])
    for file in arrow_files[1:]:
        dataset = concatenate_datasets([dataset, Dataset.from_file(file)])

    if local_rank == 0:
        print(f" Final dataset loaded with {len(dataset)} samples.")

    # Optional random sub-sampling for quicker runs
    if sample_size > 0 and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        if local_rank == 0:
            print(f" Sampled {sample_size} samples.")

    return dataset

# -------------------------------------------------
# 4. Dataloader builder for already-tokenized dataset
# -------------------------------------------------
def prepare_dataloader_from_tokenized(tokenized_dataset, batch_size=8):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=8
    )
    sampler = DistributedSampler(tokenized_dataset)  # ensures distinct shards per GPU
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

# -------------------------------------------------
# 5. Loss functions
# -------------------------------------------------
def mlm_loss(logits, labels):
    """Standard masked-LM loss; ignore padding label -100."""
    return cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL-divergence between teacher and student distributions."""
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    return -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

def cosine_loss(student_embeddings, teacher_embeddings):
    """1 – cosine similarity over last-layer embeddings."""
    return 1 - torch.nn.functional.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1).mean()

# -------------------------------------------------
# 6. Training loop with knowledge-distillation
# -------------------------------------------------
def pretrain_minprotbert_with_distillation(
        student_model,
        teacher_model,
        dataloader,
        num_epochs=3,
        learning_rate=5e-5,
        temperature=2.0
    ):
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    scaler = GradScaler()  

    student_model.train()
    teacher_model.eval()

    for epoch in range(num_epochs):
        if local_rank == 0:
            print(f"\n🔹 Starting epoch {epoch+1}/{num_epochs} at {time.strftime('%H:%M:%S')}")
        epoch_loss = 0
        start_time = time.time()

        dataloader.sampler.set_epoch(epoch)  # shuffle per epoch

        for step, batch in enumerate(dataloader):
            t0 = time.time()

            if local_rank == 0 and step % 5 == 0:
                print(f"🔹 Processing batch {step+1}/{len(dataloader)} at {time.strftime('%H:%M:%S')}")

            # Move batch to GPU
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = (inputs != tokenizer.pad_token_id).to(device)
            t0 = log_time("Batch preparation + move to GPU", t0)

            optimizer.zero_grad()

            # ----- Student forward pass (with gradients) -----
            with autocast():
                student_outputs = student_model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                student_logits = student_outputs.logits
                student_embeddings = student_outputs.hidden_states[-1]
            t0 = log_time("Student forward", t0)

            # ----- Teacher forward pass (no gradients) -----
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                teacher_logits = teacher_outputs.logits
                teacher_embeddings = teacher_outputs.hidden_states[-1]
            t0 = log_time("Teacher forward", t0)

            # ----- Loss computation -----
            loss_mlm     = mlm_loss(student_logits, labels)
            loss_distill = distillation_loss(student_logits, teacher_logits, temperature)
            loss_cosine  = cosine_loss(student_embeddings, teacher_embeddings)
            total_loss   = loss_mlm + 0.5 * loss_distill + 0.1 * loss_cosine
            t0 = log_time("Loss calculation", t0)

            # ----- Back-propagation -----
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t0 = log_time("Backward + optimizer step", t0)

            epoch_loss += total_loss.item()
            gc.collect()

            # Progress log every 100 steps
            if local_rank == 0 and (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (step + 1)) * (len(dataloader) - (step + 1))
                print(f" Step {step+1}/{len(dataloader)}, Loss: {total_loss.item():.4f}, ETA: {eta/60:.1f} min")

        # End-of-epoch summary (only rank 0)
        if local_rank == 0:
            print(f" Epoch {epoch+1} done in {time.time() - start_time:.2f}s. "
                  f"Avg Loss: {epoch_loss / len(dataloader):.4f}")
            save_path = f"../savings/model/minprotbert_epoch{epoch+1}"
            student_model.module.save_pretrained(save_path)
            print(f" Model saved at {save_path}")

# -------------------------------------------------
# 7. MAIN SCRIPT
# -------------------------------------------------
gc.collect()

# ---- Dataset loading ----
if local_rank == 0:
    print("\n🔹 Loading tokenized data from arrow files...")
directory = '../datasets/uniref50_tokenized/train'
tokenized_dataset = load_arrow_files(directory, max_files=0)  

if tokenized_dataset is None:
    if local_rank == 0:
        print(" Failed to load dataset from arrow files. Exiting.")
    exit()

tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# ---- Dataloader ----
if local_rank == 0:
    print("\n🔹 Preparing dataloader...")
dataloader = prepare_dataloader_from_tokenized(tokenized_dataset, batch_size=8)
if local_rank == 0:
    print(f" Dataloader ready with {len(dataloader)} batches")

# ---- Load student & teacher models ----
if local_rank == 0:
    print("\n🔹 Loading models...")
student_model = AutoModelForMaskedLM.from_pretrained("../savings/model/minprotbert_model")
student_model.to(device)
student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)  # wrap for multi-GPU

teacher_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert").half().to(device).eval()

if local_rank == 0:
    print(" Models loaded")
    print("\n Starting pretraining...")

# ---- Pretraining with distillation ----
pretrain_minprotbert_with_distillation(student_model, teacher_model, dataloader, num_epochs=3)

# ---- Clean-up distributed resources ----
dist.destroy_process_group()

if local_rank == 0:
    print(" Pretraining complete!")
