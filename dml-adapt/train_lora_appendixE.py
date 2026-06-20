"""LoRA (Appendix E) finetuning on an AMD GPU via DirectML.

Adapts LLMs-from-scratch Appendix E (LoRA — Low-Rank Adaptation) to run on AMD
Radeon GPUs via torch-directml. Applies LoRA to the ch06 SMS spam classification
task (the appendix's own canonical example).

Why LoRA matters on an 8GB card:
  Full-parameter finetuning of GPT-2 needs weights + gradients + Adam states in
  memory (for 124M that's ~2GB, for 355M ~6GB) -- ch07 showed 355M OOMs an 8GB
  card. LoRA freezes the base model and trains only tiny low-rank adapters
  (rank=16 here), cutting trainable params from ~124M to a few hundred thousand
  and slashing optimizer memory. Same pattern unlocks 355M+ on small GPUs.

What it does (same as the appendix):
  1. Load pretrained GPT-2 small (124M) from the cached .pt (reuse ch06 cache)
  2. Replace the output head with a 2-class Linear (classification)
  3. Freeze ALL base params
  4. Replace every Linear with LinearWithLoRA (rank=16, alpha=16) -- only the
     LoRA A/B matrices become trainable
  5. Finetune 5 epochs, report trainable-param count + accuracy

Run (native Windows python with torch + torch-directml + tiktoken + pandas):
    python train_lora_appendixE.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
APX_DIR = os.path.join(REPO_ROOT, "appendix-E", "01_main-chapter-code")

sys.path.insert(0, HERE)     # for dml_device
sys.path.insert(0, APX_DIR)  # for previous_chapters (GPTModel, load_weights_into_gpt)

from dml_device import pick_device  # noqa: E402
from previous_chapters import GPTModel, load_weights_into_gpt  # noqa: E402

WEIGHTS_CACHE = os.path.join(HERE, "gpt2-124M-params.pt")
DATA_DIR = os.path.join(HERE, "data")

RANK = 16
ALPHA = 16
EPOCHS = 5
BATCH = 8
LR = 4e-5
EVAL_FREQ = 50
EVAL_ITER = 5


# ---------------------------------------------------------------------------
# LoRA (faithful inline copy of Appendix E notebook classes).
# ---------------------------------------------------------------------------
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # B starts at 0
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


# ---------------------------------------------------------------------------
# Spam dataset prep (faithful inline copy, avoids extra module imports).
# ---------------------------------------------------------------------------
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        return
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.rename(Path(extracted_path) / "SMSSpamCollection", data_file_path)


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(t) for t in self.data["Text"]]
        if max_length is None:
            self.max_length = max(len(e) for e in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [e[:self.max_length] for e in self.encoded_texts]
        self.encoded_texts = [
            e + [pad_token_id] * (self.max_length - len(e)) for e in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Loss / accuracy / training (device = DirectML).
# ---------------------------------------------------------------------------
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, n = 0, 0
    nb = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= nb:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            logits = model(inp)[:, -1, :]
        pred = torch.argmax(logits, dim=-1)
        n += pred.shape[0]
        correct += (pred == tgt).sum().item()
    return correct / n


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches):
    if len(data_loader) == 0:
        return float("nan")
    total, nb = 0.0, min(num_batches, len(data_loader))
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= nb:
            break
        with torch.no_grad():
            total += calc_loss_batch(inp, tgt, model, device).item()
    return total / nb


def main():
    dev, name = pick_device()
    print(f"\nLoRA finetuning GPT-2 (124M) for spam classification on: {name}  ({dev})\n",
          flush=True)

    # --- 1. Spam data ---
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "sms_spam_collection.zip")
    extracted = os.path.join(DATA_DIR, "sms_spam_collection")
    data_file = Path(extracted) / "SMSSpamCollection.tsv"
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    try:
        download_and_unzip_spam_data(url, zip_path, extracted, data_file)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...", flush=True)
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted, data_file)

    df = pd.read_csv(data_file, sep="\t", header=None, names=["Label", "Text"])
    balanced = create_balanced_dataset(df)
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
    train_df, val_df, test_df = random_split(balanced, 0.7, 0.1)
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv = os.path.join(DATA_DIR, "validation.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_csv, index=None)
    val_df.to_csv(val_csv, index=None)
    test_df.to_csv(test_csv, index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_ds = SpamDataset(train_csv, tokenizer, max_length=None)
    val_ds = SpamDataset(val_csv, tokenizer, max_length=train_ds.max_length)
    test_ds = SpamDataset(test_csv, tokenizer, max_length=train_ds.max_length)

    torch.manual_seed(123)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    # --- 2. Pretrained GPT-2 124M from cache ---
    if not os.path.exists(WEIGHTS_CACHE):
        raise SystemExit(f"Weights cache not found: {WEIGHTS_CACHE} (see dml-adapt/README.md)")
    print(f"Loading cached GPT-2 weights: {WEIGHTS_CACHE}", flush=True)
    cached = torch.load(WEIGHTS_CACHE, map_location="cpu", weights_only=False)
    params = cached["params"]

    BASE_CONFIG = {
        "vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0,
        "qkv_bias": True, "emb_dim": 768, "n_layers": 12, "n_heads": 12,
    }
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    # --- 3. Classification head + freeze all + LoRA replace ---
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)
    for p in model.parameters():
        p.requires_grad = False              # freeze everything
    replace_linear_with_lora(model, rank=RANK, alpha=ALPHA)  # LoRA A/B auto-trainable
    model.to(dev)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"rank={RANK} alpha={ALPHA} | total params: {total/1e6:.1f}M | "
          f"trainable (LoRA only): {trainable/1e3:.1f}K "
          f"({100*trainable/total:.3f}% of total)\n", flush=True)

    # --- 4. Finetune only LoRA params ---
    t0 = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.1
    )
    print(f"train batches: {len(train_loader)} | max_len: {train_ds.max_length}\n", flush=True)
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        ep_t = time.time()
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, dev)
            loss.backward()
            optimizer.step()
            step += 1
            if step % EVAL_FREQ == 0:
                model.eval()
                tr = calc_loss_loader(train_loader, model, dev, EVAL_ITER)
                va = calc_loss_loader(val_loader, model, dev, EVAL_ITER)
                model.train()
                print(f"Ep {epoch+1} (Step {step:04d}): Train loss {tr:.3f}, Val loss {va:.3f}",
                      flush=True)
        tr_acc = calc_accuracy_loader(train_loader, model, dev, EVAL_ITER)
        va_acc = calc_accuracy_loader(val_loader, model, dev, EVAL_ITER)
        print(f"  epoch {epoch+1}/{EPOCHS} done in {time.time()-ep_t:.1f}s | "
              f"Train acc {tr_acc*100:.2f}% | Val acc {va_acc*100:.2f}%", flush=True)

    test_acc = calc_accuracy_loader(test_loader, model, dev)
    print(f"\nTraining completed in {(time.time()-t0)/60:.2f} min")
    print(f"Test accuracy: {test_acc*100:.2f}%  (full finetune baseline ~95.67%)")
    print("APPENDIX-E LoRA FINETUNE ON AMD GPU (DirectML): OK")


if __name__ == "__main__":
    main()
