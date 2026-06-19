"""ch06 classification finetuning on an AMD GPU via DirectML.

Adapts LLMs-from-scratch Chapter 6 (SMS spam classification) to run on AMD
Radeon GPUs that are NOT on the ROCm WSL support matrix, using torch-directml.

What it does (same as the chapter):
  1. Download + balance + split the SMS spam dataset
  2. Load pretrained GPT-2 small (124M) weights
  3. Freeze all layers, replace the output head with a 2-class Linear,
     unfreeze the last transformer block + final norm
  4. Finetune 5 epochs on the AMD GPU, report train/val accuracy

Weight-loading trick (keeps the dml env tensorflow-free):
  gpt_download.py needs tensorflow to parse OpenAI's checkpoint. Rather than
  install tensorflow next to torch-directml (coexistence risk), we pre-cache the
  parsed params dict once (see cache step in README) and load it via
  torch.load + load_weights_into_gpt here -- no tensorflow needed.

Run (native Windows python with torch + torch-directml + tiktoken + pandas):
    python train_classify_ch06.py
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CH06_DIR = os.path.join(REPO_ROOT, "ch06", "01_main-chapter-code")

sys.path.insert(0, HERE)        # for dml_device
sys.path.insert(0, CH06_DIR)    # for previous_chapters (GPTModel, load_weights_into_gpt)

from dml_device import pick_device  # noqa: E402
from previous_chapters import GPTModel, load_weights_into_gpt  # noqa: E402

# Cached pretrained weights (produced by the one-time cache step in README).
WEIGHTS_CACHE = os.path.join(HERE, "gpt2-124M-params.pt")
# Local data dir for the spam csv files.
DATA_DIR = os.path.join(HERE, "data")

EPOCHS = 5
BATCH = 8
LR = 5e-5
EVAL_FREQ = 50
EVAL_ITER = 5


# ---------------------------------------------------------------------------
# Spam dataset prep (faithful inline copy of ch06/01 gpt_class_finetune.py,
# inlined to avoid importing that module which pulls in matplotlib/tensorflow).
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
    original = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original, data_file_path)


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
            self.max_length = self._longest_encoded_length()
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

    def _longest_encoded_length(self):
        return max(len(e) for e in self.encoded_texts)


# ---------------------------------------------------------------------------
# Loss / accuracy / training (faithful inline copy, device = DirectML).
# ---------------------------------------------------------------------------
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, n = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= num_batches:
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


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(inp, tgt, model, device).item()
    return total / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, device, eval_iter)
        va = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return tr, va


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        ep_t = time.time()
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += inp.shape[0]
            global_step += 1
            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr)
                val_losses.append(va)
                print(f"Ep {epoch+1} (Step {global_step:04d}): "
                      f"Train loss {tr:.3f}, Val loss {va:.3f}", flush=True)
        train_acc = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, eval_iter)
        print(f"  epoch {epoch+1}/{num_epochs} done in {time.time()-ep_t:.1f}s | "
              f"Train acc {train_acc*100:.2f}% | Val acc {val_acc*100:.2f}%", flush=True)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def main():
    dev, name = pick_device()
    print(f"\nFinetuning GPT-2 (124M) for spam classification on: {name}  ({dev})\n",
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
        raise SystemExit(
            f"Weights cache not found: {WEIGHTS_CACHE}\n"
            "Run the one-time cache step (see dml-adapt/README.md) using a python "
            "that has tensorflow, e.g.:\n"
            "  python -c \"import sys; sys.path.insert(0, 'CH06_DIR'); "
            "from gpt_download import download_and_load_gpt2; "
            "import torch; s,p=download_and_load_gpt2('124M','gpt2'); "
            "torch.save({'settings':s,'params':p}, '" + WEIGHTS_CACHE.replace("\\", "/") + "')\""
        )
    print(f"Loading cached GPT-2 weights: {WEIGHTS_CACHE}", flush=True)
    cached = torch.load(WEIGHTS_CACHE, map_location="cpu", weights_only=False)
    settings, params = cached["settings"], cached["params"]

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }
    assert train_ds.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_ds.max_length} exceeds context "
        f"{BASE_CONFIG['context_length']}"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    # --- 3. Replace head, freeze, partial unfreeze ---
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)
    model.to(dev)
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True

    # --- 4. Finetune ---
    t0 = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    print(f"train batches: {len(train_loader)} | max_len: {train_ds.max_length} | "
          f"epochs: {EPOCHS}\n", flush=True)
    train_classifier_simple(model, train_loader, val_loader, optimizer, dev,
                            EPOCHS, EVAL_FREQ, EVAL_ITER)

    test_acc = calc_accuracy_loader(test_loader, model, dev)
    print(f"\nTraining completed in {(time.time()-t0)/60:.2f} min")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print("CH06 CLASSIFICATION FINETUNE ON AMD GPU (DirectML): OK")


if __name__ == "__main__":
    main()
