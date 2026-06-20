"""ch07 instruction finetuning on an AMD GPU via DirectML.

Adapts LLMs-from-scratch Chapter 7 (instruction-following finetuning on the
Alpaca-style instruction dataset) to run on AMD Radeon GPUs that are NOT on the
ROCm WSL support matrix, using torch-directml.

What it does (same as the chapter):
  1. Load the bundled instruction-data.json (1100 entries, no download needed)
  2. Load pretrained GPT-2 small (124M) weights from the cached .pt
  3. Full-parameter finetune 2 epochs with the instruction/response format
  4. Generate responses on a few test instructions to show instruction-following

Model size note:
  The book uses gpt2-medium (355M) for ch07. Full-parameter finetuning of 355M
  (weights + grads + Adam states in fp32) needs ~6GB+ just for optimizer state,
  plus activations -- too tight for an 8GB card via DirectML. So we use
  gpt2-small (124M) here. On a >=16GB card you can switch CHOOSE_MODEL to 355M
  (and cache the 355M weights separately).

Reuses ch06's cached GPT-2 weights (gpt2-124M-params.pt) -- same small weights.
Reuses ch07/01 previous_chapters.py (GPTModel, load_weights_into_gpt, generate,
train_model_simple, ...). InstructionDataset / custom_collate_fn / format_input
are inlined here to avoid importing gpt_instruction_finetuning.py which pulls in
gpt_download -> tensorflow.

Run (native Windows python with torch + torch-directml + tiktoken + matplotlib):
    python train_instruction_ch07.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import partial

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CH07_DIR = os.path.join(REPO_ROOT, "ch07", "01_main-chapter-code")

sys.path.insert(0, HERE)        # for dml_device
sys.path.insert(0, CH07_DIR)    # for previous_chapters

from dml_device import pick_device  # noqa: E402
from previous_chapters import (  # noqa: E402
    GPTModel,
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_loader,
    train_model_simple,
)

# Reuse ch06's cached 124M weights.
WEIGHTS_CACHE = os.path.join(HERE, "gpt2-124M-params.pt")
DATA_FILE = os.path.join(CH07_DIR, "instruction-data.json")
OUT_DIR = os.path.join(HERE, "data")

EPOCHS = 2
BATCH = 4              # reduced from 8 to fit 8GB VRAM (full 124M finetune)
LR = 5e-5
EVAL_FREQ = 5
EVAL_ITER = 5
ALLOWED_MAX_LENGTH = 512   # 512 covers most instruction examples; caps worst-case activation
GENERATE_N = 3          # generate responses for the first N test entries


# ---------------------------------------------------------------------------
# Instruction dataset + collate (faithful inline copy of ch07/01
# gpt_instruction_finetuning.py, inlined to avoid importing gpt_download->TF).
# ---------------------------------------------------------------------------
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def main():
    dev, name = pick_device()
    print(f"\nInstruction-finetuning GPT-2 (124M) on: {name}  ({dev})\n", flush=True)

    # --- 1. Instruction data (bundled, no download) ---
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print(f"train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}",
          flush=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    collate = partial(custom_collate_fn, device=dev,
                      allowed_max_length=ALLOWED_MAX_LENGTH)

    torch.manual_seed(123)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                              batch_size=BATCH, collate_fn=collate,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer),
                            batch_size=BATCH, collate_fn=collate,
                            shuffle=False, drop_last=False)

    # --- 2. Pretrained GPT-2 124M from cache ---
    if not os.path.exists(WEIGHTS_CACHE):
        raise SystemExit(
            f"Weights cache not found: {WEIGHTS_CACHE}\n"
            "Run the one-time cache step in dml-adapt/README.md first."
        )
    print(f"Loading cached GPT-2 weights: {WEIGHTS_CACHE}", flush=True)
    cached = torch.load(WEIGHTS_CACHE, map_location="cpu", weights_only=False)
    params = cached["params"]

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(dev)
    model.eval()

    print("Initial losses", flush=True)
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, dev, num_batches=EVAL_ITER)
        va = calc_loss_loader(val_loader, model, dev, num_batches=EVAL_ITER)
    print(f"   Training loss: {tr:.4f}\n   Validation loss: {va:.4f}", flush=True)

    # --- 3. Full-parameter finetune ---
    t0 = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    train_losses, val_losses, _ = train_model_simple(
        model, train_loader, val_loader, optimizer, dev,
        num_epochs=EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
        start_context=format_input(val_data[0]), tokenizer=tokenizer,
    )
    print(f"\nTraining completed in {(time.time() - t0) / 60:.2f} minutes.", flush=True)

    # --- 4. Generate responses on a few test instructions ---
    print(f"\nGenerating responses for first {GENERATE_N} test entries:", flush=True)
    model.eval()
    for entry in test_data[:GENERATE_N]:
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(dev),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generated = token_ids_to_text(token_ids, tokenizer)
        response = generated[len(input_text):].replace("### Response:", "").strip()
        print(f"\n- Input:    {entry['instruction']}", flush=True)
        if entry.get("input"):
            print(f"  Input(f): {entry['input']}", flush=True)
        print(f"  Expected: {entry['output']}", flush=True)
        print(f"  Model:    {response[:300]}", flush=True)

    print("\nCH07 INSTRUCTION FINETUNE ON AMD GPU (DirectML): OK")


if __name__ == "__main__":
    main()
