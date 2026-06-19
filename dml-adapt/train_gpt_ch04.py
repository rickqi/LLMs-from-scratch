"""Train the ch04 GPT model on an AMD GPU via DirectML.

Adaptation of LLMs-from-scratch ch04/ch05 to run on AMD Radeon GPUs that are NOT
on the ROCm WSL support matrix (e.g. Radeon RX 7600M XT / Navi 33). Instead of
`torch.device("cuda")`, it uses torch-directml through `dml_device.pick_device()`.

This is the "quick training verification" variant (5 epochs). For the fuller
ch05-style pretraining with periodic text generation, see
`train_pretrain_ch05.py`.

It reuses this repo's own model + data (no code duplication):
  - model:  ch04/01_main-chapter-code/gpt.py  (GPTModel, 124M config)
  - data:   ch02/01_main-chapter-code/the-verdict.txt

REQUIREMENTS (native Windows python, NOT WSL python):
  - conda env with: torch (cpu build is fine), torch-directml, tiktoken
  - see dml-adapt/README.md for the one-time setup

Run (from WSL, via the run.sh bridge, or directly in a Windows conda env):
    python train_gpt_ch04.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import tiktoken

# Locate sibling/parent paths relative to this file so CWD doesn't matter.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Import this repo's own ch04 GPTModel (gpt.py), not a copy.
sys.path.insert(0, os.path.join(REPO_ROOT, "ch04", "01_main-chapter-code"))
from gpt import (  # noqa: E402
    GPTModel,
    create_dataloader_v1,
    generate_text_simple,
)

from dml_device import pick_device  # noqa: E402

DATA_PATH = os.path.join(REPO_ROOT, "ch02", "01_main-chapter-code", "the-verdict.txt")

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # shortened for the tiny the-verdict.txt + VRAM
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

EPOCHS = 5
BATCH = 2
LR = 4e-4
EVAL_ITER = 5


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches):
    total = 0.0
    n = min(num_batches, len(data_loader))
    for i, (x, y) in enumerate(data_loader):
        if i >= n:
            break
        with torch.no_grad():
            total += calc_loss_batch(x, y, model, device).item()
    return total / n


def train(model, train_loader, val_loader, optimizer, device, num_epochs):
    step = 0
    for epoch in range(num_epochs):
        model.train()
        ep_t = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            step += 1
        model.eval()
        tr = calc_loss_loader(train_loader, model, device, EVAL_ITER)
        va = calc_loss_loader(val_loader, model, device, EVAL_ITER)
        print(
            f"  epoch {epoch + 1:>2}/{num_epochs} | "
            f"train_loss={tr:.4f} val_loss={va:.4f} | "
            f"{time.time() - ep_t:.1f}s/epoch (step {step})",
            flush=True,
        )


def main():
    dev, name = pick_device()
    print(f"\nTraining GPT-124M on: {name}  ({dev})\n", flush=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    split = int(0.9 * len(text))
    ctx = GPT_CONFIG_124M["context_length"]
    train_loader = create_dataloader_v1(
        text[:split], batch_size=BATCH, max_length=ctx,
        stride=ctx, drop_last=True, shuffle=True,
    )
    val_loader = create_dataloader_v1(
        text[split:], batch_size=BATCH, max_length=ctx,
        stride=ctx, drop_last=False, shuffle=False,
    )

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    print(
        f"params: {n_params / 1e6:.1f}M | "
        f"train batches: {len(train_loader)} | val batches: {len(val_loader)}",
        flush=True,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    t0 = time.time()
    train(model, train_loader, val_loader, optimizer, dev, EPOCHS)

    # Generate a sample to prove the model learned (works on DirectML too).
    model.eval()
    start = "Every effort moves you"
    ids = torch.tensor(tokenizer.encode(start), device=dev).unsqueeze(0)
    with torch.no_grad():
        out = generate_text_simple(
            model, ids, max_new_tokens=30, context_size=ctx,
        )
    sample = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\nTotal: {time.time() - t0:.1f}s on {name}")
    print(f'Sample from "{start}":\n{sample}\n')
    print("CH04 GPT TRAIN ON AMD GPU (DirectML): OK")


if __name__ == "__main__":
    main()
