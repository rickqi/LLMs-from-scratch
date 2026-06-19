"""Pretrain the ch04 GPT model on an AMD GPU via DirectML (ch05 style).

The "full pretraining" companion to `train_gpt_ch04.py`. Mirrors LLMs-from-scratch
Chapter 5: trains more epochs and periodically generates sample text so you can
watch the output go from gibberish to coherent prose as training progresses.

Reuses this repo's own model + data:
  - model:  ch04/01_main-chapter-code/gpt.py
  - data:   ch02/01_main-chapter-code/the-verdict.txt

Device: AMD Radeon GPU via torch-directml (`dml_device.pick_device()`), for GPUs
not on the ROCm WSL support matrix.

NOTE: the-verdict.txt (~20 KB) is a teaching-grade dataset; generated text
quality is limited. The goal is to verify the pretraining loop runs end-to-end
on an AMD GPU and loss keeps dropping.

Run (native Windows python with torch + torch-directml + tiktoken):
    python train_pretrain_ch05.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

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
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

EPOCHS = 10
BATCH = 2
LR = 4e-4
EVAL_FREQ = 5       # evaluate every N steps mid-epoch
EVAL_ITER = 5
GEN_EVERY_EPOCH = 2  # print a generated sample every N epochs
START_CONTEXT = "Every effort moves you"


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


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, device=device).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)


def generate_and_print_sample(model, device, tokenizer, ctx):
    """Generate a short sample at each eval point to visualize progress."""
    model.eval()
    with torch.no_grad():
        out = generate_text_simple(
            model=model,
            idx=text_to_token_ids(START_CONTEXT, tokenizer, device),
            max_new_tokens=30,
            context_size=ctx,
        )
    text = token_ids_to_text(out, tokenizer)
    print(f"    sample: {text.replace(chr(10), ' ')}", flush=True)


def train(model, train_loader, val_loader, optimizer, device, tokenizer, num_epochs):
    step = 0
    global_t = time.time()
    for epoch in range(num_epochs):
        model.train()
        ep_t = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()

            if step % EVAL_FREQ == 0:
                model.eval()
                tr = calc_loss_loader(train_loader, model, device, EVAL_ITER)
                va = calc_loss_loader(val_loader, model, device, EVAL_ITER)
                model.train()
                if step == 0:
                    print(f"    [init]      train_loss={tr:.4f} val_loss={va:.4f}",
                          flush=True)

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
        if (epoch + 1) % GEN_EVERY_EPOCH == 0 or epoch == num_epochs - 1:
            generate_and_print_sample(model, device, tokenizer,
                                      GPT_CONFIG_124M["context_length"])

    print(f"\nTotal pretrain time: {time.time() - global_t:.1f}s", flush=True)


def main():
    dev, name = pick_device()
    print(f"\nPretraining GPT-124M on: {name}  ({dev})\n", flush=True)

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
    tokenizer = tiktoken.get_encoding("gpt2")

    print(
        f"params: {n_params / 1e6:.1f}M | epochs: {EPOCHS} | "
        f"train batches/epoch: {len(train_loader)} | val batches: {len(val_loader)}",
        flush=True,
    )

    train(model, train_loader, val_loader, optimizer, dev, tokenizer, EPOCHS)
    print("\nCH05 PRETRAIN ON AMD GPU (DirectML): OK")


if __name__ == "__main__":
    main()
