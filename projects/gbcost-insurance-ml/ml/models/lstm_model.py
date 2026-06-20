"""LSTM time-series model for policy-group claim prediction.

Complements LightGBM with sequential pattern learning. Trained on monthly
policy-group aggregations (from global_stats), predicts next-month totals.

Integration: LSTM predictions are added as features to LightGBM training
(see feature_store.py §9).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class PolicyLSTM:
    """2-layer LSTM for policy-group monthly claim prediction.

    Input:  12-month window of (total, count, mean, members, claims_per_member)
    Output: next-month total claim amount
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 12,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len

        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn

        class LSTMPredictor(nn.Module):
            def __init__(self, inp, hid, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
                self.head = nn.Sequential(
                    nn.Linear(hid, hid // 2),
                    nn.ReLU(),
                    nn.Linear(hid // 2, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.model = LSTMPredictor(input_dim, hidden_dim, num_layers, dropout)
        self.optimizer = None
        self._device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        self.train_losses: List[float] = []
        self.feature_names: List[str] = []

    @property
    def device(self) -> str:
        return self._device

    def prepare_data(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.2,
    ) -> Tuple:
        """Prepare PyTorch tensors from numpy arrays.

        Args:
            sequences: (n_samples, seq_len, input_dim)
            targets: (n_samples,)
            val_split: validation fraction

        Returns:
            (X_train, y_train, X_val, y_val) as torch tensors
        """
        n = len(sequences)
        split_n = int(n * (1 - val_split))

        X_train = self.torch.tensor(sequences[:split_n], dtype=self.torch.float32).to(self._device)
        y_train = self.torch.tensor(targets[:split_n], dtype=self.torch.float32).to(self._device)
        X_val = self.torch.tensor(sequences[split_n:], dtype=self.torch.float32).to(self._device)
        y_val = self.torch.tensor(targets[split_n:], dtype=self.torch.float32).to(self._device)

        logger.info("LSTM data: train=%d, val=%d, shape=%s", split_n, n - split_n, sequences.shape)
        return X_train, y_train, X_val, y_val

    def train(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        patience: int = 10,
        val_split: float = 0.2,
    ) -> Dict[str, float]:
        X_train, y_train, X_val, y_val = self.prepare_data(sequences, targets, val_split)

        self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = self.nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Training LSTM: %d params, epochs=%d, device=%s",
                     sum(p.numel() for p in self.model.parameters()), epochs, self._device)

        for epoch in range(epochs):
            self.model.train()
            perm = self.torch.randperm(len(X_train))
            epoch_loss = 0.0

            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                self.optimizer.zero_grad()
                pred = self.model(X_train[idx])
                loss = loss_fn(pred, y_train[idx])
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(idx)

            epoch_loss /= len(X_train)
            self.train_losses.append(epoch_loss)

            self.model.eval()
            with self.torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = loss_fn(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info("  Epoch %3d: train_loss=%.4f, val_loss=%.4f", epoch, epoch_loss, val_loss)

            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d (val_loss=%.4f)", epoch, val_loss)
                break

        logger.info("LSTM training complete: best_val_loss=%.4f", best_val_loss)
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self.torch.tensor(sequences, dtype=self.torch.float32).to(self._device)
        with self.torch.no_grad():
            return self.model(X).cpu().numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "train_losses": self.train_losses,
        }, str(path))
        logger.info("LSTM saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyLSTM":
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        lstm = cls(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            seq_len=ckpt["seq_len"],
        )
        lstm.model.load_state_dict(ckpt["model_state"])
        lstm.model.to(lstm._device)
        lstm.train_losses = ckpt.get("train_losses", [])
        logger.info("LSTM loaded: %s (%d params)", path,
                     sum(p.numel() for p in lstm.model.parameters()))
        return lstm
