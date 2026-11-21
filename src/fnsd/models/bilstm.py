"""Bidirectional LSTM with additive attention over time steps (PyTorch)."""
from __future__ import annotations

import torch
from torch import nn

from ..text import PAD_ID


class AttnBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, hidden: int = 128,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        mask = x != PAD_ID                                   # (B, T) True on real tokens
        emb = self.drop(self.emb(x))
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)               # (B, T), sums to 1 over real tokens
        context = (out * weights.unsqueeze(-1)).sum(1)       # (B, 2H)
        logits = self.fc(self.drop(context))
        return (logits, weights) if return_attention else logits
