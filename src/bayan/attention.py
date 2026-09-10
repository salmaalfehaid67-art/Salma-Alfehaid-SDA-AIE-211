"""Lab 2: scaled dot-product attention and multi-head attention."""

import math
import torch
from torch import nn


def attention(q, k, v, mask=None):
    """
    Scaled dot-product attention.

    Returns only the output tensor to match the course tests.
    """
    d_k = q.size(-1)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(
        weights,
        v
    )

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_proj(q)
        k = self
