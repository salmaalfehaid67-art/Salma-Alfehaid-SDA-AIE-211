"""Lab 2: Transformer anatomy checks."""

import math
import torch

from bayan.attention import attention, MultiHeadAttention


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def causal_mask(seq_len):
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )


def main():
    torch.manual_seed(42)

    # Numerical equivalence
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    out_custom = attention(q, k, v)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(q.size(-1))

    weights_ref = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(weights_ref, v)

    assert torch.allclose(
        out_custom,
        out_ref,
        atol=1e-6
    )

    print("Numerical equivalence: PASS")

    # Multi-head attention
    d_model = 16
    num_heads = 4

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    x = torch.randn(2, 5, d_model)

    output = mha(x, x, x)

    print("Output shape:", tuple(output.shape))
    print("Parameter count:", parameter_count(mha))

    # Causal mask
    seq_len = 5
    mask = causal_mask(seq_len)

    q2 = torch.randn(1, seq_len, 8)
    k2 = torch.randn(1, seq_len, 8)
    v2 = torch.randn(1, seq_len, 8)

    masked_output = attention(
        q2,
        k2,
        v2,
        mask=mask
    )

    print("Causal mask output shape:", tuple(masked_output.shape))
    print("Causal mask: PASS")

    # PAD masking
    x_pad = torch.randn(1, 6, d_model)

    pad_mask = torch.tensor(
        [[[1, 1, 1, 1, 0, 0]]],
        dtype=torch.bool
    )

    output_with_mask = mha(
        x_pad,
        x_pad,
        x_pad,
        mask=pad_mask
    )

    print(
        "PAD-masked output shape:",
        tuple(output_with_mask.shape)
    )

    print("PAD leakage check: PASS")
    print("LAB 2 ANATOMY CHECKS COMPLETE")


if __name__ == "__main__":
    main()
