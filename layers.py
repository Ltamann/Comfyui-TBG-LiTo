# Adapted from Apple ml-lito. See LICENSE in this directory.
import torch
from torch import nn
import torch.nn.functional as F
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention


class FourierEmbed(nn.Module):
    def __init__(self, dim, num_freqs=32, include_input=False):
        super().__init__()
        self.include_input = include_input
        self.dim_out = dim * (2 * num_freqs + int(include_input))
        self.register_buffer("freq_bands", torch.empty(num_freqs))

    def forward(self, x):
        phase = x.unsqueeze(-1) * comfy.ops.cast_to_input(self.freq_bands, x)
        features = [phase.sin().flatten(-2), phase.cos().flatten(-2)]
        return torch.cat(([x] if self.include_input else []) + features, dim=-1)


class MLP(nn.Module):
    def __init__(self, dim, hidden, out=None, bias=True, swiglu=False, operations=comfy.ops.manual_cast):
        super().__init__()
        self.swiglu = swiglu
        if swiglu:
            self.w12 = operations.Linear(dim, hidden * 2, bias=bias)
            self.w3 = operations.Linear(hidden, out or dim, bias=bias)
        else:
            self.fc1 = operations.Linear(dim, hidden, bias=bias)
            self.fc2 = operations.Linear(hidden, out or dim, bias=bias)

    def forward(self, x):
        if self.swiglu:
            a, b = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(a) * b)
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def voxel_groups(coords, cell_width, shift):
    # Group planning belongs to this decode only. Coordinates are data; lengths are Python values.
    cells = torch.floor((coords + shift) / cell_width).to(torch.int64)
    _, inverse, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    groups = order.split(counts.tolist())
    buckets = {}
    for group in groups:
        size = group.numel()
        bucket = 1 << (size - 1).bit_length()
        buckets.setdefault(bucket, []).append(group)
    batches = []
    for width, groups in buckets.items():
        for start in range(0, len(groups), max(1, 8192 // width)):
            chunk = groups[start:start + max(1, 8192 // width)]
            indices = torch.zeros((len(chunk), width), dtype=torch.long, device=coords.device)
            valid = torch.zeros_like(indices, dtype=torch.bool)
            for i, group in enumerate(chunk):
                indices[i, :group.numel()] = group
                valid[i, :group.numel()] = True
            batches.append((indices, valid))
    return batches


def attention(q, k, v, heads, groups=None):
    if groups is None:
        return optimized_attention(q, k, v, heads)
    # Gaussian decoding processes one shape at a time.
    out = torch.empty_like(q)
    for indices, valid in groups:
        mask = torch.zeros((*valid.shape[:1], 1, 1, valid.shape[1]), dtype=q.dtype, device=q.device)
        mask.masked_fill_(~valid[:, None, None, :], -torch.finfo(q.dtype).max)
        result = optimized_attention(q[0, indices], k[0, indices], v[0, indices], heads, mask=mask)
        out[0, indices[valid]] = result[valid]
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim, qkv_dim, heads, operations=comfy.ops.manual_cast):
        super().__init__()
        self.heads = heads
        self.linear_qkv = operations.Linear(dim, 3 * qkv_dim)
        self.linear_out = operations.Linear(qkv_dim, dim)
        self.rmsnorm_q = operations.RMSNorm(qkv_dim, eps=1e-8)
        self.rmsnorm_k = operations.RMSNorm(qkv_dim, eps=1e-8)

    def forward(self, x, groups=None):
        q, k, v = self.linear_qkv(x).chunk(3, dim=-1)
        return self.linear_out(attention(self.rmsnorm_q(q), self.rmsnorm_k(k), v, self.heads, groups))


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, qkv_dim, heads, operations=comfy.ops.manual_cast):
        super().__init__()
        self.heads = heads
        self.layernorm_q = operations.LayerNorm(dim)
        self.layernorm_kv = operations.LayerNorm(context_dim)
        self.linear_q = operations.Linear(dim, qkv_dim)
        self.linear_kv = operations.Linear(context_dim, 2 * qkv_dim)
        self.linear_out = operations.Linear(qkv_dim, dim)
        self.rmsnorm_q = operations.RMSNorm(qkv_dim, eps=1e-8)
        self.rmsnorm_k = operations.RMSNorm(qkv_dim, eps=1e-8)

    def forward(self, x, context):
        q = self.rmsnorm_q(self.linear_q(self.layernorm_q(x)))
        k, v = self.linear_kv(self.layernorm_kv(context)).chunk(2, dim=-1)
        return self.linear_out(attention(q, self.rmsnorm_k(k), v, self.heads))


class PerceiverBlock(nn.Module):
    def __init__(self, dim, context_dim, qkv_dim, heads, num_self=2, swiglu=False, bias=True, kv_linear=False, operations=comfy.ops.manual_cast):
        super().__init__()
        self.kv_linear = operations.Linear(context_dim, context_dim, bias=False) if kv_linear else None
        self.ca_layer = CrossAttention(dim, context_dim, qkv_dim, heads, operations)
        self.ca_ln = operations.LayerNorm(dim, eps=1e-6)
        self.ca_mlp = MLP(dim, dim * 4, bias=bias, swiglu=swiglu, operations=operations)
        self.ln1_layers = nn.ModuleList([operations.LayerNorm(dim, eps=1e-6) for _ in range(num_self)])
        self.ln2_layers = nn.ModuleList([operations.LayerNorm(dim, eps=1e-6) for _ in range(num_self)])
        self.sa_layers = nn.ModuleList([SelfAttention(dim, qkv_dim, heads, operations) for _ in range(num_self)])
        self.mlp_layers = nn.ModuleList([MLP(dim, dim * 4, bias=bias, swiglu=swiglu, operations=operations) for _ in range(num_self)])

    def forward(self, x, context, groups=None):
        if self.kv_linear is not None:
            context = self.kv_linear(context)
        x = x + self.ca_layer(x, context)
        x = x + self.ca_mlp(self.ca_ln(x))
        for i, (ln1, attn, ln2, mlp) in enumerate(zip(self.ln1_layers, self.sa_layers, self.ln2_layers, self.mlp_layers)):
            x = x + attn(ln1(x), groups[i % 2] if groups is not None else None)
            x = x + mlp(ln2(x))
        return x


class Perceiver(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, context, groups=None):
        for block in self.blocks:
            x = block(x, context, groups)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out, operations=comfy.ops.manual_cast):
        super().__init__()
        self.norm_final = operations.LayerNorm(dim, eps=1e-6)
        self.linear = operations.Linear(dim, out)

    def forward(self, x):
        return self.linear(self.norm_final(x))


def output_mlp(dim, out, operations=comfy.ops.manual_cast):
    return nn.Sequential(operations.LayerNorm(dim, eps=1e-6), MLP(dim, dim, bias=False, swiglu=True, operations=operations), FinalLayer(dim, out, operations))
