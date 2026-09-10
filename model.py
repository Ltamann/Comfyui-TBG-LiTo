# Adapted from Apple ml-lito. See LICENSE in this directory.
import torch
from torch import nn
import torch.nn.functional as F
import comfy.ops
import comfy_kitchen as ck
from comfy.image_encoders.dino2 import Dinov2Model
from .layers import FourierEmbed, MLP, SelfAttention, CrossAttention, Perceiver, PerceiverBlock, FinalLayer, output_mlp, voxel_groups


class ImageConditioning(nn.Module):
    def __init__(self, operations=comfy.ops.manual_cast):
        super().__init__()
        config = dict(num_hidden_layers=24, hidden_size=1024, num_attention_heads=16,
                      layer_norm_eps=1e-6, use_swiglu_ffn=False)
        self.backbone = Dinov2Model(config, None, None, operations)
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 1024))
        self.learnable_model = operations.Conv2d(4, 1024, 14, stride=14)
        self.learnable_paddings = nn.Parameter(torch.empty(5, 1024))

    def forward(self, rgba):
        premultiplied_rgb = rgba[:, :3] * rgba[:, 3:4]
        rgb = (premultiplied_rgb - rgba.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / rgba.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tokens = self.backbone.embeddings(rgb)
        registers = comfy.ops.cast_to_input(self.register_tokens, tokens).expand(rgba.shape[0], -1, -1)
        tokens = torch.cat((tokens[:, :1], registers, tokens[:, 1:]), dim=1)
        tokens, _ = self.backbone.encoder(tokens)
        tokens = F.layer_norm(tokens, (1024,))
        learned = self.learnable_model(torch.cat((rgb, rgba[:, 3:4]), dim=1)).flatten(2).transpose(1, 2)
        padding = comfy.ops.cast_to_input(self.learnable_paddings, learned)[None].expand(rgba.shape[0], -1, -1)
        return torch.cat((tokens, torch.cat((padding, learned), dim=1)), dim=-1)


class DiTMLP(nn.Module):
    def __init__(self, dim, operations):
        super().__init__()
        hidden = 256 * ((int(2 * dim * 4 / 3) + 255) // 256)
        self.w1 = operations.Linear(dim, hidden, bias=False)
        self.w2 = operations.Linear(hidden, dim)
        self.w3 = operations.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, operations):
        super().__init__()
        self.attn = SelfAttention(dim, dim, heads, operations)
        self.cross_attn = CrossAttention(dim, dim, dim, heads, operations)
        self.mlp = DiTMLP(dim, operations)
        self.scale_shift_table = nn.Parameter(torch.empty(6, dim))

    def forward(self, x, cond, t):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (comfy.ops.cast_to_input(self.scale_shift_table, x)[None] + t.reshape(t.shape[0], 6, -1)).chunk(6, dim=1)
        x = x + gate * self.attn(ck.adaln(x, scale, shift, eps=1e-6))
        x = F.layer_norm(x + self.cross_attn(x, cond), x.shape[-1:], eps=1e-6)
        # LiTo deliberately has no normalization before the MLP modulation.
        return x + gate_mlp * self.mlp(x * (1 + scale_mlp) + shift_mlp)


class ConditionEmbedder(nn.Module):
    def __init__(self, dim, hidden, operations):
        super().__init__()
        self.register_buffer("y_embedding", torch.empty(dim))
        self.y_proj = MLP(dim, hidden, hidden, operations=operations)

    def forward(self, cond, cfg):
        positive = self.y_proj(cond)
        if cfg > 1:
            negative = self.y_proj(comfy.ops.cast_to_input(self.y_embedding, cond)[None, None].expand_as(cond))
            return torch.cat((positive, negative), dim=0)
        return positive


class DiTFinal(nn.Module):
    def __init__(self, dim, operations):
        super().__init__()
        self.linear = operations.Linear(dim, 32)
        self.adaLN_modulation = nn.Sequential(operations.Linear(dim, dim), nn.SiLU(), operations.Linear(dim, 2 * dim))

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        return self.linear(ck.adaln(x, scale[:, None], shift[:, None], eps=1e-6))


class LiToDiT(nn.Module):
    def __init__(self, operations=comfy.ops.manual_cast):
        super().__init__()
        dim = 1152
        self.pos_mtx = nn.Parameter(torch.empty(8192, dim))
        self.z_proj = operations.Linear(32, dim)
        self.z_proj_ln = operations.LayerNorm(dim, eps=1e-6)
        self.pos_proj = operations.Linear(dim, dim)
        self.t_embedder = FourierEmbed(1)
        self.t_proj = nn.Sequential(operations.Linear(64, dim), nn.SiLU(), operations.Linear(dim, dim))
        self.t0_proj = nn.Sequential(nn.SiLU(), operations.Linear(dim, 6 * dim))
        self.cond_embedder = ConditionEmbedder(2048, dim, operations)
        self.blocks = nn.ModuleList([DiTBlock(dim, 16, operations) for _ in range(28)])
        self.final_layer = DiTFinal(dim, operations)

    def forward(self, tokens, timestep, conditioning, position):
        t = self.t_proj(self.t_embedder(timestep[:, None]).to(tokens.dtype))
        modulation = self.t0_proj(t)
        x = self.z_proj_ln(self.z_proj(tokens)) + position
        for block in self.blocks:
            x = block(x, conditioning, modulation)
        return self.final_layer(x, t)

    def preprocess(self, conditioning, cfg):
        position = self.pos_proj(comfy.ops.cast_to_input(self.pos_mtx, conditioning))[None]
        return self.cond_embedder(conditioning, cfg), position


class VoxelQueries(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.init_query = nn.Parameter(torch.empty(16, 16, 16, 512))
        self.zyx_pos_encoder = FourierEmbed(3, 128, include_input=True)
        self.init_query_linear = operations.Linear(771, 512)
        self.encoder = Perceiver([PerceiverBlock(512, 512, 1024, 8, operations=operations) for _ in range(4)])

    def forward(self, latent):
        axis = (torch.arange(16, device=latent.device, dtype=torch.float32) + 0.5) / 8 - 1
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
        pos = self.init_query_linear(self.zyx_pos_encoder(coords).to(latent.dtype))
        query = (comfy.ops.cast_to_input(self.init_query, latent) + pos).reshape(1, 4096, 512)
        return self.encoder(query.expand(latent.shape[0], -1, -1), latent)


class VoxelPredictor(nn.Module):
    def __init__(self, operations=comfy.ops.manual_cast):
        super().__init__()
        self.input_linear = operations.Linear(32, 512)
        self.net = VoxelQueries(operations)
        self.final_layer = FinalLayer(512, 8, operations)

    def forward(self, latent):
        x = self.final_layer(self.net(self.input_linear(latent)))
        return x.reshape(latent.shape[0], 16, 16, 16, 8).movedim(-1, 1)


class GaussianDecoder(nn.Module):
    def __init__(self, operations=comfy.ops.manual_cast):
        super().__init__()
        self.xyz_encoding = FourierEmbed(3)
        self.point_linear = operations.Linear(195, 512)
        self.point_mlp = output_mlp(512, 512, operations)
        self.perceiver = Perceiver([PerceiverBlock(512, 32, 512, 8, swiglu=True, bias=False, kv_linear=True, operations=operations) for _ in range(6)])
        self.gs_output_shape_mlp = output_mlp(512, 10 * 64, operations)
        self.gs_output_color_mlp = output_mlp(512, 49 * 64, operations)

    def forward(self, latent, coords):
        groups = [voxel_groups(coords, 0.25, shift) for shift in (0, 0.125)]
        query = torch.cat((coords, self.xyz_encoding(coords)), dim=-1).to(latent.dtype)
        query = self.point_mlp(self.point_linear(query))[None]
        query = self.perceiver(query, latent, groups)[0]
        # Chunk only pointwise output heads; attention always sees complete voxel groups.
        parts = []
        for start in range(0, query.shape[0], 4096):
            chunk = query[start:start + 4096]
            shape = self.gs_output_shape_mlp(chunk).float().reshape(-1, 64, 10)
            color = self.gs_output_color_mlp(chunk).float().reshape(-1, 64, 49)
            xyz = (shape[..., :3].sigmoid() * 2 - 1) * 0.05 + coords[start:start + 4096, None]
            # LiTo's build_rotation uses wxyz, despite its xyzw docstring.
            rotation = F.normalize(shape[..., 3:7], dim=-1)
            scale = ((shape[..., 7:10].sigmoid() * 0.01).square() + 0.001 ** 2).sqrt()
            opacity = (color[..., :1] + 0.1).sigmoid()
            sh = color[..., 1:].reshape(-1, 16, 3)
            parts.append((xyz.flatten(0, 1), scale.flatten(0, 1), rotation.flatten(0, 1), opacity.flatten(0, 1), sh))
        if not parts:
            return (coords.new_empty(0, 3), coords.new_empty(0, 3), coords.new_empty(0, 4), coords.new_empty(0, 1), coords.new_empty(0, 16, 3))
        return tuple(torch.cat(items, dim=0) for items in zip(*parts))
