import logging

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.trellis2.vae import SparseStructureDecoder
from .model import ImageConditioning, LiToDiT, VoxelPredictor, GaussianDecoder


def _dino_weights(sd):
    top = {
        "cls_token": "embeddings.cls_token", "pos_embed": "embeddings.position_embeddings",
        "mask_token": "embeddings.mask_token", "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "norm.weight": "layernorm.weight", "norm.bias": "layernorm.bias",
    }
    result = {}
    for key, weight in sd.items():
        if key == "register_tokens":
            result["register_tokens"] = weight
        elif key in top:
            result["backbone." + top[key]] = weight
        elif key.startswith("blocks."):
            _, index, name = key.split(".", 2)
            prefix = f"backbone.encoder.layer.{index}."
            if name.startswith("attn.qkv."):
                suffix = name.rsplit(".", 1)[-1]
                for part, tensor in zip(("query", "key", "value"), weight.chunk(3, dim=0)):
                    result[prefix + f"attention.attention.{part}.{suffix}"] = tensor
            else:
                name = name.replace("attn.proj.", "attention.output.dense.").replace("ls1.gamma", "layer_scale1.lambda1").replace("ls2.gamma", "layer_scale2.lambda1")
                result[prefix + name] = weight
        else:
            raise ValueError(f"Comfyui-tbg-LiTo: unsupported DINO weight {key}")
    return result


def _weights(sd, prefix):
    return {key[len(prefix):].replace("rmsnorm_q.scale", "rmsnorm_q.weight").replace("rmsnorm_k.scale", "rmsnorm_k.weight"): value
            for key, value in sd.items() if key.startswith(prefix)}


def _load_stage(constructor, weights, dtype, device, offload):
    with torch.device("meta"):
        model = constructor()
    # Copy also detaches split Q/K/V weights from their checkpoint storage.
    weights = {key: value.to(device=offload, dtype=torch.float32 if key.endswith("freq_bands") else dtype, copy=True)
               for key, value in weights.items()}
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval()
    return comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=offload)


def preprocess(image, mask, crop=True):
    rgb = image[..., :3].movedim(-1, 0)[None]
    if mask is None:
        alpha = image[..., 3] if image.shape[-1] == 4 else torch.ones_like(image[..., 0])
    else:
        alpha = mask
    alpha = alpha[None, None].to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
    if alpha.shape[-2:] != rgb.shape[-2:]:
        alpha = F.interpolate(alpha, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    rgba = torch.cat((rgb, alpha), dim=1)
    if crop:
        foreground = alpha[0, 0] > 0.8
        if not foreground.any():
            foreground = alpha[0, 0] > 0
        ys, xs = foreground.nonzero(as_tuple=True)
        if xs.numel():
            h, w = image.shape[:2]
            # Match LiTo's determine_crop_and_pad(..., keep_optical_axis=True,
            # fill_ratio=0.8, pad_x_ratio=0.5, pad_y_ratio=0.5). The crop is
            # centered on the original optical axis rather than the object's
            # bounding-box center, so image-camera alignment is preserved.
            box_width = int(xs.max() - xs.min() + 1)
            box_height = int(ys.max() - ys.min() + 1)
            size = max(1, int(max(box_width, box_height) / 0.8 + 0.5))
            left = int((w - size) / 2)
            top = int((h - size) / 2)
            right, bottom = left + size, top + size
            crop_left, crop_top = max(left, 0), max(top, 0)
            crop_right, crop_bottom = min(right, w), min(bottom, h)
            rgba = rgba[:, :, crop_top:crop_bottom, crop_left:crop_right]
            rgba = F.pad(rgba, (crop_left - left, right - crop_right,
                                crop_top - top, bottom - crop_bottom))
    return F.interpolate(rgba, size=(518, 518), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)


class LiToPipeline:
    def __init__(self, checkpoint, occupancy_checkpoint, precision="auto"):
        self.device = mm.get_torch_device()
        offload = mm.unet_offload_device()
        if precision == "auto":
            self.dtype = mm.unet_dtype(device=self.device, supported_dtypes=[torch.bfloat16, torch.float32])
        else:
            self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        sd = comfy.utils.load_torch_file(checkpoint, safe_load=True)
        required = ("patch_encoder.learnable_model.weight", "velocity_estimator_ema.module.pos_mtx", "pretrained_tokenizer.gs_decoder.point_linear.weight")
        if any(key not in sd for key in required) or sd[required[0]].shape != (1024, 4, 14, 14):
            raise ValueError("Comfyui-tbg-LiTo requires Apple's recommended lito_dit_rgba.ckpt.")
        conditioning = _dino_weights(_weights(sd, "patch_encoder.dinov2_model.model."))
        conditioning.update({k: v for k, v in _weights(sd, "patch_encoder.").items() if not k.startswith("dinov2_model.")})
        self.conditioning = _load_stage(ImageConditioning, conditioning, self.dtype, self.device, offload)
        self.dit = _load_stage(LiToDiT, _weights(sd, "velocity_estimator_ema.module."), self.dtype, self.device, offload)
        self.voxel = _load_stage(VoxelPredictor, _weights(sd, "pretrained_tokenizer.voxel_decoder."), self.dtype, self.device, offload)
        self.gaussian = _load_stage(GaussianDecoder, _weights(sd, "pretrained_tokenizer.gs_decoder."), self.dtype, self.device, offload)
        del sd, conditioning
        occupancy = comfy.utils.load_torch_file(occupancy_checkpoint, safe_load=True)
        self.occupancy = _load_stage(lambda: SparseStructureDecoder(out_channels=1, latent_channels=8, num_res_blocks=2, channels=[512, 128, 32]), occupancy, self.dtype, self.device, offload)
        logging.info("Comfyui-tbg-LiTo loaded (%s, native attention and SPLAT).", self.dtype)

    def generate(self, rgba, seed, steps, cfg, method):
        progress = comfy.utils.ProgressBar(max(steps - 1, 0) + 4)
        mm.throw_exception_if_processing_interrupted()
        mm.load_models_gpu([self.conditioning], memory_required=1024 ** 3)
        cond = self.conditioning.model(rgba.to(device=self.device, dtype=self.dtype))
        progress.update(1)
        mm.load_models_gpu([self.dit], memory_required=2 * 1024 ** 3)
        cond, position = self.dit.model.preprocess(cond, cfg)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn((1, 8192, 32), generator=generator, device=self.device, dtype=torch.float32)
        # Match Apple's num_steps time points, including both t_eps and 1.
        times = [0.0001 + (1 - 0.0001) * i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0001]

        def velocity(x, t):
            mm.throw_exception_if_processing_interrupted()
            x = x.to(self.dtype)
            if cfg > 1:
                x = torch.cat((x, x), dim=0)
            timestep = torch.full((x.shape[0],), t, dtype=torch.float32, device=self.device)
            v = self.dit.model(x, timestep, cond, position).float()
            if cfg > 1:
                positive, negative = v.chunk(2)
                v = negative + cfg * (positive - negative)
            return v

        for t, next_t in zip(times, times[1:]):
            dt = next_t - t
            v = velocity(latent, t)
            estimate = latent + dt * v
            latent = latent + dt * 0.5 * (v + velocity(estimate, next_t)) if method == "heun" else estimate
            progress.update(1)
        del cond, position
        latent = (latent * 1.6464 + 0.0661).to(self.dtype)
        mm.throw_exception_if_processing_interrupted()
        mm.load_models_gpu([self.voxel], memory_required=1024 ** 3)
        voxels = self.voxel.model(latent)
        progress.update(1)
        mm.load_models_gpu([self.occupancy], memory_required=1024 ** 3)
        logits = self.occupancy.model(voxels)
        # Occupancy storage is ZYX; the Gaussian decoder consumes world XYZ.
        coords = (logits[0, 0].permute(2, 1, 0) >= 0).nonzero().to(torch.float32)
        coords = (coords + 0.5) / 32 - 1
        del voxels, logits
        progress.update(1)
        mm.throw_exception_if_processing_interrupted()
        mm.load_models_gpu([self.gaussian], memory_required=2 * 1024 ** 3 + coords.shape[0] * 64 * 59 * 4)
        splat = self.gaussian.model(latent, coords)
        progress.update(1)
        return tuple(t.to(device=mm.intermediate_device()) for t in splat)
