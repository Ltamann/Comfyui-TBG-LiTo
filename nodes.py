import os

import torch
import folder_paths
from comfy_api.latest import io, Types
from .pipeline import LiToPipeline, preprocess
from comfy.ldm.triposplat.gaussian import _matrix_to_quat, _quat_to_matrix


folder_paths.add_model_folder_path("lito", os.path.join(folder_paths.models_dir, "lito"))
LiToModel = io.Custom("TBG_LITO_MODEL")


def _to_comfy_frame(positions, rotations):
    # Save 3D Model's "-Z" up conversion: source -Z becomes viewer +Y.
    transform = positions.new_tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    positions = positions @ transform.T
    rotations = _matrix_to_quat(transform @ _quat_to_matrix(rotations))
    return positions, rotations


def _level_splat(positions, scales, rotations, opacities):
    matrices = _quat_to_matrix(rotations)
    normal = matrices[torch.arange(matrices.shape[0], device=matrices.device), :, scales.argmin(dim=-1)]
    normal = torch.where((normal[:, 1] < 0).unsqueeze(-1), -normal, normal)
    anisotropy = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-8)
    keep = (anisotropy > 2) & (normal[:, 1] > 0.7) & (opacities.reshape(-1) > 0.5)
    if not keep.any():
        return positions, rotations
    source_up = normal[keep].median(dim=0).values
    source_up = source_up / source_up.norm().clamp_min(1e-8)
    target_up = source_up.new_tensor((0, 1, 0))
    axis = torch.linalg.cross(source_up, target_up)
    sine = axis.norm()
    if sine < 1e-4:
        return positions, rotations
    axis = axis / sine
    cosine = source_up.dot(target_up).clamp(-1, 1)
    x, y, z = axis
    skew = torch.stack((
        torch.stack((x.new_zeros(()), -z, y)),
        torch.stack((z, x.new_zeros(()), -x)),
        torch.stack((-y, x, x.new_zeros(()))),
    ))
    level = torch.eye(3, dtype=positions.dtype, device=positions.device) + sine * skew + (1 - cosine) * (skew @ skew)
    return positions @ level.T, _matrix_to_quat(level @ matrices)


class LiToLoadModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.get_filename_list("lito")
        return io.Schema(
            node_id="TBGLiToLoadModel", display_name="Comfyui-tbg-LiTo Load Model", category="TBG/3D",
            description="Load local Apple LiTo and TRELLIS occupancy weights using ComfyUI model management.",
            inputs=[
                io.Combo.Input("checkpoint", options=files, default="lito_dit_rgba.ckpt"),
                io.Combo.Input("occupancy_checkpoint", options=files, default="ss_dec_conv3d_16l8_fp16.safetensors"),
                io.Combo.Input("precision", options=["auto", "bf16", "fp32"], default="auto"),
            ], outputs=[LiToModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint, occupancy_checkpoint, precision):
        model = LiToPipeline(folder_paths.get_full_path_or_raise("lito", checkpoint),
                             folder_paths.get_full_path_or_raise("lito", occupancy_checkpoint), precision)
        return io.NodeOutput(model)


class LiToImageToSplat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TBGLiToImageToSplat", display_name="Comfyui-tbg-LiTo", category="TBG/3D",
            description="Image to native Gaussian SPLAT. Mask is black background and white foreground.",
            search_aliases=["LiTo", "Apple", "image to gaussian splat"],
            inputs=[
                LiToModel.Input("model"), io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=100,
                             tooltip="Apple's time-point count: 20 means 19 integration intervals."),
                io.Int.Input("num_gaussians", default=0, min=0, max=1048576, step=32,
                             tooltip="Target splat count. 0 keeps LiTo's native density; lower values downsample deterministically."),
                io.Float.Input("cfg", default=3, min=0, max=20, step=0.1),
                io.Combo.Input("sampler", options=["heun", "euler"], default="heun"),
                io.Boolean.Input("crop", default=True, tooltip="Crop/pad around the original optical axis to 80% object fill, then resize to 518 square."),
                io.Boolean.Input("bake_preview_texture", default=True, advanced=True,
                                 tooltip="Export predicted base color only, matching native TripoSplat. Disable to retain all predicted SH bands."),
                io.Mask.Input("mask", optional=True, tooltip="Foreground mask: white = object, black = background. If absent, use image alpha, otherwise opaque."),
                io.Boolean.Input("auto_level", default=True, advanced=True,
                                 tooltip="Align the dominant decoded top surfaces to ComfyUI +Y after the fixed -Z-up conversion."),
            ], outputs=[io.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, model, image, seed, steps, num_gaussians, cfg, sampler, crop, bake_preview_texture=True, mask=None, auto_level=True):
        parts = []
        for i, img in enumerate(image):
            alpha = None if mask is None else mask[i % mask.shape[0]]
            rgba = preprocess(img, alpha, crop)
            part = model.generate(rgba, (seed + i) % (1 << 64), steps, cfg, sampler)
            # Exactly Save 3D Model's '-Z' up correction: +90 degrees about X.
            # Apply it to both centers and Gaussian frames, with no pose leveling.
            positions, scales, rotations, opacities, sh = part
            positions, rotations = _to_comfy_frame(positions, rotations)
            if auto_level:
                positions, rotations = _level_splat(positions, scales, rotations, opacities)
            if num_gaussians > 0 and positions.shape[0] > num_gaussians:
                keep = torch.linspace(0, positions.shape[0] - 1, num_gaussians,
                                      device=positions.device).round().long()
                positions, scales, rotations, opacities, sh = (tensor[keep] for tensor in (positions, scales, rotations, opacities, sh))
            if bake_preview_texture:
                sh = sh[:, :1].clone()
            parts.append((positions, scales, rotations, opacities, sh))
        counts = [part[0].shape[0] for part in parts]
        max_count = max(counts)
        tensors = []
        for index in range(5):
            first = parts[0][index]
            packed = first.new_zeros((len(parts), max_count, *first.shape[1:]))
            for i, part in enumerate(parts):
                packed[i, :counts[i]] = part[index]
            tensors.append(packed)
        lengths = torch.tensor(counts, dtype=torch.long, device=tensors[0].device) if len(set(counts)) > 1 else None
        return io.NodeOutput(Types.SPLAT(*tensors, counts=lengths))
