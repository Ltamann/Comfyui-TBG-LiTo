# Comfyui-tbg-LiTo Model Links

## Input Assets

- Input image: any RGB/RGBA image supported by ComfyUI.
- Optional mask: standard ComfyUI mask (`white = foreground`, `black = background`).
- Test image: `custom_nodes/Comfyui-tbg-LiTo/tests/assets/apple.png`

## Model Links

**lito**

- [lito_dit_rgba.ckpt](https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt) — recommended Apple LiTo image-to-3D checkpoint.
- [lito_dit.ckpt](https://ml-site.cdn-apple.com/models/lito/lito_dit.ckpt) — paper checkpoint; not recommended.

**lito occupancy decoder**

- [ss_dec_conv3d_16l8_fp16.safetensors](https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors) — official sparse-structure decoder used by LiTo.

## Model Storage Location

```text
ComfyUI/
└── models/
    └── lito/
        ├── lito_dit_rgba.ckpt
        └── ss_dec_conv3d_16l8_fp16.safetensors
```

The custom node loads these files through ComfyUI's `lito` model folder. They should not be stored inside the custom-node directory.

## Recommended Settings

- `precision`: `auto` or `bf16`
- `steps`: 40–60
- `sampler`: `heun`
- `cfg`: 3
- `num_gaussians`: `0` for LiTo's full native output, or `524288` for a lighter target
- `auto_level`: enabled when the generated object needs support-plane leveling

## Workflow Connections

```text
Load Image ───────────────► Comfyui-tbg-LiTo.image
Foreground Mask ──────────► Comfyui-tbg-LiTo.mask
LiTo Load Model ──────────► Comfyui-tbg-LiTo.model
Comfyui-tbg-LiTo.splat ───► Render Splat / Create 3D File (from Splat)
```

Use PLY when retaining LiTo's full spherical-harmonic appearance. SPZ stores base color only.

## Report Issue

- Custom-node runtime issues: report against the `Comfyui-tbg-LiTo` custom node.
- Native ComfyUI issues: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
- LiTo model issues: [Apple ml-lito](https://github.com/apple/ml-lito/issues)
