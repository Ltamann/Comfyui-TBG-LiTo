# Comfyui-tbg-LiTo

Apple LiTo image-to-3D Gaussian Splat generation as a native ComfyUI custom node.

The node runs LiTo with ComfyUI's existing model management, attention, image, mask, SPLAT, rendering, and 3D file tools. It does not modify native ComfyUI files and does not require installing the original ComfyUI-LiTo dependency stack (`nvdiffrast`, `pytorch3d`, `spconv`, `flash-attn`, or `xformers`).

## Features

- RGB/RGBA image to Gaussian SPLAT.
- Standard ComfyUI white-foreground/black-background mask input.
- Native `Render Splat` and `Create 3D File (from Splat)` compatibility.
- Full LiTo spherical-harmonic appearance in PLY export.
- Optional base-color baking for native previews and SPZ export.
- Fixed LiTo `−Z` up conversion to ComfyUI `+Y`.
- Optional automatic support-plane leveling.
- Configurable target Gaussian count.

## Nodes

### Comfyui-tbg-LiTo Load Model

Loads the LiTo image-to-3D checkpoint and sparse occupancy decoder from `models/lito`.

### Comfyui-tbg-LiTo

Generates a native ComfyUI `SPLAT` object from an image and optional foreground mask.

## Installation

Copy this directory into:

```text
ComfyUI/custom_nodes/Comfyui-tbg-LiTo/
```

Place the model files in:

```text
ComfyUI/models/lito/
```

Restart ComfyUI and add `Comfyui-tbg-LiTo Load Model` followed by `Comfyui-tbg-LiTo`.

## Models

Download the recommended files listed in [MODEL_LINKS.md](MODEL_LINKS.md). The recommended image-to-3D checkpoint is `lito_dit_rgba.ckpt`.

## Basic Workflow

```text
Load Image ───────────────► Comfyui-tbg-LiTo.image
Foreground Mask ──────────► Comfyui-tbg-LiTo.mask
LiTo Load Model ──────────► Comfyui-tbg-LiTo.model
Comfyui-tbg-LiTo.splat ───► Render Splat
                         └► Create 3D File (from Splat)
```

The mask convention is fixed: white is the object and black is the background. If no mask is supplied, the node uses the image alpha channel or treats the image as opaque.

## Recommended Settings

- `steps`: 40–60
- `sampler`: `heun`
- `cfg`: 3
- `precision`: `auto` or `bf16`
- `num_gaussians`: `0` for native LiTo density, `524288` for a lighter output
- `auto_level`: enabled when support-plane leveling is needed

Use PLY to preserve LiTo's full view-dependent spherical-harmonic appearance. SPZ and other base-color formats do not preserve the higher SH bands.

## Credits and License

LiTo was created by Apple Research. See [LICENSE](LICENSE) and [LICENSE_MODEL](LICENSE_MODEL). The model license controls use of the pretrained weights.
