# Comfyui-tbg-LiTo licensing

This repository combines original ComfyUI node integration with code and model
artifacts from other projects. The files below do not all have the same license.

| Component | Files/artifacts | Governing terms |
| --- | --- | --- |
| Original ComfyUI node and integration work | `nodes.py` (integration portions), workflow helpers, tests, and documentation written for this node | [`LICENSE_NODE.md`](LICENSE_NODE.md) |
| Apple LiTo-derived implementation | `model.py`, `layers.py`, and LiTo-derived portions of `pipeline.py` | Apple software license in [`LICENSE`](LICENSE); details in [`LICENSE_DERIVATIVE_CODE.md`](LICENSE_DERIVATIVE_CODE.md) |
| Apple LiTo checkpoint | `lito_dit_rgba.ckpt` and any Apple LiTo model derivative | Apple Machine Learning Research Model License in [`LICENSE_MODEL`](LICENSE_MODEL) |
| TRELLIS occupancy decoder checkpoint | `ss_dec_conv3d_16l8_fp16.safetensors` | The checkpoint's upstream Microsoft TRELLIS terms; see [`MODEL_LINKS.md`](MODEL_LINKS.md). This project does not relicense it. |
| ComfyUI and runtime dependencies | Imported native ComfyUI modules and separately installed packages | Their respective upstream licenses. They are not relicensed by this project. |

The model files are ignored by Git and are expected to be downloaded by the
user. Do not redistribute a checkpoint unless its own license permits it.

The custom node makes no changes to native ComfyUI source files.
