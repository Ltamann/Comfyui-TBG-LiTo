# Apple LiTo-derived code

Parts of this custom node were adapted from Apple’s **ml-lito** reference
implementation. In this repository that includes the LiTo model and layer
implementation in `model.py` and `layers.py`, plus LiTo-derived sections of
`pipeline.py`.

Those portions remain governed by Apple’s software license, reproduced in
[`LICENSE`](LICENSE). They are not covered by the MIT license in
[`LICENSE_NODE.md`](LICENSE_NODE.md). Apple attribution and the original
license must remain with copies or substantial portions of that code.

The surrounding ComfyUI adapter, node registration, mask handling, coordinate
conversion, Gaussian-count control, and output wiring are project-specific
integration work. Where a file combines both kinds of work, the Apple terms
continue to apply to the Apple-derived portions.

No native ComfyUI files are modified or redistributed by this custom node.
