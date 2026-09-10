import json
import hashlib
from io import BytesIO
from pathlib import Path

from aiohttp import web
import folder_paths
import nodes
import torch
from comfy_api.latest import ComfyExtension, IO, InputImpl, Types, UI
from comfy_extras.nodes_gaussian_splat import File3DToSplat, RenderSplat, _mat_to_quat, _quat_to_mat
from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
from server import PromptServer


@PromptServer.instance.routes.get("/tbg/load3d-modules")
async def load3d_modules(request):
    assets = Path(PromptServer.instance.web_root) / "assets"
    required_exports = {"settingStore": "uploadTempImage"}
    modules = {}
    for name in ("useLoad3d", "Load3DConfiguration", "settingStore", "load3dSerialize"):
        paths = sorted(assets.glob(f"{name}-*.js"))
        export = required_exports.get(name)
        if export:
            paths = [path for path in paths if export in path.read_text(encoding="utf-8")]
        if not paths:
            raise web.HTTPNotFound(text=f"Native 3D frontend module not found: {name}")
        modules[name] = f"/assets/{paths[0].name}"
    return web.json_response(modules)


def _normalize_path(path):
    return path.replace("\\", "/")


def _connected_input(input_types, name):
    if not input_types:
        return False
    if isinstance(input_types, dict):
        return input_types.get(name) is not None
    return name in input_types


def _valid_preview_file(model_file):
    if not model_file:
        return "none"
    value = model_file.replace("\\", "/")
    if value.endswith(" [temp]"):
        filename = value[:-7]
        if not (Path(folder_paths.get_temp_directory()) / filename).is_file():
            return "none"
    return model_file


def _file_for_preview(model_3d):
    if model_3d.format not in {"gltf", "glb", "obj", "fbx", "stl", "ply", "spz", "splat", "ksplat"}:
        raise ValueError(f"Unsupported 3D viewer format: {model_3d.format!r}")
    digest = hashlib.sha256(model_3d.get_bytes()).hexdigest()
    filename = f"trellis2_model_{digest}.{model_3d.format}"
    path = Path(folder_paths.get_temp_directory()) / filename
    if not path.is_file():
        model_3d.save_to(str(path))
    return f"{filename} [temp]"


_SPLAT_FORMATS = {"ply", "splat", "spz", "ksplat"}


def _apply_model_info_to_splat(splat, model_3d_info):
    if not model_3d_info:
        return splat
    transform = model_3d_info[0] if isinstance(model_3d_info, list) else model_3d_info
    if not isinstance(transform, dict) or not all(key in transform for key in ("position", "quaternion", "scale")):
        return splat
    dev, dtype = splat.positions.device, splat.positions.dtype
    vec = lambda value: torch.tensor([float(value["x"]), float(value["y"]), float(value["z"])], device=dev, dtype=dtype)
    position, scale = vec(transform["position"]), vec(transform["scale"])
    quaternion = transform["quaternion"]
    q = torch.tensor([float(quaternion.get(key, 0.0)) for key in ("w", "x", "y", "z")], device=dev, dtype=dtype)
    q = q / q.norm().clamp_min(1e-8)
    rotation = _quat_to_mat(q[None])[0]
    # Viewer transforms are Y-up; RenderSplat consumes the (x, -y, -z) frame.
    axes = position.new_tensor([1.0, -1.0, -1.0])
    linear = axes[:, None] * (rotation @ torch.diag(scale)) * axes[None, :]
    position = position * axes
    positions = splat.positions @ linear.T + position
    cov_rotation = _quat_to_mat(splat.rotations.reshape(-1, 4))
    covariance = (cov_rotation * splat.scales.reshape(-1, 3)[:, None, :].square()) @ cov_rotation.transpose(-1, -2)
    covariance = linear @ covariance @ linear.T
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvectors = eigenvectors * torch.where(torch.linalg.det(eigenvectors) < 0, -1.0, 1.0)[..., None, None]
    scales = eigenvalues.clamp_min(0).sqrt().reshape(splat.scales.shape)
    rotations = _mat_to_quat(eigenvectors).reshape(splat.rotations.shape)
    return type(splat)(positions, scales, rotations, splat.opacities, splat.sh, counts=splat.counts)


def _render_splat_outputs(model_3d, camera_info, width, height, background_image=None, model_3d_info=None):
    if model_3d is None or (model_3d.format or "").lower() not in _SPLAT_FORMATS:
        return None

    splat = _apply_model_info_to_splat(File3DToSplat.execute(model_3d).result[0], model_3d_info)
    options = dict(
        splat=splat,
        width=width,
        height=height,
        frames=1,
        splat_scale=1.0,
        sharpen=1.0,
        headlight_shading=0.0,
        opacity_threshold=0.0,
        background="#000000",
        camera_info=camera_info or None,
    )
    color = RenderSplat.execute(**options, render_style="color", bg_image=background_image)
    normal = RenderSplat.execute(**options, render_style="normal")
    return color.result[0], 1.0 - color.result[1], normal.result[0]


class Trellis2PreviewUI(UI.PreviewUI3D):
    def __init__(self, model_file, camera_info, background_image, width, height):
        super().__init__(model_file, camera_info, bg_image=background_image)
        self.width = width
        self.height = height
        if background_image is not None and self.bg_image_path:
            digest = hashlib.sha256(background_image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            source = Path(folder_paths.get_temp_directory()) / Path(self.bg_image_path).name
            target = source.with_name(f"tbg_background_{digest}.png")
            if source != target:
                if target.exists():
                    source.unlink(missing_ok=True)
                else:
                    source.replace(target)
                self.bg_image_path = f"temp/{target.name}"

    def as_dict(self):
        result = super().as_dict()
        result["result"].append({"width": self.width, "height": self.height})
        return result


def _output_model_files():
    output_dir = Path(folder_paths.get_output_directory())
    return [
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".gltf", ".glb", ".obj", ".fbx", ".stl", ".ply"}
    ]


class Trellis2Load3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = Path(folder_paths.get_input_directory()) / "3d"
        input_dir.mkdir(parents=True, exist_ok=True)
        base_path = Path(folder_paths.get_input_directory())
        files = [
            _normalize_path(str(path.relative_to(base_path)))
            for path in input_dir.rglob("*")
            if path.suffix.lower() in {".gltf", ".glb", ".obj", ".fbx", ".stl", ".spz", ".splat", ".ply", ".ksplat"}
        ]

        return IO.Schema(
            node_id="Trellis2Load3D",
            display_name="TBG Preview Splat or 3D & Animation",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.Combo.Input("model_file", options=["none"] + sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("image"),
                IO.Mesh.Input("mesh", optional=True,
                              tooltip="Optional mesh input. When connected, it is used instead of model_file."),
                IO.Image.Input("background_image", optional=True),
                IO.MultiType.Input("model_3d", optional=True, types=[
                    IO.File3DAny, IO.File3DGLB, IO.File3DGLTF, IO.File3DFBX, IO.File3DOBJ,
                    IO.File3DSTL, IO.File3DPLY, IO.File3DSplatAny, IO.File3DPointCloudAny,
                    IO.File3DSPLAT, IO.File3DSPZ, IO.File3DKSPLAT,
                ], tooltip="Optional 3D file, including gaussian splats. Takes priority over mesh and model_file."),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.Image.Output(display_name="normal"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Video.Output(display_name="recording_video"),
                IO.File3DAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def validate_inputs(cls, model_file, input_types=None, **kwargs):
        if (_connected_input(input_types, "model_3d") or _connected_input(input_types, "mesh")
                or kwargs.get("model_3d") is not None or kwargs.get("mesh") is not None):
            return True
        if not model_file or model_file == "none":
            return True
        if (Path(folder_paths.get_output_directory()) / model_file).is_file():
            return True
        if not folder_paths.exists_annotated_filepath(model_file):
            return f"Invalid 3D model file: {model_file}"
        return True

    @classmethod
    def execute(cls, model_file, image, width, height, background_image=None, mesh=None, model_3d=None, **kwargs):
        if background_image is not None:
            height, width = background_image.shape[1:3]
        splat_outputs = _render_splat_outputs(model_3d, image.get("camera_info"), width, height, background_image,
                                              image.get("model_3d_info"))
        if splat_outputs is not None:
            output_image, output_mask, normal_image = splat_outputs
        else:
            load_image_node = nodes.LoadImage()
            output_image, _ = load_image_node.load_image(image=image["image"])
            _, output_mask = load_image_node.load_image(image=image["mask"])
            normal_image, _ = load_image_node.load_image(image=image["normal"])

        video = None
        if image.get("recording", ""):
            video = InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(image["recording"]))

        file_3d = None
        mesh_path = ""
        preview_file = None
        if model_3d is not None:
            file_3d = model_3d
            preview_file = _file_for_preview(model_3d)
        elif mesh is None and model_file and model_file != "none":
            model_path = Path(folder_paths.get_annotated_filepath(model_file))
            if not model_path.is_file():
                model_path = Path(folder_paths.get_output_directory()) / model_file
                model_file = f"{model_file} [output]"
            file_3d = Types.File3D(str(model_path))
            mesh_path = model_file
        elif mesh is not None:
            glb = mesh_item_to_glb_bytes(mesh, 0)
            if glb is None:
                raise ValueError("TBG Preview Splat or 3D & Animation: mesh is empty (no vertices/faces).")
            file_3d = Types.File3D(BytesIO(glb), file_format="glb")
            mesh_path = ""
            preview_file = _file_for_preview(file_3d)

        preview_ui = Trellis2PreviewUI(
            preview_file or model_file or "none", image["camera_info"], background_image, width, height
        )

        return IO.NodeOutput(
            output_image,
            output_mask,
            mesh_path,
            normal_image,
            image["camera_info"],
            video,
            file_3d,
            image.get("model_3d_info", []),
            width,
            height,
            ui=preview_ui,
        )


class Trellis2Load3DLegacy:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = Path(folder_paths.get_input_directory()) / "3d"
        input_dir.mkdir(parents=True, exist_ok=True)
        base_path = Path(folder_paths.get_input_directory())
        files = [
            _normalize_path(str(path.relative_to(base_path)))
            for path in input_dir.rglob("*")
            if path.suffix.lower() in {".gltf", ".glb", ".obj", ".fbx", ".stl", ".spz", ".splat", ".ply", ".ksplat"}
        ]
        model_files = sorted(set(files + _output_model_files()))
        return {
            "required": {
                "model_file": (["none"] + model_files, {"upload": "model"}),
                "image": ("LOAD_3D",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "mesh": ("MESH",),
                "model_3d": ("FILE_3D,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_FBX,FILE_3D_OBJ,FILE_3D_STL,FILE_3D_PLY,FILE_3D_SPLAT_ANY,FILE_3D_POINT_CLOUD_ANY,FILE_3D_SPLAT,FILE_3D_SPZ,FILE_3D_KSPLAT", {
                    "tooltip": "Optional 3D file, including gaussian splats. Takes priority over mesh and model_file.",
                }),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "IMAGE", "LOAD3D_CAMERA", "VIDEO",
        "FILE_3D", "LOAD3D_MODEL_INFO", "INT", "INT",
    )
    RETURN_NAMES = (
        "image", "mask", "mesh_path", "normal", "camera_info", "recording_video",
        "model_3d", "model_3d_info", "width", "height",
    )
    FUNCTION = "execute"
    CATEGORY = "3d"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, model_file, input_types=None):
        return Trellis2Load3D.validate_inputs(model_file, input_types=input_types)

    def execute(self, model_file, image, width, height, background_image=None, mesh=None, model_3d=None):
        if background_image is not None:
            height, width = background_image.shape[1:3]
        if isinstance(image, str) and image:
            image = json.loads(image)
        if not image:
            viewer_background = background_image
            if background_image is None:
                background_image = torch.zeros((1, 1, 1, 3))
            output_mask = background_image.new_zeros((background_image.shape[0], background_image.shape[1], background_image.shape[2]))
            file_3d = model_3d
            if file_3d is None and mesh is not None:
                glb = mesh_item_to_glb_bytes(mesh, 0)
                if glb is None:
                    raise ValueError("TBG Preview Splat or 3D & Animation: mesh is empty (no vertices/faces).")
                file_3d = Types.File3D(BytesIO(glb), file_format="glb")
            preview_file = _valid_preview_file(model_file)
            if file_3d is not None:
                preview_file = _file_for_preview(file_3d)
            preview_ui = Trellis2PreviewUI(preview_file, {}, viewer_background, width, height)
            result_args = (
                background_image,
                output_mask,
                "",
                background_image,
                {},
                None,
                file_3d,
                [],
                width,
                height,
            )
            splat_outputs = _render_splat_outputs(model_3d, None, width, height, viewer_background,
                                                  image.get("model_3d_info"))
            if splat_outputs is not None:
                result_args = (
                    splat_outputs[0], splat_outputs[1], result_args[2], splat_outputs[2],
                    *result_args[4:],
                )
            return {"result": result_args, "ui": preview_ui.as_dict()}
        result = Trellis2Load3D.execute(model_file, image, width, height, background_image=background_image, mesh=mesh, model_3d=model_3d)
        if result.ui is None:
            return result.args
        return {"result": result.args, "ui": result.ui.as_dict()}


class Trellis2Load3DExtension(ComfyExtension):
    async def get_node_list(self):
        return [Trellis2Load3D]


async def comfy_entrypoint():
    return Trellis2Load3DExtension()
