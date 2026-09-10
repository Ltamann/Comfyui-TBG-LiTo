from comfy_api.latest import ComfyExtension
from .nodes import LiToLoadModel, LiToImageToSplat
from .preview_3d import Trellis2Load3D

WEB_DIRECTORY = "./web"


class TBGLiToExtension(ComfyExtension):
    async def get_node_list(self):
        return [LiToLoadModel, LiToImageToSplat, Trellis2Load3D]


async def comfy_entrypoint():
    return TBGLiToExtension()
