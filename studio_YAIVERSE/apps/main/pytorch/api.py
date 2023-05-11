import os
import io
import torch
from collections import namedtuple
from functools import lru_cache

from .nn import get_generator_ema, get_device
from .utils import inference_mode
from .functions import (
    inference_core_logic,
    thumbnail_to_pil, postprocess_texture_map, format_mesh_obj, format_material,
    convert_obj_to_extension
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


inference_result = namedtuple("inference_result", ["file", "thumbnail"])


@inference_mode()
def inference_image(name: "str", image: "Any" = None, extension: "Optional[str]" = "glb") -> inference_result:
    return inference_text(name, image, extension)


@inference_mode()
def inference_text(name: "str", text: "Optional[str]" = None, extension: "Optional[str]" = "glb") -> inference_result:

    print("Running inference({})".format(", ".join("{0}={1!r}".format(*i) for i in locals().items())))

    device = get_device()
    if device is None:
        return dummy_inference()

    g_ema = get_generator_ema()

    geo_z = torch.randn([1, g_ema.z_dim], device=device)
    tex_z = torch.randn([1, g_ema.z_dim], device=device)

    c_to_compute_w_avg = None
    g_ema.update_w_avg(c_to_compute_w_avg)

    generated_thumbnail, generated_mesh = inference_core_logic(
        g_ema, geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7
    )

    img, _ = generated_thumbnail
    rgb_img = img[:, :3]
    thumbnail_img = thumbnail_to_pil(rgb_img)
    thumbnail = io.BytesIO()
    thumbnail_img.save(thumbnail, format="PNG")

    (mesh_v,), (mesh_f,), (all_uvs,), (all_mesh_tex_idx,), (tex_map,) = generated_mesh

    mesh_obj = format_mesh_obj(
        mesh_v.data.cpu().numpy(),
        all_uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        all_mesh_tex_idx.data.cpu().numpy(),
        name
    )
    material = format_material(name)
    texture_map = postprocess_texture_map(tex_map)
    file = convert_obj_to_extension(name, mesh_obj, material, texture_map, extension)

    return inference_result(file=file, thumbnail=thumbnail)


@lru_cache(maxsize=None)
def _dummy_inference_data() -> "Tuple[Optional[bytes], Optional[bytes]]":
    file_data = thumbnail_data = None
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    if os.path.exists(os.path.join(assets_dir, "dummy_file.glb")):
        with open(os.path.join(assets_dir, "dummy_file.glb"), "rb") as f:
            file_data = f.read()
    if os.path.exists(os.path.join(assets_dir, "dummy_thumbnail.png")):
        with open(os.path.join(assets_dir, "dummy_thumbnail.png"), "rb") as f:
            thumbnail_data = f.read()
    return file_data, thumbnail_data


def dummy_inference() -> inference_result:
    file_data, thumbnail_data = _dummy_inference_data()
    return inference_result(file=io.BytesIO(file_data), thumbnail=io.BytesIO(thumbnail_data))
