import os
import io
import torch
from collections import namedtuple
from functools import lru_cache

from django.conf import settings

from .nn import using_generator_ema, get_camera, get_clip_loss, get_clip_map, get_device
from .utils import inference_mode
from .functions import (
    inference_core_logic,
    mapping_checkpoint, load_nada_checkpoint_from_keys,
    postprocess_thumbnail, postprocess_mesh,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


inference_result = namedtuple("inference_result", ["file", "thumbnail"])


@inference_mode()
def inference(name: "str", target: "Union[str, io.BytesIO]") -> inference_result:

    log_name = "inference({})".format(", ".join("{0}={1!r}".format(*i) for i in locals().items()))
    print(log_name, ": init")

    device = get_device()
    if device is None:
        return dummy_inference()

    print(log_name, ": run clip")
    clip_loss = get_clip_loss()
    clip_map = get_clip_map()
    key_src_key_dst = mapping_checkpoint(clip_loss, clip_map, target)

    print(log_name, ": loading checkpoint from", *key_src_key_dst)
    g_ema_checkpoint = load_nada_checkpoint_from_keys(settings.NADA_WEIGHT_DIR, device, *key_src_key_dst)
    camera = get_camera()

    with using_generator_ema() as g_ema:
        g_ema.load_state_dict(g_ema_checkpoint)

        print(log_name, ": run main inference")
        geo_z = torch.randn([1, g_ema.z_dim], device=device)
        tex_z = torch.randn([1, g_ema.z_dim], device=device)
        generated_thumbnail, generated_mesh = inference_core_logic(
            g_ema, geo_z=geo_z, tex_z=tex_z, c=None, camera=camera, truncation_psi=0.7
        )

    print(log_name, ": postprocessing")
    thumbnail = postprocess_thumbnail(generated_thumbnail)
    file = postprocess_mesh(generated_mesh, name)

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
