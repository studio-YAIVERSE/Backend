from .register import using_generator_ema, get_camera, get_clip_loss, get_clip_map, get_device, load_nada_checkpoint
from .functions import generate_latent, inference_logic, map_checkpoint, postprocess_outputs
from .utils import inference_mode, log_pytorch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from io import BytesIO
    from .utils import inference_result


@inference_mode()
def inference_impl(name: "str", target: "Union[str, BytesIO]") -> "inference_result":
    log_name = "inference({})".format(", ".join("{0}={1!r}".format(*i) for i in locals().items()))

    log_pytorch(log_name, ": called", level=1)

    log_pytorch(log_name, ": run clip", level=2)
    clip_loss = get_clip_loss()
    clip_map = get_clip_map()
    key_src_key_dst = map_checkpoint(clip_loss, clip_map, target)

    log_pytorch(log_name, ": loading checkpoint from", *key_src_key_dst, level=2)
    g_ema_checkpoint = load_nada_checkpoint(*key_src_key_dst)
    camera = get_camera()
    device = get_device()
    with using_generator_ema() as g_ema:
        g_ema.load_state_dict(g_ema_checkpoint)

        log_pytorch(log_name, ": run main inference", level=2)
        geo_z = generate_latent(g_ema)
        tex_z = generate_latent(g_ema)
        t, m = inference_logic(g_ema, geo_z=geo_z, tex_z=tex_z, c=None, camera=camera, truncation_psi=0.7)

    log_pytorch(log_name, ": postprocessing", level=2)
    return postprocess_outputs(generated_thumbnail=t, generated_mesh=m, name=name)
