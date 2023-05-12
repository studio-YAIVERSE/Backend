import sys
import threading
from contextlib import contextmanager
from functools import lru_cache
from tqdm.auto import trange
import torch
from django.conf import settings

from .utils import at_working_directory

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from training.networks_get3d import GeneratorDMTETMesh
    from ..pytorch_deps.clip_loss import CLIPLoss


@lru_cache(maxsize=None)
def get_device() -> "Optional[torch.device]":
    if not settings.TORCH_ENABLED:
        return
    return torch.device(settings.TORCH_DEVICE)


# NOTE: In this project we support only single-GPU runtime,
# so we can use global lock for all GPU-related operations.
# If you want to support multi-GPU runtime, you should implement
# extra allocating logic for each GPU.
G_EMA_LOCK = threading.Lock()
G_EMA_MODULE: "Optional[GeneratorDMTETMesh]" = None


@contextmanager
def using_generator_ema():
    assert CONSTRUCTED
    with G_EMA_LOCK:
        yield G_EMA_MODULE


CAMERA: "Optional[torch.Tensor]" = None


def get_camera():
    assert CONSTRUCTED
    return CAMERA


CLIP_LOSS_MODULE: "Optional[CLIPLoss]" = None


def get_clip_loss() -> "CLIPLoss":
    assert CONSTRUCTED
    return CLIP_LOSS_MODULE


CLIP_MAP: "dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]]" = {}


def get_clip_map() -> "dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]]":
    assert CONSTRUCTED
    return CLIP_MAP


CONSTRUCTED = False


def construct_all():
    global G_EMA_MODULE, CAMERA, CLIP_LOSS_MODULE, CLIP_MAP, CONSTRUCTED

    # Condition Check
    from .setup import setup
    setup()
    if not settings.TORCH_ENABLED:
        return
    if G_EMA_MODULE is not None:
        return

    # TORCH: init device
    device = get_device()

    # CLIP: Init
    print("Initializing CLIP Loss for Inference...")
    from ..pytorch_deps.clip_loss import CLIPLoss
    clip_loss = CLIPLoss(device).eval().requires_grad_(False)

    # GET3D: Init
    print("Initializing GET3D Model for Inference...")
    if settings.MODEL_OPTS["fp32"]:
        extra_kwargs = dict()
        extra_kwargs["num_fp16_res"] = 0
        extra_kwargs["conv_clamp"] = None
    else:
        extra_kwargs = {}
    with at_working_directory(settings.BASE_DIR / "GET3D"):
        from training.networks_get3d import GeneratorDMTETMesh
        generator_ema = GeneratorDMTETMesh(
            c_dim=0,
            img_resolution=settings.TORCH_RESOLUTION,
            img_channels=3,
            mapping_kwargs=dict(num_layers=8),
            fused_modconv_default='inference_only',
            device=device,
            z_dim=settings.MODEL_OPTS["latent_dim"],
            w_dim=settings.MODEL_OPTS["latent_dim"],
            one_3d_generator=settings.MODEL_OPTS["one_3d_generator"],
            deformation_multiplier=settings.MODEL_OPTS["deformation_multiplier"],
            use_style_mixing=settings.MODEL_OPTS["use_style_mixing"],
            dmtet_scale=settings.MODEL_OPTS["dmtet_scale"],
            feat_channel=settings.MODEL_OPTS["feat_channel"],
            mlp_latent_channel=settings.MODEL_OPTS["mlp_latent_channel"],
            tri_plane_resolution=settings.MODEL_OPTS["tri_plane_resolution"],
            n_views=settings.MODEL_OPTS["n_views"],
            render_type=settings.MODEL_OPTS["render_type"],
            use_tri_plane=settings.MODEL_OPTS["use_tri_plane"],
            tet_res=settings.MODEL_OPTS["tet_res"],
            geometry_type=settings.MODEL_OPTS["geometry_type"],
            data_camera_mode=settings.MODEL_OPTS["data_camera_mode"],
            channel_base=settings.MODEL_OPTS["cbase"],
            channel_max=settings.MODEL_OPTS["cmax"],
            n_implicit_layer=settings.MODEL_OPTS["n_implicit_layer"],
            **extra_kwargs
        )
    generator_ema.eval().requires_grad_(False).to(device)

    # GET3D: Load State Dict
    print("Loading state dict from: {}".format(settings.TORCH_WEIGHT_PATH))
    model_state_dict = torch.load(settings.TORCH_WEIGHT_PATH, map_location=device)
    generator_ema.load_state_dict(model_state_dict['G_ema'], strict=True)

    # Load CLIP-feature Map
    print("Loading CLIP-feature Mapping from: {}".format(settings.CLIP_MAP_PATH))
    clip_map = torch.load(settings.CLIP_MAP_PATH, map_location=device)

    # GET3D: Warm Up
    total = settings.TORCH_WARM_UP_ITER
    geo_z = torch.randn([1, generator_ema.z_dim], device=device)
    tex_z = torch.randn([1, generator_ema.z_dim], device=device)
    for _ in trange(1, total + 1, desc="Warming up...", leave=False):
        generator_ema.update_w_avg(None)
        generator_ema.generate_3d_mesh(geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7)

    # GET3D: Get Camera
    camera = generator_ema.synthesis.generate_rotate_camera_list(n_batch=1)[5]

    # Complete
    print("Successfully loaded models.")
    G_EMA_MODULE = generator_ema
    CAMERA = camera
    CLIP_LOSS_MODULE = clip_loss
    CLIP_MAP = clip_map
    CONSTRUCTED = True


# Encapsulation
nn_module = type(sys)(__name__)
nn_module.get_device = get_device
nn_module.using_generator_ema = using_generator_ema
nn_module.get_camera = get_camera
nn_module.get_clip_loss = get_clip_loss
nn_module.get_clip_map = get_clip_map
nn_module.construct_all = construct_all
sys.modules[__name__] = nn_module
