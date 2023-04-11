from django.conf import settings
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from training.networks_get3d import GeneratorDMTETMesh


G_EMA: "Optional[GeneratorDMTETMesh]" = None


def get_generator_ema() -> "GeneratorDMTETMesh":
    assert G_EMA is not None
    return G_EMA


def construct_all():

    global G_EMA

    from .setup import setup
    setup()

    if not settings.TORCH_ENABLED:
        return

    if G_EMA is not None:
        return

    import torch
    from training.networks_get3d import GeneratorDMTETMesh

    device = torch.device(settings.TORCH_DEVICE)

    # Performance-related toggles.
    if settings.MODEL_OPTS["fp32"]:
        extra_kwargs = dict()
        extra_kwargs["num_fp16_res"] = 0
        extra_kwargs["conv_clamp"] = None
    else:
        extra_kwargs = {}

    print("Initializing Model for Inference...")

    G = GeneratorDMTETMesh(
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
    # G.train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    # G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    generator_ema = G.eval().requires_grad_(False).to(device)  # use same object

    print("Loading state dict from: {}".format(settings.TORCH_WEIGHT_PATH))

    model_state_dict = torch.load(settings.TORCH_WEIGHT_PATH, map_location=device)
    # G.load_state_dict(model_state_dict['G'], strict=True)
    # G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
    generator_ema.load_state_dict(model_state_dict['G_ema'], strict=True)

    print("Warming up model...")
    # TODO

    print("Successfully loaded model.")

    G_EMA = generator_ema
