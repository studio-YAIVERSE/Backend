"""
This package is a wrapper of the backend's pytorch implementation of GET3D.
Init pytorch with `init` function before using `inference`, the encapsulated inference function.
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from io import BytesIO
    from .utils import inference_result
del TYPE_CHECKING


def init(settings: "dict") -> "None":
    """
    :param settings: A dictionary of settings.
    :return: None

    Configure your settings (`TORCH_SETTINGS`) as below:

    >>> import pathlib
    >>> import os
    >>>
    >>> BASE_DIR = pathlib.Path(os.getcwd())  # YOUR_BASE_DIR
    >>>
    >>> TORCH_SETTINGS = {
    ...     "BASE_DIR": BASE_DIR,
    ...     "GET3D_PATH": BASE_DIR / "GET3D",  # Path to GET3D (for import and model initialization)
    ...     "TORCH_ENABLED": True,  # False disables all torch operations
    ...     "TORCH_LOG_LEVEL": 2,  # 0: silent, 1: call, 2: 1 + process, 3: 2 + nada output
    ...     "TORCH_WARM_UP_ITER": 10,  # Number of warm up iterations
    ...     "TORCH_WITHOUT_CUSTOM_OPS_COMPILE": False,  # True disables custom c++ ops, instead use pytorch impl
    ...     "TORCH_DEVICE": "cuda:0",  # Device to use
    ...     "NADA_WEIGHT_DIR": BASE_DIR / "weights/get3d_nada",  # Path of NADA weights
    ...     "CLIP_MAP_PATH": BASE_DIR / "weights/clip_map/checkpoint_group.pt",  # Path of embedding group
    ...     "MODEL_OPTS": {  # Model initialization kwargs which is compatible with script arguments
    ...         'latent_dim': 512,
    ...         'one_3d_generator': True,
    ...         'deformation_multiplier': 1.,
    ...         'use_style_mixing': True,
    ...         'dmtet_scale': 1.,
    ...         'feat_channel': 16,
    ...         'mlp_latent_channel': 32,
    ...         'tri_plane_resolution': 256,
    ...         'n_views': 1,
    ...         'render_type': 'neural_render',  # or 'spherical_gaussian'
    ...         'use_tri_plane': True,
    ...         'tet_res': 90,
    ...         'geometry_type': 'conv3d',
    ...         'data_camera_mode': 'shapenet_car',
    ...         'n_implicit_layer': 1,
    ...         'cbase': 32768,
    ...         'cmax': 512,
    ...         'fp32': False
    ...     },
    ...     "TORCH_SEED": 0,  # Random seed
    ...     "TORCH_RESOLUTION": 1024  # Resolution of the output image
    ... }

    """
    from .register import construct_all
    construct_all(settings)


def inference(name: "str", target: "Union[str, BytesIO]") -> "inference_result":
    """
    :param name: str (3D object's name)
    :param target: str or BytesIO (target text or image to make 3D object from)
    :return: inference_result
        (result tuple containing 4 attributes:
         "file", "thumbnail", "voxelized_file", "voxelized_thumbnail")
    """
    global inference_impl
    if inference_impl is not None:
        return inference_impl(name, target)
    from .register import is_constructed
    if is_constructed():
        from .api import inference_impl
        return inference_impl(name, target)
    else:
        from .fallback import fallback_inference
        return fallback_inference()


inference_impl = None
