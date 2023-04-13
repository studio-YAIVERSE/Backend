import functools
from django.conf import settings


def __setup_path():
    import sys
    import os.path
    for path in settings.EXTRA_PYTHON_PATH:
        if os.path.isdir(path):
            if str(path) not in sys.path:
                sys.path.append(str(path))
            print("Registered to sys.path: {}".format(path))
    __setup_path.done = True


def __check_pytorch():
    if not settings.TORCH_ENABLED:
        return
    try:
        import torch
        torch.cuda.init()
    except (ImportError, RuntimeError, AssertionError):
        import os
        import sys
        if sys.argv[1] == "runserver" or "gunicorn" in sys.modules or "TORCH_ENABLED" in os.environ:
            raise
        import warnings
        warnings.warn("Pytorch cuda runtime is not available, skipping pytorch ops...")
        settings.TORCH_ENABLED = False


def __setup_torch_extensions():
    if not settings.TORCH_ENABLED:
        return
    print("Initializing Pytorch Extensions")
    import torch
    from torch.backends import cuda, cudnn
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    from torch_utils.ops import grid_sample_gradfix
    from torch_utils.ops import conv2d_gradfix
    torch.cuda.init()
    cudnn.enabled = True
    cudnn.benchmark = True  # Improves training speed.
    cudnn.allow_tf32 = True  # Improves numerical accuracy.
    cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    upfirdn2d._init()  # NOQA
    bias_act._init()  # NOQA
    filtered_lrelu._init()  # NOQA
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.
    conv2d_gradfix.enabled = True  # Improves training speed.
    torch.set_grad_enabled(False)


def __setup_torch_device():
    if not settings.TORCH_ENABLED:
        return
    print("Initializing Pytorch Device with: {}".format(settings.TORCH_DEVICE))
    import torch
    torch.cuda.set_device(settings.TORCH_DEVICE)
    torch.cuda.empty_cache()


def __setup_seed():
    if not settings.TORCH_ENABLED:
        return
    print("Initializing Pytorch Generator with seed: {}".format(settings.TORCH_SEED))
    import numpy as np
    import torch
    np.random.seed(settings.TORCH_SEED)
    torch.manual_seed(settings.TORCH_SEED)


@functools.lru_cache(maxsize=None)
def setup():
    __setup_path()
    __check_pytorch()
    __setup_torch_extensions()
    __setup_torch_device()
    __setup_seed()


__all__ = ['setup']
