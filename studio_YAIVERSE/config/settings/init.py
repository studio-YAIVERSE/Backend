from threading import RLock
__init_lock = RLock()


def __init_path(extra_import_path):
    import sys
    import os.path
    for path in extra_import_path:
        if os.path.isdir(path):
            if str(path) not in sys.path:
                sys.path.append(str(path))
            print("Registered to sys.path: {}".format(path))
    __init_path.done = True


def __init_torch_extensions():
    print("Initializing Pytorch Extensions")
    try:
        import torch
        from torch.backends import cuda, cudnn
        from torch_utils.ops import upfirdn2d
        from torch_utils.ops import bias_act
        from torch_utils.ops import filtered_lrelu
        from torch_utils.ops import grid_sample_gradfix
        from torch_utils.ops import conv2d_gradfix
    except ModuleNotFoundError:
        assert getattr(__init_path, "done", False), "call __init_path() before __init_extensions()"
        raise
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


def __init_torch_device(device):
    print("Initializing Pytorch Device with: {}".format(device))
    import torch
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()


def __init_seed(seed):
    print("Initializing Pytorch Generator with seed: {}".format(seed))
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_all(**settings):
    with __init_lock:
        __init_path(settings['EXTRA_IMPORT_PATH'])
        if settings['TORCH_ENABLED']:
            __init_torch_extensions()
            __init_torch_device(settings['TORCH_DEVICE'])
            __init_seed(settings['TORCH_SEED'])
