import os
import sys
import functools
import threading

SETUP_LOCK = threading.Lock()
SETUP_DONE = False


def __setup_log(settings):
    from .utils import set_log_level
    if "TORCH_LOG_LEVEL" in settings:
        set_log_level(settings["TORCH_LOG_LEVEL"])


def __setup_path(settings):
    from .utils import log_pytorch
    for path in settings["EXTRA_PYTHON_PATH"]:
        if os.path.isdir(path):
            if str(path) not in sys.path:
                sys.path.append(str(path))
            log_pytorch("Registered to sys.path: {}".format(path), level=1)
    __setup_path.done = True


def __check_pytorch(settings):
    if not settings["TORCH_ENABLED"]:
        return
    try:
        import torch
        torch.cuda.init()
    except (ImportError, RuntimeError, AssertionError):
        if sys.argv[1:][:1] == ["runserver"] or "gunicorn" in sys.modules or "TORCH_ENABLED" in os.environ:
            raise ImportError(
                "Pytorch cuda runtime is not available, please install pytorch with cuda support. "
                "If you want to run without pytorch, please set TORCH_ENABLED=False in your settings, "
                "or set the environment variable TORCH_ENABLED=0."
            )
        import warnings
        warnings.warn("Pytorch cuda runtime is not available, skipping pytorch ops...")
        settings["TORCH_ENABLED"] = False


def __setup_torch_mode(settings):
    if not settings["TORCH_ENABLED"]:
        return
    from .utils import log_pytorch
    log_pytorch("Initializing Pytorch", level=1)
    import torch
    from torch.backends import cuda, cudnn
    torch.cuda.init()
    cudnn.enabled = True
    cudnn.benchmark = True  # Improves training speed.
    cudnn.allow_tf32 = True  # Improves numerical accuracy.
    cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.set_grad_enabled(False)


def __setup_torch_extensions(settings):
    if not settings["TORCH_ENABLED"]:
        return
    from .utils import log_pytorch
    log_pytorch(
        "Initializing Pytorch Extensions: with{} compiling custom ops"
        .format("out" * settings["TORCH_WITHOUT_CUSTOM_OPS_COMPILE"]), level=1
    )
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    from torch_utils.ops import grid_sample_gradfix
    from torch_utils.ops import conv2d_gradfix
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.
    conv2d_gradfix.enabled = True  # Improves training speed.
    if not settings["TORCH_WITHOUT_CUSTOM_OPS_COMPILE"]:
        try:
            upfirdn2d._init()  # NOQA
            bias_act._init()  # NOQA
            filtered_lrelu._init()  # NOQA
        except RuntimeError:
            settings["TORCH_WITHOUT_CUSTOM_OPS_COMPILE"] = True
    if settings["TORCH_WITHOUT_CUSTOM_OPS_COMPILE"]:
        def fallback_ops(func):
            @functools.wraps(func)
            def wrapper(*args, impl=None, **kwargs):  # noqa
                return func(*args, **kwargs)
            return wrapper
        upfirdn2d.upfirdn2d = fallback_ops(upfirdn2d._upfirdn2d_ref)  # NOQA
        bias_act.bias_act = fallback_ops(bias_act._bias_act_ref)  # NOQA
        filtered_lrelu.filtered_lrelu = fallback_ops(filtered_lrelu._filtered_lrelu_ref)  # NOQA
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def __setup_torch_device(settings):
    if not settings["TORCH_ENABLED"]:
        return
    from .utils import log_pytorch
    log_pytorch("Initializing Pytorch Device with: {}".format(settings["TORCH_DEVICE"]), level=1)
    import torch
    torch.cuda.set_device(settings["TORCH_DEVICE"])
    torch.cuda.empty_cache()


def __setup_seed(settings):
    if not settings["TORCH_ENABLED"]:
        return
    from .utils import log_pytorch
    log_pytorch("Initializing Pytorch Generator with seed: {}".format(settings["TORCH_SEED"]), level=1)
    import numpy as np
    import torch
    np.random.seed(settings["TORCH_SEED"])
    torch.manual_seed(settings["TORCH_SEED"])


def setup(settings):
    global SETUP_DONE
    if not SETUP_DONE:
        with SETUP_LOCK:
            if not SETUP_DONE:
                __setup_log(settings)
                __setup_path(settings)
                __check_pytorch(settings)
                __setup_torch_mode(settings)
                __setup_torch_extensions(settings)
                __setup_torch_device(settings)
                __setup_seed(settings)
                SETUP_DONE = True


# Encapsulation
nn_module = type(sys)(__name__)
nn_module.setup = setup
sys.modules[__name__] = nn_module
del nn_module
