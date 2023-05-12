import datetime
import os
import sys
from collections import namedtuple
from contextlib import contextmanager


inference_result = namedtuple("inference_result", ["file", "thumbnail", "voxelized_file", "voxelized_thumbnail"])


def set_log_level(verbosity: int):
    assert verbosity in range(4)
    global LOG_VERBOSITY
    LOG_VERBOSITY = verbosity


def log_pytorch(*args, level: int = 1, **kwargs):
    if level <= LOG_VERBOSITY:
        kwargs.pop("file", None)
        print(datetime.datetime.now().strftime("[%d/%b/%Y %H:%M:%S]"), end=" ", file=sys.stderr)
        print(*args, **kwargs, file=sys.stderr)


def should_log(*, level: int = 1):
    return level <= LOG_VERBOSITY


LOG_VERBOSITY = 1  # default


@contextmanager
def at_working_directory(work_dir):
    prev = os.getcwd()
    try:
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(prev)


def trange(*args, **kwargs):
    try:
        from tqdm.auto import trange
        return trange(*args, **kwargs)
    except ImportError:
        return range(*args)


def inference_mode():  # supports pytorch >= 1.9
    try:
        import torch
        return getattr(torch, "inference_mode", torch.no_grad)()
    except ImportError:
        return lambda fn: fn
