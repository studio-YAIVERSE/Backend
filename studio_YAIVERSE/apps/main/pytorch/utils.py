from contextlib import contextmanager


@contextmanager
def at_working_directory(work_dir):
    import os
    prev = os.getcwd()
    try:
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(prev)


def inference_mode():  # supports pytorch >= 1.9
    import torch
    return getattr(torch, "inference_mode", torch.no_grad)()
