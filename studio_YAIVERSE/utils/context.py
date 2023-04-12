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
