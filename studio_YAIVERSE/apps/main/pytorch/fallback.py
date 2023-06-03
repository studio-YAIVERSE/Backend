import os
import io
import functools

from .utils import inference_result

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@functools.lru_cache(maxsize=None)
def _fallback_file() -> bytes:
    with open(os.path.join(ASSETS_DIR, "dummy_file.glb"), "rb") as f:
        file_data = f.read()
    return file_data


@functools.lru_cache(maxsize=None)
def _fallback_image() -> bytes:
    with open(os.path.join(ASSETS_DIR, "dummy_thumbnail.png"), "rb") as f:
        thumbnail_data = f.read()
    return thumbnail_data


def fallback_inference() -> inference_result:
    file_data, thumbnail_data = _fallback_file(), _fallback_image()
    return inference_result(
        file=io.BytesIO(file_data), thumbnail=io.BytesIO(thumbnail_data),
        voxelized_file=io.BytesIO(file_data), voxelized_thumbnail=io.BytesIO(thumbnail_data)
    )
