from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from io import BytesIO
    from .utils import inference_result
del TYPE_CHECKING

inference_impl = None


def init():
    from .register import construct_all
    construct_all()


def inference(name: "str", target: "Union[str, BytesIO]") -> "inference_result":
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
