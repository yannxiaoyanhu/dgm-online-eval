import inspect

from .encoder import Encoder
from .inception import InceptionEncoder
from .clip import CLIPEncoder
from .dinov2 import DINOv2Encoder
MODELS = {
    "inception": InceptionEncoder,
    "clip": CLIPEncoder,
    "dinov2": DINOv2Encoder,
    "I3D": NotImplementedError,
    "VGGish": NotImplementedError,
    "swav": NotImplementedError,
}
FEATURE_DIMS = {
    "inception": 2048,
    "clip": 1024,
    "dinov2": 1024,
    "I3D": 400,
    "VGGish": 128,
    "swav": 2048,
}


def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""

    model_cls = MODELS[model_name]

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:] # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    assert isinstance(encoder, Encoder), "Can only get representations with Encoder subclasses!"

    return encoder.to(device)
