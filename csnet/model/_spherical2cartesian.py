import logging
from geqtrain.utils import Config

from geqtrain.nn import SequentialGraphNetwork
from csnet.nn import SphericalToCartesianModule


def SphericalToCartesian(model, config: Config) -> SequentialGraphNetwork:
    logging.info("--- Building SphericalToCartesian Module ---")

    layers = {
        "wrapped_model": model,
        "sph2cart": SphericalToCartesianModule,
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )