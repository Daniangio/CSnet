import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from csnet.nn import AnisotropicCorrectionModule


def AnisotropicCorr(model: GraphModuleMixin, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    """Base model architecture.

    """

    logging.info("--- Building AnisotropicCorr Module ---")

    layers = {
        "wrapped_model": model,
        "anicorr": (AnisotropicCorrectionModule, dict(
            scalar_field="node_output",
            vector_field="anisotropic_component",
            out_field="node_output",
            out_irreps=config.get('out_irreps', None),
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )