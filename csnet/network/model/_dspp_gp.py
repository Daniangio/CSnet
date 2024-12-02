import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.data import AtomicDataDict

from csnet.network.nn import DSPPGPModule


def DSPPGP(model: GraphModuleMixin, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> DSPPGPModule:
    logging.info("--- Building DSPPGP Module ---")

    layers = {
        "wrapped_model": model,
        "dsppgp": (DSPPGPModule, dict(
            field=AtomicDataDict.NODE_OUTPUT_KEY,
            out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
