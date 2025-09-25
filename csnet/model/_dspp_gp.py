import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.data import AtomicDataDict

from csnet.nn import DSPPGPModule


def DSPPGP(model: GraphModuleMixin, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> DSPPGPModule:
    logging.info("--- Building DSPPGP Module ---")

    layers = {
        "wrapped_model": model,
        "dsppgp": (DSPPGPModule, dict(
            out_field="node_output",
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
