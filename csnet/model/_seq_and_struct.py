import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from csnet.nn import SeqAndStructModule


def SeqAndStructHead(model: GraphModuleMixin, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    """Base model architecture.

    """

    logging.info("--- Building SeqAndStructHead Module ---")

    layers = {
        "wrapped_model": model,
        "seqandstruct": (SeqAndStructModule, dict()),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )