from typing import Optional
import logging

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    EmbeddingNodeAttrs,
)
from csnet.network.nn import ScalarInteractionModule

def HeadlessScalarNodeModel(
    config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildScalarNodeModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildScalarNodeModelLayers(config):
    logging.info("--- Building Node Model ---")

    update_config(config)

    layers = {
        # -- Encode -- #
        "node_attrs": EmbeddingNodeAttrs,
    }

    layers.update({
        "interaction": (ScalarInteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        )),
        "edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
    })
    
    return layers