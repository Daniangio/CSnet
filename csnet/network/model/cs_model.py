from typing import Optional
import logging

from e3nn import o3
from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
)
from csnet.network.nn import (
    InteractionModule,
    EdgewiseReduce,
    ReadoutModule,
)


def CSModel(
    config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    logging.debug("Building model")

    update_config(config)

    layers = {
        # -- Encode --
        "node_attrs":         EmbeddingNodeAttrs,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
    }

    layers.update(
        {
            "interaction": (
            InteractionModule,
                dict(
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                    output_mul="hidden",
                    output_ls=[0, 1],
                ),
            ),
            "pooling": (
                EdgewiseReduce,
                dict(
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_FEATURES_KEY,
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "noise_head": (
                ReadoutModule,
                dict(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=AtomicDataDict.NOISE,
                    out_irreps=o3.Irreps("1x1o"),
                ),
            ),
            "head": (
                ReadoutModule,
                dict(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_OUTPUT_KEY,
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )