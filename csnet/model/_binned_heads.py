import logging
from typing import List, Optional

from geqtrain.nn import SequentialGraphNetwork
from geqtrain.utils import Config
from torch.utils.data import ConcatDataset
from csnet.nn import BinnedReadoutModule


def BinnedHeads(model, config: Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    '''
    instanciates a layer with multiple BinnedReadoutModules
    '''

    logging.info("--- Building Heads Module ---")

    layers = {
        "wrapped_model": model,
    }

    for head_tuple in config.get("binned_heads", []):
        assert isinstance(head_tuple, List) or isinstance(head_tuple, tuple), f"Elements of 'heads' must be tuples ([field], out_field, out_irreps). Found type {type(head_tuple)}"

        if len(head_tuple) == 4:
            field, out_field, min_value, max_value = head_tuple
        else:
            raise Exception(f"Elements of 'binned_heads' must be tuples of the following type (field, out_field, min_value, max_value, bins).")

        layers.update({
            f"head_{out_field}": (
                BinnedReadoutModule,
                dict(
                    field=field,
                    out_field=out_field,
                    min_value=float(min_value),
                    max_value=float(max_value),
                ),
            ),
        })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )