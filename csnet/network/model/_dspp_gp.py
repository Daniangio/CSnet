from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataDict
from csnet.network.nn import DSPPGPModule


def DSPPGP(model: GraphModuleMixin, config) -> DSPPGPModule:
    r"""Compute dipole moment.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.NODE_OUTPUT_KEY`` as an output.

    Returns:
        A ``DSPPGPModule`` wrapping ``model``.
    """
    
    return DSPPGPModule(
        func=model,
        field=AtomicDataDict.NODE_OUTPUT_KEY,
        out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        out_irreps=config.get('out_irreps', None),
        num_inducing_points=config.get('num_inducing_points', 512),
        Q=config.get('Q', 1),
    )
