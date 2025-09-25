from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataDict
from csnet.nn import VariationalGPModule


def VariationalGP(model: GraphModuleMixin, config) -> VariationalGPModule:
    r"""Compute dipole moment.

    Args:
        model: the model to wrap. Must have ``"node_output"`` as an output.

    Returns:
        A ``VariationalGPModule`` wrapping ``model``.
    """
    
    return VariationalGPModule(
        func=model,
        field="node_output",
        out_field="node_output",
        out_irreps=config.get('out_irreps', None),
        num_inducing_points=config.get('num_inducing_points', 512),
    )
