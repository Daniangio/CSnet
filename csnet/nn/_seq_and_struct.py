import functools
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.utils import add_tags_to_module
from geqtrain.utils._global_options import register_fields


@compile_mode("script")
class SeqAndStructModule(GraphModuleMixin, torch.nn.Module):
    """
    """

    def __init__(
        self,
        structure_field="structure_correction",
        in_field=AtomicDataDict.NODE_ATTRS_KEY,
        out_field="node_output",
        table_dim: int = 4800,
        out_irreps=Irreps("0e"),
        fc        = ScalarMLPFunction,
        fc_kwargs = {"mlp_latent_dimensions": [256, 256]},
        # Other
        irreps_in = None,
    ):
        super().__init__()
        self.structure_field = structure_field
        self.in_field   = in_field
        self.out_field  = out_field
        self.out_irreps = out_irreps
        self.table_dim  = table_dim
        
        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
        )

        fc = functools.partial(fc, **fc_kwargs)
        self.fc = fc(
            mlp_input_dimension=irreps_in[self.in_field].dim,
            mlp_output_dimension=self.table_dim,
            use_layer_norm=True,
        )

        self.irreps_out.update({
            self.out_field: self.out_irreps,
            self.structure_field: irreps_in[self.structure_field],
        })

        register_fields(node_fields=[self.structure_field])

        # Create a 1D parameters tensor of length "table_dim"
        self.param_tensor = torch.nn.Parameter(torch.randn(self.table_dim))
        
        add_tags_to_module(self, 'dampen')

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        atom_centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
        structure_corr = data[self.structure_field][atom_centers]
        x = data[self.in_field][atom_centers]
        x = self.fc(x)

        # Perform softmax on x
        x = torch.nn.functional.softmax(x, dim=-1)

        # Select the output value of x as weighted sum of values in param tensor
        x = torch.matmul(x, self.param_tensor).unsqueeze(-1)

        data[self.out_field] = x + structure_corr

        return data