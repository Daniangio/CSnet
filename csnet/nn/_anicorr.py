import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils._global_options import register_fields


@compile_mode("script")
class AnisotropicCorrectionModule(GraphModuleMixin, torch.nn.Module):
    """
    """

    def __init__(
        self,
        scalar_field="node_output",
        vector_field="anisotropic_component",
        out_field="node_output",
        out_irreps=Irreps("0e"),
        # Other
        irreps_in = None,
    ):
        super().__init__()
        self.scalar_field = scalar_field
        self.vector_field = vector_field
        self.out_field    = out_field
        self.out_irreps   = out_irreps
        self.anisotropic_norm_field = "anisotropic_norm"
        self.register_buffer("scale_w", torch.as_tensor([1.0]))

        sigma_refs = torch.zeros((128,))
        sigma_refs[1]  = 31.88 # 1H
        sigma_refs[6]  = 184.0 # 13C
        sigma_refs[7]  = 263.3 # 15N
        sigma_refs[8]  = 328.5 # 17O
        sigma_refs[15] = 280.0 # 31P
        self.register_buffer("sigma_refs", torch.as_tensor(sigma_refs))
        
        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
        )
        self.irreps_out.update({
            self.out_field: self.out_irreps,
            self.anisotropic_norm_field: Irreps("0e"),
        })

        register_fields(node_fields=[self.vector_field, self.anisotropic_norm_field])

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()

        num_nodes = data['ptr'].max()
        sigma_iso = data[self.scalar_field]
        if len(sigma_iso) == num_nodes:
            sigma_iso = sigma_iso[edge_centers]
        sigma_ani = data[self.vector_field]
        if len(sigma_ani) == num_nodes:
            sigma_ani = sigma_ani[edge_centers]

        anisotropic_norm = torch.norm(sigma_ani, dim=-1, keepdim=True)
        sigma_final = self.scale_w * sigma_iso + anisotropic_norm # Adding anisotropic correction
        
        # atom_numbers = data["atom_numbers"][edge_centers].long()
        # chemical_shifts_pred = self.sigma_refs[atom_numbers] - sigma_final  # Compute chemical shift
        chemical_shifts_pred = sigma_final  # Compute chemical shift

        data[self.anisotropic_norm_field] = anisotropic_norm
        data[self.out_field] = chemical_shifts_pred

        return data