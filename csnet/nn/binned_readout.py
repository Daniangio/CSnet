from typing import Optional
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class BinnedReadoutModule(GraphModuleMixin, torch.nn.Module):
    '''
    out_irreps options evaluated in the following order:
        1) o3.Irreps obj
        2) str castable to o3.Irreps obj (eg: 1x0e)
        3) a irreps_in key (and get its o3.Irreps)
        4) same irreps of out_field (if out_field in GraphModuleMixin.irreps_in dict)
        if out_irreps=None is passed, then option 4 is triggered is valid, else 5)
        5) if none of the above: outs irreps of same size of field
        if out_irreps is not provided it takes out_irreps from yaml
    '''

    def __init__(
        self,
        field: str,
        min_value: float,
        max_value: float,
        out_field: Optional[str] = None,
        irreps_in=None, # if output is only scalar, this is required
    ):
        super().__init__()
        # --- start definition of input/output irreps --- #

        # define input irreps
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.bins = irreps_in[self.field].dim
        self.bin_width = (self.max_value - self.min_value) / self.bins
        self.register_buffer("bin_centers", self.min_value + self.bin_width * (torch.arange(self.bins) + 0.5))

        # define output irreps
        self.out_field = out_field or field
        
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps("1x0e")},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        data[self.out_field] = (torch.softmax(features, dim=-1) * self.bin_centers).sum(dim=-1, keepdim=True)

        return data