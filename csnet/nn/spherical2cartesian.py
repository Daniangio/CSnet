import math
import torch

from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@torch.jit.script
def convert_spherical_to_cartesian(spherical_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of tensors from a spherical representation (l=0, l=1, l=2)
    to a 3x3 Cartesian tensor representation, flattened.

    The conversion follows the direct summation of isotropic, antisymmetric,
    and symmetric-traceless parts.

    Args:
        spherical_tensor: A tensor of shape (n_batch, 9) where:
                          - column 0: scalar part (l=0)
                          - columns 1-3: vector part for l=1 (vx, vy, vz)
                          - columns 4-8: 5 components for l=2

    Returns:
        A tensor of shape (n_batch, 9) representing the flattened 3x3 matrix.
    """
    if spherical_tensor.shape[1] != 9:
        raise ValueError("Input tensor must have 9 columns (1+3+5 components).")

    n_batch = spherical_tensor.shape[0]
    device = spherical_tensor.device

    # --- 1. Extract Components ---
    # Isotropic part (l=0)
    s_iso = spherical_tensor[:, 0]

    # Antisymmetric part (l=1)
    v_x, v_y, v_z = spherical_tensor[:, 1], spherical_tensor[:, 2], spherical_tensor[:, 3]

    # Symmetric-traceless part (l=2)
    c1, c2, c3, c4, c5 = spherical_tensor[:, 4], spherical_tensor[:, 5], spherical_tensor[:, 6], spherical_tensor[:, 7], spherical_tensor[:, 8]

    # --- 2. Build the Cartesian Tensor ---
    # Initialize an empty (n_batch, 3, 3) tensor
    cartesian_tensor = torch.zeros((n_batch, 3, 3), device=device)
    
    # Pre-calculate the constant for the l=2 part
    sqrt6 = math.sqrt(6)

    # Reconstruct the matrix by adding the components for each element
    # T_final = T_iso + T_anti + T_sym_traceless

    # Row 1
    cartesian_tensor[:, 0, 0] = s_iso + c1 - c2 / sqrt6
    cartesian_tensor[:, 0, 1] = -v_z + c3
    cartesian_tensor[:, 0, 2] = v_y + c4

    # Row 2
    cartesian_tensor[:, 1, 0] = v_z + c3
    cartesian_tensor[:, 1, 1] = s_iso - c1 - c2 / sqrt6
    cartesian_tensor[:, 1, 2] = -v_x + c5

    # Row 3
    cartesian_tensor[:, 2, 0] = -v_y + c4
    cartesian_tensor[:, 2, 1] = v_x + c5
    cartesian_tensor[:, 2, 2] = s_iso + 2 * c2 / sqrt6

    # --- 3. Flatten the Tensor ---
    # Reshape from (n_batch, 3, 3) to (n_batch, 9)
    return cartesian_tensor.reshape(n_batch, 9)


@compile_mode("script")
class SphericalToCartesianModule(GraphModuleMixin, torch.nn.Module):
    """
    """

    def __init__(
        self,
        field: str = "cs_tensor_spherical",
        out_iso_field: str = "cs_iso",
        out_tensor_field: str = "cs_tensor",
        irreps_in = None,
    ):
        super().__init__()
        self.field            = field
        self.out_iso_field    = out_iso_field
        self.out_tensor_field = out_tensor_field
        self._init_irreps(irreps_in=irreps_in)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        spherical_tensor = data[self.field]
        data[self.out_iso_field] = spherical_tensor[:, :1]
        cartesian_tensor = convert_spherical_to_cartesian(spherical_tensor)
        data[self.out_tensor_field] = cartesian_tensor
        return data