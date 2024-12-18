from typing import List, Optional, Union
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils import add_tags_to_parameters

import gpytorch
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


@compile_mode("script")
class DSPPGPModule(GraphModuleMixin, DSPP):
    """
        ScaleModule applies a scaling transformation to a specified field of the input data.
        
        This module can add a per-type bias and scale by per-type standard deviation.
        The result is stored in an output field.

        Args:
            func (GraphModuleMixin): The function to apply to the data.
            field (str): The field in the input data to be scaled.
            out_field (str): The field where the output data will be stored.
            num_types (int): The number of types for the per-type bias and std.
            per_type_bias (Optional[List], optional): The per-type bias values. Defaults to None.
            per_type_std (Optional[List], optional): The per-type standard deviation values. Defaults to None.
    """

    def __init__(
        self,
        field: str,
        out_field: str,
        num_types: int,
        out_irreps: Optional[Union[Irreps, str]] = None,
        grid_range: Optional[List[float]] = None,
        num_inducing_points: Optional[int] = None,
        num_quadrature_sites = 8,
        # Scaling
        per_type_bias: Optional[List] = None,
        per_type_std: Optional[List] = None,
        # Other
        irreps_in = None,
    ):
        # Set defaults
        if grid_range is None: grid_range = [-10., 10.]
        if num_inducing_points is None: num_inducing_points = 512

        # check and init irreps
        in_irreps = irreps_in[field]
        if out_irreps is None:
            out_irreps = in_irreps[field]
        else:
            out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)

        # This module will scale the NN features so that they're nice values
        scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(*grid_range)

        self.distribution = MultitaskMultivariateNormal if out_irreps.dim > 1 else MultivariateNormal
        head = DSPPHiddenModule(
            input_dim=in_irreps.dim,
            output_dim=out_irreps.dim,
            mean_type='linear',
            num_inducing_points=num_inducing_points,
            Q=num_quadrature_sites, # Number of quadrature sites (see paper for a description of this. 5-10 generally works well).
            out_distribution=self.distribution,
        )

        super().__init__(num_quadrature_sites)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.field = field
        self.out_field = out_field
        self.scale_to_bounds = scale_to_bounds
        self.head = head

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
        )
        self.irreps_out.update({self.out_field: out_irreps})

        # Scaling
        if per_type_bias is not None:
            assert len(per_type_bias) == num_types, (
                f"Expected per_type_bias to have length {num_types}, "
                f"but got {len(per_type_bias)}"
            )
            per_type_bias = torch.tensor(per_type_bias, dtype=torch.float32)
            self.per_type_bias = torch.nn.Parameter(per_type_bias.reshape(num_types, -1))
        else:
            self.per_type_bias = None
        
        if per_type_std is not None:
            assert len(per_type_std) == num_types, (
                f"Expected per_type_std to have length {num_types}, "
                f"but got {len(per_type_std)}"
            )
            per_type_std = torch.tensor(per_type_std, dtype=torch.float32)
            self.per_type_std = torch.nn.Parameter(per_type_std.reshape(num_types, -1))
        else:
            self.per_type_std = None

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        # Exclude non-center nodes to avoid having NaNs in loss
        edge_center = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
        features = features[edge_center]
        
        # features = self.scale_to_bounds(features)
        features = self.head(features)
        mean, covar = features.mean.clone(), features.covariance_matrix.clone()

        if self.per_type_std is not None or self.per_type_bias is not None:

            # Scaling
            center_species = data[AtomicDataDict.NODE_TYPE_KEY][edge_center].squeeze(dim=-1)

            # Apply per-type std scaling if available
            if self.per_type_std is not None:
                scaling = self.per_type_std[center_species].transpose(0, 1).abs()
                mean = mean * scaling

                scaling_diag = torch.diag_embed(scaling.flatten())
                covar = scaling_diag @ covar @ scaling_diag

            # Apply per-type bias if available
            if self.per_type_bias is not None:
                mean += self.per_type_bias[center_species].transpose(0, 1)
        
        try:
            features = self.distribution(mean, covar)
        except:
            features = features * torch.nan
        data[self.out_field] = features

        return data

    # Implement __call__ to skip output validation, which does not accept dict as output
    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


@compile_mode("script")
class DSPPHiddenModule(DSPPLayer):
    def __init__(self, input_dim, output_dim, mean_type, num_inducing_points, Q, out_distribution, distribution='Cholesky'):

        if output_dim <= 1: output_dim = None
        if output_dim is not None:
            inducing_points = torch.randn(output_dim, num_inducing_points, input_dim)
            batch_shape = torch.Size([output_dim])
        else:
            inducing_points = torch.randn(num_inducing_points, input_dim)
            batch_shape = torch.Size([])

        # Variational distribution for the inducing points
        """
        - CholeskyVariationalDistribution:
            When you have a small number of inducing points.
            When you believe there are complex dependencies between the inducing points.
            When you're willing to sacrifice computational efficiency for increased model flexibility.
        
        - MeanFieldVariationalDistribution:
            When you have a large number of inducing points.
            When you believe the inducing points are relatively independent.
            When you prioritize computational efficiency over model flexibility.
        """

        assert distribution in ['Cholesky', 'MeanField']
        if distribution == 'Cholesky':
            distribution = gpytorch.variational.CholeskyVariationalDistribution
        elif distribution == 'MeanField':
            distribution = gpytorch.variational.MeanFieldVariationalDistribution
        variational_distribution = distribution(
            num_inducing_points=num_inducing_points,
            batch_shape=batch_shape,
        )

        # Base variational strategy with multitask handling
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(DSPPHiddenModule, self).__init__(variational_strategy, input_dim, output_dim, Q)

        # Mean and covariance modules
        assert mean_type in ['constant', 'linear']
        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            self.mean_module = gpytorch.means.LinearMean(input_dim, batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=input_dim,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

        self.distribution = out_distribution

        # add_tags_to_parameters(self, 'strengthen')

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return self.distribution(mean, covar)