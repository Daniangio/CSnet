import math
from typing import Optional, Union
import torch
import gpytorch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils import add_tags_to_parameters


@compile_mode("script")
class VariationalGPModule(GraphModuleMixin, torch.nn.Module):
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
        func: GraphModuleMixin,
        field: str,
        out_field: str,
        out_irreps: Optional[Union[Irreps, str]] = None,
        grid_size=512,
        grid_bounds=(-10., 10.),
    ):
        super().__init__()
        self.func = func
        self.field = field
        self.out_field = out_field

        # check and init irreps
        irreps_out = func.irreps_out
        if out_irreps is None:
            out_irreps = irreps_out[self.field]
        else:
            out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)
        irreps_out.update({self.out_field: out_irreps})

        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
            irreps_out=irreps_out,
        )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.irreps_out[self.out_field].dim,
            batch_shape=torch.Size([self.irreps_out[self.out_field].dim]),
        )
        self.gp_module = GaussianProcessModule(
            input_dim=self.irreps_out[self.field].dim,
            output_dim=self.irreps_out[self.out_field].dim,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
        )

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        features = data[self.field]
        
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_module(features)

        data[self.out_field] = res

        return data


@compile_mode("script")
class GaussianProcessModule(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, grid_size=512, grid_bounds=(-10., 10.)):
        self.grid_bounds = grid_bounds

        # Variational distribution for the inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size,
            batch_shape=torch.Size([output_dim])  # One variational distribution per task
        )

        # Base variational strategy with multitask handling
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=output_dim, # Number of tasks (output_dim)
        )
        super().__init__(variational_strategy)

        # Mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=input_dim,
                        lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                            math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                        )
                    ),
                    batch_shape=torch.Size([output_dim]),
                ),
                num_dims=input_dim, # Matches input dimensionality
                grid_size=grid_size,
            )

        add_tags_to_parameters(self, 'strengthen2x')

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)