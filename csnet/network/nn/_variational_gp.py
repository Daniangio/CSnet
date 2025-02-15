import math
from typing import Optional, Union
import torch
import gpytorch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils import add_tags_to_module


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
        num_inducing_points: int = 512,
    ):
        super().__init__()
        self.func = func
        self.field = field
        self.out_field = out_field
        self.out_field_dspp = out_field + '_dspp'

        # check and init irreps
        func_irreps_out = func.irreps_out
        in_irreps = func_irreps_out[self.field]
        if out_irreps is None:
            out_irreps = func_irreps_out[self.field]
        else:
            out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.gp_module = GaussianProcessModule(
            input_dim=in_irreps.dim,
            output_dim=out_irreps.dim,
            num_inducing_points=num_inducing_points,
        )

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-10., 10.)

        func_irreps_out.update({self.out_field: out_irreps, self.out_field_dspp: out_irreps})
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
            irreps_out=func_irreps_out,
        )


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        features = data[self.field]
        
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_module(features)

        data[self.out_field_dspp] = res
        data[self.out_field] = res.mean

        return data


@compile_mode("script")
class GaussianProcessModule(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_inducing_points):

        if output_dim > 1:
            inducing_points = torch.randn(num_inducing_points, input_dim)
            batch_shape = torch.Size([output_dim])
        else:
            inducing_points = torch.randn(output_dim, num_inducing_points, input_dim)
            batch_shape = torch.Size([])

        # Variational distribution for the inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points,
            batch_shape=batch_shape,
        )

        # Base variational strategy with multitask handling
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True
        )
        super(GaussianProcessModule, self).__init__(variational_strategy)

        # Mean and covariance modules
        self.mean_module = gpytorch.means.LinearMean(input_dim, batch_shape=batch_shape)

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(
        #         batch_shape=batch_shape,
        #         ard_num_dims=input_dim,
        #     ),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )

        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape,
                ard_num_dims=input_dim,
            ),
            num_dims=input_dim,
        )

        # add_tags_to_module(self, 'strengthen')

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)