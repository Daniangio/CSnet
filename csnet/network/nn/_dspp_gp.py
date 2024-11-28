from typing import Optional, Union
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils import add_tags_to_parameters

import gpytorch
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP


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
        func: GraphModuleMixin,
        field: str,
        out_field: str,
        out_irreps: Optional[Union[Irreps, str]] = None,
        num_inducing_points: int = 512,
        Q = 10,
    ):
        # check and init irreps
        func_irreps_out = func.irreps_out
        in_irreps = func_irreps_out[field]
        if out_irreps is None:
            out_irreps = func_irreps_out[field]
        else:
            out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)

        head = DSPPHiddenModule(
            input_dim=in_irreps.dim,
            output_dim=out_irreps.dim,
            num_inducing_points=num_inducing_points,
            Q=Q, # Number of quadrature sites (see paper for a description of this. 5-10 generally works well).
        )

        # This module will scale the NN features so that they're nice values
        # scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-10., 10.)

        super().__init__(Q)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.func = func
        self.field = field
        self.out_field = out_field
        self.head = head

        func_irreps_out.update({self.out_field: out_irreps})
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.POSITIONS_KEY: Irreps("1o")},
            irreps_out=func_irreps_out,
        )


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        features = data[self.field]
        
        # features = self.scale_to_bounds(features)
        res = self.head(features)

        data[self.out_field] = res

        return data

    # Implement __call__ to skip output validation, which does not accept dict as output
    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


@compile_mode("script")
class DSPPHiddenModule(DSPPLayer):
# class GaussianProcessModule(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_inducing_points, Q=8):

        if output_dim > 1:
            inducing_points = torch.randn(num_inducing_points, input_dim)
            batch_shape = torch.Size([output_dim])
        else:
            inducing_points = torch.randn(output_dim, num_inducing_points, input_dim)
            batch_shape = torch.Size([])

        # Variational distribution for the inducing points
        # --- Option 1 ---

        # # variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
        # #     num_inducing_points=num_inducing_points,
        # #     batch_shape=batch_shape,
        # # )

        # --- Option 2 ---

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
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
        super(DSPPHiddenModule, self).__init__(variational_strategy, input_dim, output_dim, Q)

        # Mean and covariance modules
        # self.mean_module = gpytorch.means.LinearMean(input_dim, batch_shape=batch_shape)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(
        #         batch_shape=batch_shape,
        #         ard_num_dims=input_dim,
        #     ),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=input_dim,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )
        
        # self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
        #     gpytorch.kernels.RBFKernel(
        #         batch_shape=batch_shape,
        #         ard_num_dims=input_dim,
        #     ),
        #     num_dims=input_dim,
        # )

        add_tags_to_parameters(self, 'strengthen')

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)