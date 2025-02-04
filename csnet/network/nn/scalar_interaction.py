import math
import functools
import torch
import torch.nn.functional as F

from typing import Optional, Union
from torch_scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from csnet.network.nn import ScalarMLPFunction

from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    GraphModuleMixin,
    BesselBasis,
)
from geqtrain.nn.cutoffs import TanhCutoff
from geqtrain.nn._film import FiLMFunction


@compile_mode("script")
class ScalarInteractionModule(GraphModuleMixin, torch.nn.Module):

    '''
    '''

    # saved params
    num_layers: int
    node_invariant_field: str
    out_field: str

    def __init__(
        self,
        # required params
        num_layers: int,
        r_max:      float,
        # optional params
        out_irreps: Optional[Union[o3.Irreps, str]] = None,
        # radial embedding
        num_basis: int = 8,
        # cutoffs
        TanhCutoff_n: float = 6.,
        # general hyperparameters:
        node_invariant_field   = AtomicDataDict.NODE_ATTRS_KEY,
        out_field              = AtomicDataDict.EDGE_FEATURES_KEY,
        latent_dim: int        = 64,

        # MLP parameters:
        latent                 = ScalarMLPFunction,
        latent_kwargs          = {},

        # Other:
        irreps_in=None,
    ):
        super().__init__()

        assert (num_layers >= 1)

        # save parameters
        self.num_layers             = num_layers
        self.r_max                  = r_max
        self.num_basis              = int(num_basis)
        self.tanh_cutoff_n          = float(TanhCutoff_n)

        self.node_invariant_field   = node_invariant_field
        self.out_field              = out_field
        self.latent_dim             = latent_dim

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.node_invariant_field],
        )

        latent = functools.partial(latent, **latent_kwargs)

        if out_irreps is None:
            out_irreps = o3.Irreps(f"{self.latent_dim}x0e")
        self.out_irreps = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)

        self.out_multiplicity = self.out_irreps[0].mul
        
        self.interaction_blocks = torch.nn.ModuleList([])
        for layer_index in range(self.num_layers):
            self.interaction_blocks.append(
                ScalarInteractionBlock(
                    layer_index=layer_index,
                    parent=self,
                    latent=latent,
                )
            )

        # - End Interaction Layers - #

        self.irreps_out.update({self.out_field: self.out_irreps})

    def forward(self, data: AtomicDataDict.Type, recycles: int = 3) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        num_edges: int  = edge_index.shape[1]

        # Initialize state
        latents = torch.zeros((num_edges, self.latent_dim), dtype=torch.float32, device=edge_index.device)

        for recycle in range(recycles):
            for interaction_block in self.interaction_blocks:
                latents = interaction_block(data=data, latents=latents)

        data[self.out_field] = latents
        return data


@compile_mode("script")
class ScalarInteractionBlock(torch.nn.Module):

    def __init__(
        self,
        layer_index: int,
        parent: ScalarInteractionModule,
        latent: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.layer_index = layer_index
        self.node_invariant_field = parent.node_invariant_field
        self.latent_dim = parent.latent_dim
        self.node_attrs_embedding_dim = parent.irreps_in[parent.node_invariant_field].num_irreps

        self.basis  = BesselBasis(parent.r_max, parent.num_basis)
        self.cutoff = TanhCutoff(parent.r_max, parent.tanh_cutoff_n)
        
        # FiLM layer for conditioning on graph input features
        self.film = None

        # MLP to compute edge features from inputs
        self.edge_feature_mlp = latent(
            # Node invariants for center and neighbor (chemistry) + edge invariants for the edge (radius).
            mlp_input_dimension=(2 * self.node_attrs_embedding_dim + parent.num_basis),
            mlp_output_dimension=self.latent_dim,
            use_layer_norm=True,
        )
        
        # Attention mechanism for pooling
        self.attn_mlp = latent(
            mlp_input_dimension=(self.latent_dim + self.node_attrs_embedding_dim),
            mlp_output_dimension=1,
            use_layer_norm=True,
        )

        # MLP to update latents using pooled node features
        self.latent_update_mlp = latent(
            mlp_input_dimension=(2* self.latent_dim),
            mlp_output_dimension=self.latent_dim,
            use_layer_norm=True,
        )

        # Use embedder to modulate features based on radial distance
        self.rbf_embedder = FiLMFunction(
            mlp_input_dimension=parent.num_basis,
            mlp_latent_dimensions=[2 * self.latent_dim],
            mlp_output_dimension=self.latent_dim,
            mlp_nonlinearity='swiglu',
            zero_init_last_layer_weights=False,
            has_bias=False,
            final_non_lin='sigmoid'
        )

        self.resnet_update_param = torch.nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        data,
        latents,
    ):
        edge_center   = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_length   = data[AtomicDataDict.EDGE_LENGTH_KEY]
        node_attrs    = data[self.node_invariant_field].float()

        num_nodes: int  = node_attrs.shape[0]
        
        # Step 1: Compute edge features
        edge_radial_emb = self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        edge_input = torch.cat([node_attrs[edge_center], node_attrs[edge_neighbor], edge_radial_emb], dim=-1)
        edge_features = self.edge_feature_mlp(edge_input)

        # Step 2: Attention-based pooling over center atom
        central_attrs = node_attrs[edge_center]  # Central atom is the source node
        attn_input = torch.cat([edge_features, central_attrs], dim=-1)
        attn_scores = self.attn_mlp(attn_input).squeeze(-1)  # [num_edges]
        attn_weights = F.softmax(attn_scores, dim=0)  # Normalize attention scores

        # Weighted sum of edge features to pool node feature
        pooled_node_features = scatter(
            attn_weights.unsqueeze(-1) * edge_features,
            edge_center,
            dim=0,
            dim_size=num_nodes,
        )

        # Step 3: Update latent tensor using pooled node features and edge features
        update_input = torch.cat([edge_features, pooled_node_features[edge_center]], dim=-1)
        updated_latent = self.latent_update_mlp(update_input)
        

        # Apply cutoff modulation, which propagates through to everything else
        updated_latent = self.rbf_embedder(updated_latent, edge_radial_emb)

        update_coeff = self.resnet_update_param.sigmoid()
        coefficient_old = torch.rsqrt(update_coeff.square() + 1)
        coefficient_new = update_coeff * coefficient_old
        latents = coefficient_old * latents + coefficient_new * updated_latent

        return latents