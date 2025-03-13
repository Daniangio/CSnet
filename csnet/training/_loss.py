import math
from typing import Dict
import gpytorch
import torch.nn

from geqtrain.train._loss import SimpleLoss, SimpleGraphLoss
from geqtrain.data import AtomicDataDict
from geqtrain.nn import SequentialGraphNetwork


class SimpleNodeLoss(SimpleGraphLoss):
    """
    """

    def __init__(self, func_name, params = ...):
        super().__init__(func_name, params)

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, corrections, center_nodes_filter = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key, ref_key) * not_nan_filter
        if mean:
            # penalty = sum(correction ** 2 for correction in corrections)
            return loss.sum() / not_nan_filter.sum() # + penalty
        loss[~not_nan_filter.bool()] = torch.nan
        return loss
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key, not_nan_pred_filter, ref_key, not_nan_ref_filter = super().prepare(pred, ref, key, **kwargs)
        center_nodes_filter = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
        num_atoms = len(pred[AtomicDataDict.POSITIONS_KEY])
        if len(pred_key) == num_atoms:
            pred_key = pred_key[center_nodes_filter]
            not_nan_pred_filter = not_nan_pred_filter[center_nodes_filter]
        
        if len(ref_key)  == num_atoms:
            ref_key  = ref_key [center_nodes_filter]
            not_nan_ref_filter = not_nan_ref_filter[center_nodes_filter]
        
        pred_key = pred_key.view_as(ref_key)
        not_nan_filter = not_nan_pred_filter * not_nan_ref_filter

        if hasattr(self, "rereference") and self.rereference:
            def rereference(y_pred, y_true, dataset_idcs, node_types, alpha=1.):
                """
                Calculate and correct systematic referencing offsets for each atom type and dataset_id in the batch.
                """
                unique_node_types = torch.unique(node_types)
                unique_dataset_ID = torch.unique(dataset_idcs)
                for atom in unique_node_types:
                    for dataset_id in unique_dataset_ID:
                        mask = ((node_types == atom).flatten() * (dataset_idcs == dataset_id))
                        if mask.sum() > 0:
                            correction = alpha * torch.nanmean(y_true[mask] - y_pred[mask])
                            y_pred = y_pred.clone() # Avoid in-place operations
                            y_pred[mask] += correction
                            yield correction

            epoch = kwargs.get('epoch', None)
            alpha_breakeven_epoch = self.alpha_breakeven_epoch if hasattr(self, "alpha_breakeven_epoch") else 0
            alpha = math.tanh(epoch - alpha_breakeven_epoch)/2 + 0.5 if epoch is not None else 1.
            node_centers = ref[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            # Find the batch index for each node_center
            batch_indices = torch.searchsorted(pred[f'{key}_slices'], node_centers, right=False) - 1
            # Apply offsets to targets
            corrections = list(rereference(pred_key, ref_key, pred[AtomicDataDict.BATCH_KEY][batch_indices], pred[AtomicDataDict.ATOM_NUMBER_KEY][node_centers], alpha))
        else:
            corrections = []

        return pred_key, ref_key, not_nan_filter, corrections, center_nodes_filter


class NodeSimpleEnsembleLoss(SimpleNodeLoss):
    """
    """
    def __init__(self, func_name, params=...):
        super().__init__(func_name, params)
        self.initial_temperature = 100.0
        self.max_temperature = 10.0
        self._factor = math.e**(1/self.max_temperature)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, corrections, center_nodes_filter = self.prepare(pred, ref, key, **kwargs)
        
        ensemble_indices = pred['ensemble_index']
        unique_ensembles = torch.unique(ensemble_indices)
        batch_indices    = pred['batch'][center_nodes_filter]
        unique_batches   = torch.unique(batch_indices)
        
        weighted_loss = 0.0
        total_weight  = 0.0

        epoch = kwargs.get('epoch', 0)
        
        peak_epoch = self.peak_epoch if hasattr(self, "peak_epoch") else 50.
        temperature = self.initial_temperature - (self.initial_temperature - self.max_temperature) * min(epoch / peak_epoch, 1.0)
        
        for ensemble in unique_ensembles:
            ensemble_batch_mask = (ensemble_indices == ensemble)
            ensemble_batches = unique_batches[ensemble_batch_mask]
            ensemble_mask = torch.isin(batch_indices, ensemble_batches)
            ensemble_pred = pred_key[ensemble_mask]
            ensemble_ref  = ref_key [ensemble_mask]
            ensemble_not_nan = not_nan_filter[ensemble_mask]
            
            distances = torch.abs(ensemble_pred - ensemble_ref).detach()
            ensemble_batch_indices = batch_indices[ensemble_mask]
            
            # Calculate weights for each atom independently
            weights = torch.zeros_like(distances)
            n_atoms_per_ensemble_batch = sum(ensemble_batch_indices == ensemble_batch_indices[0])
            
            for atom_idx in range(n_atoms_per_ensemble_batch):
                atom_mask = slice(atom_idx, None, n_atoms_per_ensemble_batch)
                atom_distances = distances[atom_mask]
                min_distance = atom_distances.min()
                weights[atom_mask] = self._factor * torch.exp(-atom_distances / (min_distance * temperature))
                
            weighted_loss += (self.func(ensemble_pred, ensemble_ref) * weights * ensemble_not_nan).sum()
            total_weight  += (weights * ensemble_not_nan).sum()
        
        if mean:
            return weighted_loss / total_weight
        
        loss = self.func(pred_key, ref_key) * not_nan_filter
        loss[~not_nan_filter.bool()] = torch.nan
        return loss


class GPLoss(SimpleLoss): # Marginal Log Likelihood
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        super(GPLoss, self).__init__(func_name, params)
        self.likelihood = None
        self.mll = None
        
    def init_loss(self, model, num_data, obj=gpytorch.mlls.VariationalELBO, beta=0.05):
        self.likelihood = model.likelihood
        self.obj = obj(self.likelihood, model.gp_module, num_data=num_data, beta=beta)
    
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, corrections = self.prepare(pred, ref, key, **kwargs)
        
        assert mean
        loss = -self.obj(pred_key, ref_key.transpose(0, 1))
        # penalty = sum(correction ** 2 for correction in corrections)
        return loss # + 0.1 * penalty

    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key = pred.get(key)
        ref_key = ref.get(key, ref.get(key[:-5]))
        
        center_nodes_filter = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
        if len(ref_key) > len(center_nodes_filter):
            ref_key = ref_key[center_nodes_filter]
        
        if hasattr(self, "rereference") and self.rereference:
            def rereference(y_pred, y_true, dataset_idcs, node_types, alpha=1.):
                """
                Calculate and correct systematic referencing offsets for each atom type and dataset_id in the batch.
                """
                unique_node_types = torch.unique(node_types)
                unique_dataset_ID = torch.unique(dataset_idcs)
                y = y_pred.mean
                corrections = []
                for atom in unique_node_types:
                    for dataset_id in unique_dataset_ID:
                        mask = ((node_types == atom).flatten() * (dataset_idcs == dataset_id))
                        if mask.sum() > 0:
                            correction = alpha * torch.mean(y_true[mask] - y.mean(dim=0)[mask])
                            y = y.clone() # Avoid in-place operations
                            y[:, mask] += correction.item()
                            corrections.append(correction)
                return y_pred.__class__(y, y_pred.lazy_covariance_matrix), corrections

            epoch = kwargs.get('epoch', None)
            alpha_breakeven_epoch = self.alpha_breakeven_epoch if hasattr(self, "alpha_breakeven_epoch") else 0
            alpha = math.tanh(epoch - alpha_breakeven_epoch)/2 + 0.5 if epoch is not None else 1.
            node_centers = ref[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            # Find the batch index for each node_center
            batch_indices = torch.searchsorted(pred[f'{key}_slices'], node_centers, right=False) - 1
            # Apply offsets to targets
            pred_key, corrections = rereference(pred_key, ref_key, pred[AtomicDataDict.DATASET_INDEX_KEY][batch_indices], pred[AtomicDataDict.ATOM_NUMBER_KEY][node_centers], alpha)
        else:
            corrections = []

        return pred_key, ref_key, corrections


class GPMAELoss(GPLoss):
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        super(GPLoss, self).__init__(func_name, params)
    
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, corrections = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key.mean.transpose(-1, -2), ref_key)
        if mean:
            loss = loss.mean()
            # penalty = sum(correction ** 2 for correction in corrections)
            return loss # + penalty
        return self.likelihood(loss).mean


class GPdpllLoss(GPLoss): # Deep Predictive Log Likelihood
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        super(GPdpllLoss, self).__init__(func_name, params)
        
    def init_loss(self, model: SequentialGraphNetwork, num_data, obj=gpytorch.mlls.DeepPredictiveLogLikelihood, beta=0.05):
        try:
            self.likelihood = model.get_param('likelihood')
        except AttributeError as e:
            raise AttributeError("Are you missing a Gaussian Process module in your modules?") from e
        self.obj = obj(self.likelihood, model.get_module('dsppgp'), num_data=num_data, beta=beta)


class GPdpllMAELoss(GPdpllLoss):
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        super(GPdpllMAELoss, self).__init__(func_name, params)
    
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, corrections = self.prepare(pred, ref, key, **kwargs)
        loss = self.func(pred_key.mean, ref_key.transpose(0, 1))
        loss = loss.mean(dim=-2)
        if mean:
            loss = loss.mean()
            # penalty = sum(correction ** 2 for correction in corrections)
            return loss # + penalty
        return loss


class GNLLoss(SimpleNodeLoss): # Uncertainty-aware loss.
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):

        self.func_name = func_name
        for key, value in params.items():
            setattr(self, key, value)

        # instanciates torch.nn loss func
        self.func = torch.nn.GaussianNLLLoss()

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, corrections, center_nodes_filter = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key, ref_key, pred['uncertainty']**2) * not_nan_filter
        if mean:
            return loss.sum() / not_nan_filter.sum()
        loss[~not_nan_filter.bool()] = torch.nan
        return loss


class GNLNodeLoss(SimpleNodeLoss): # Uncertainty-aware loss.
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):

        self.func_name = func_name
        for key, value in params.items():
            setattr(self, key, value)

        # instanciates torch.nn loss func
        self.func = torch.nn.GaussianNLLLoss(reduction='none')

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, corrections, center_nodes_filter = self.prepare(pred, ref, key, **kwargs)
        uncertainty = pred['uncertainty'][center_nodes_filter]

        loss = self.func(pred_key, ref_key, uncertainty**2) * not_nan_filter
        if mean:
            return loss.sum() / not_nan_filter.sum()
        loss[~not_nan_filter.bool()] = torch.nan
        return loss