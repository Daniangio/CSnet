import math
from typing import Dict
import gpytorch
import torch.nn

from geqtrain.train._loss import SimpleLoss
from geqtrain.data import AtomicDataDict
from geqtrain.nn import SequentialGraphNetwork


class NodeSimpleLoss(SimpleLoss): # SimpleLoss with Rereference
    """
    """

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, corrections = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key, ref_key) * not_nan_filter
        if mean:
            penalty = sum(correction ** 2 for correction in corrections)
            return loss.sum() / not_nan_filter.sum() + penalty
        loss[~not_nan_filter.bool()] = torch.nan
        return loss
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter = super(NodeSimpleLoss, self).prepare(pred, ref, key, **kwargs)
        center_nodes_filter = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
        pred_key       = pred_key[center_nodes_filter]
        ref_key        = ref_key[center_nodes_filter]
        not_nan_filter = not_nan_filter[center_nodes_filter]

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
            corrections = list(rereference(pred_key, ref_key, pred[AtomicDataDict.DATASET_INDEX_KEY][batch_indices], pred[AtomicDataDict.ATOM_NUMBER_KEY][node_centers], alpha))
        else:
            corrections = []

        return pred_key, ref_key, not_nan_filter, corrections


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
        penalty = sum(correction ** 2 for correction in corrections)
        return loss + penalty
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key = pred.get(key)
        ref_key = ref.get(key)
        
        center_nodes_filter = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
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
            penalty = sum(correction ** 2 for correction in corrections)
            return loss + penalty
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
            penalty = sum(correction ** 2 for correction in corrections)
            return loss + penalty
        return loss


class NoiseLoss(SimpleLoss):
    """
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):

        super(NoiseLoss, self).__init__(func_name, params)
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        assert key == AtomicDataDict.NOISE_KEY
        pred_key, ref_key, not_nan_filter = super(NoiseLoss, self).prepare(pred, ref, key, **kwargs)
        ref_key = -ref_key
        return pred_key, ref_key, not_nan_filter