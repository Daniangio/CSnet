import math
from typing import Dict
import gpytorch
import torch.nn

from geqtrain.train._loss import SimpleLoss
from geqtrain.data import AtomicDataDict
from geqtrain.nn import SequentialGraphNetwork


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
        pred_key, ref_key, has_nan, not_zeroes = self.prepare(pred, ref, key, **kwargs)
        not_zeroes = not_zeroes.transpose(0, 1)
        
        assert mean
        loss = -self.obj(pred_key, ref_key.transpose(0, 1))
        return loss
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key = pred.get(key)
        ref_key  = ref.get(key)
        has_nan = torch.isnan(ref_key.sum())
        if has_nan:
            if not (hasattr(self, "ignore_nan") and self.ignore_nan):
                raise Exception(f"Either the predicted or true property '{key}' has nan values. "
                                "If this is intended, set 'ignore_nan' to true in config file for this loss.")
            fltr = ~torch.isnan(ref_key[:, 0])
            if fltr.sum() > pred_key.mean.size(-1):
                # This means that there are 1 or more atoms which have a label but have no neighbours
                fltr = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
            ref_key = ref_key[fltr]


        if hasattr(self, "ignore_zeroes") and self.ignore_zeroes:
            not_zeroes = (~torch.all(ref_key == 0., dim=-1)).int() if len(ref_key.shape) > 1 else (ref_key != 0)
        else:
            not_zeroes = torch.ones(*ref_key.shape[:max(1, len(ref_key.shape)-1)], device=ref_key.device).int()
        not_zeroes = not_zeroes.reshape(*([-1] + [1] * (len(ref_key.shape)-1)))

        if hasattr(self, "rereference") and self.rereference:
            def rereference(y_pred, y_true, dataset_idcs, node_types, alpha=1.):
                """
                Calculate and correct systematic referencing offsets for each atom type and dataset_id in the batch.
                """
                unique_node_types = torch.unique(node_types)
                unique_dataset_ID = torch.unique(dataset_idcs)
                mean = y_pred.mean
                for atom in unique_node_types:
                    for dataset_id in unique_dataset_ID:
                        mask = ((node_types == atom).flatten() * (dataset_idcs == dataset_id))
                        if mask.sum() > 0:
                            mean[:, mask] += alpha * torch.nanmean(y_true[mask].flatten() - mean.mean(dim=0)[mask]).item()
                
                return y_pred.__class__(mean, y_pred.lazy_covariance_matrix)

            epoch = kwargs.get('epoch', None)
            alpha_breakeven_epoch = self.alpha_breakeven_epoch if hasattr(self, "alpha_breakeven_epoch") else 0
            alpha = math.tanh(epoch - alpha_breakeven_epoch)/2 + 0.5 if epoch is not None else 1.
            node_centers = ref[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            # Find the batch index for each node_center
            batch_indices = torch.searchsorted(pred[f'{key}_slices'], node_centers, right=False) - 1
            # Apply offsets to targets
            pred_key = rereference(pred_key, ref_key, pred[AtomicDataDict.DATASET_INDEX_KEY][batch_indices], pred[AtomicDataDict.ATOM_NUMBER_KEY][node_centers], alpha)
            
        return pred_key, ref_key, has_nan, not_zeroes


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
        pred_key, ref_key, has_nan, not_zeroes = self.prepare(pred, ref, key, **kwargs)

        if has_nan:
            not_nan_zeroes = (ref_key == ref_key).int() * not_zeroes
            loss = self.func(pred_key.mean.transpose(-1, -2), torch.nan_to_num(ref_key, nan=0.)) * not_nan_zeroes
            if mean:
                return loss.sum() / not_nan_zeroes.sum()
            else:
                loss = self.likelihood(loss).mean
                loss[~not_nan_zeroes.bool()] = torch.nan
                return loss
        else:
            loss = self.func(pred_key.mean.transpose(-1, -2), ref_key) * not_zeroes
            if mean:
                return loss.mean(dim=-1).sum() / not_zeroes.sum()
            else:
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
        pred_key, ref_key, has_nan, not_zeroes = self.prepare(pred, ref, key, **kwargs)
        not_zeroes = not_zeroes.transpose(0, 1)

        loss = self.func(pred_key.mean, ref_key.transpose(0, 1)) * not_zeroes
        loss = loss.mean(dim=-2)
        if mean:
            return loss.sum() / not_zeroes.sum()
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
        assert AtomicDataDict.NOISE in ref, "Noise is missing from dataset. Please add 'noise: [float]' to your config file to use this loss."
        ref_key = -1 * ref.get(AtomicDataDict.NOISE)
        assert isinstance(ref_key, torch.Tensor)
        pred_key = pred.get(key, None)
        assert isinstance(pred_key, torch.Tensor)
        pred_key = pred_key.view_as(ref_key)

        has_nan = torch.isnan(ref_key.sum()) or torch.isnan(pred_key.sum())
        if has_nan and not (hasattr(self, "ignore_nan") and self.ignore_nan):
            raise Exception(f"Either the predicted or true property '{key}' has nan values. "
                             "If this is intended, set 'ignore_nan' to true in config file for this loss.")

        if hasattr(self, "ignore_zeroes") and self.ignore_zeroes:
            not_zeroes = (~torch.all(ref_key == 0., dim=-1)).int() if len(ref_key.shape) > 1 else (ref_key != 0)
        else:
            not_zeroes = torch.ones(*ref_key.shape[:max(1, len(ref_key.shape)-1)], device=ref_key.device).int()
        not_zeroes = not_zeroes.reshape(*([-1] + [1] * (len(pred_key.shape)-1)))
        return pred_key, ref_key, has_nan, not_zeroes