from typing import Dict
import torch.nn

from geqtrain.train._loss import SimpleLoss
from geqtrain.data import AtomicDataDict


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