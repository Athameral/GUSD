import torch.nn as nn
from mmdet.models.builder import LOSSES, build_loss


@LOSSES.register_module()
class MultiLoss(nn.Module):
    """组合多个子 loss，与 GaussianFormer 的 MultiLoss 一致。

    Args:
        loss_cfgs (list[dict]): 子 loss 配置列表。每个 dict 包含 type 和其他参数。
    """

    def __init__(self, loss_cfgs):
        super().__init__()
        assert isinstance(loss_cfgs, list), 'loss_cfgs must be a list of dict'
        self.num_losses = len(loss_cfgs)

        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(build_loss(loss_cfg))
        self.losses = nn.ModuleList(losses)

    def forward(self, **inputs):
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(**inputs)
            tot_loss += loss
        return tot_loss
