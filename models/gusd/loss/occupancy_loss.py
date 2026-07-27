import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from mmdet.models.builder import LOSSES
from .utils.lovasz_softmax import lovasz_softmax


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


# ============================================================================
# 辅助函数 (从 GaussianFormer 复制)
# ============================================================================

def CE_wo_softmax(pred, target, class_weights=None, ignore_index=255):
    """Cross-entropy on pre-softmax probabilities.
    pred: (N, C) probability distribution (sums to 1)
    """
    pred = torch.clamp(pred, 1e-6, 1. - 1e-6)
    # breakpoint()
    loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index)
    return loss


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """Standard CrossEntropyLoss with autocast disabled."""
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    with autocast(False):
        loss = criterion(pred, target.long())
    return loss


# ============================================================================
# OccupancyLoss — 与 GaussianFormer 一致，适配 mmcv 1.6 LOSSES registry
# ============================================================================

@LOSSES.register_module()
class OccupancyLoss(nn.Module):
    """用于 occupancy 语义分割的 loss，与 GaussianFormer 实现一致。

    组合 CrossEntropy + Lovasz softmax loss，支持类平衡权重。

    Args:
        weight: loss 总权重 (默认 1.0)
        empty_label: 空类标签 (默认 17)
        num_classes: 类别数 (含 empty, 默认 18)
        balance_cls_weight: 是否使用频率均衡权重
        manual_class_weight: 手动指定类权重列表
        multi_loss_weights: dict, 各子 loss 权重
            - loss_voxel_ce_weight (默认 1.0)
            - loss_voxel_lovasz_weight (默认 1.0)
            - loss_voxel_sem_scal_weight (默认 1.0)
            - loss_voxel_geo_scal_weight (默认 1.0)
        use_sem_geo_scal_loss: 是否使用 sem_scal + geo_scal loss
        use_lovasz_loss: 是否使用 Lovasz softmax loss
        lovasz_ignore: Lovasz 忽略标签
        lovasz_use_softmax: 输入是否已经过 softmax (aggregator 输出为 True)
        ignore_empty: 是否在 CE 中忽略 empty 类
    """

    def __init__(
        self,
        weight=1.0,
        empty_label=17,
        num_classes=18,
        use_focal_loss=False,
        use_dice_loss=False,
        balance_cls_weight=False,
        multi_loss_weights=dict(),
        use_sem_geo_scal_loss=True,
        use_lovasz_loss=True,
        lovasz_ignore=255,
        manual_class_weight=None,
        ignore_empty=False,
        lovasz_use_softmax=True,
    ):
        super().__init__()

        self.weight = weight
        self.empty_label = empty_label
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.use_sem_geo_scal_loss = use_sem_geo_scal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.lovasz_ignore = lovasz_ignore
        self.ignore_empty = ignore_empty
        self.lovasz_use_softmax = lovasz_use_softmax

        self.loss_voxel_ce_weight = multi_loss_weights.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = multi_loss_weights.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = multi_loss_weights.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = multi_loss_weights.get('loss_voxel_lovasz_weight', 1.0)

        if balance_cls_weight:
            if manual_class_weight is not None:
                self.class_weights = torch.tensor(manual_class_weight)
            else:
                class_freqs = nusc_class_frequencies
                self.class_weights = torch.from_numpy(
                    1 / np.log(class_freqs[:num_classes] + 0.001))
            self.class_weights = num_classes * F.normalize(self.class_weights, 1, -1)
        else:
            self.class_weights = torch.ones(num_classes)

        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss

    # ========================================================================
    # Forward: 兼容两种调用方式
    #   - kwargs 方式: OccupancyLoss()(**inputs)
    #   - inputs dict: OccupancyLoss()(inputs)
    # ========================================================================

    def forward(self, pred_occ=None, sampled_xyz=None, sampled_label=None,
                occ_mask=None, **kwargs):
        """计算 loss。

        Args:
            pred_occ:   list of (1, C, N) — 语义概率分布
            sampled_xyz: [可选] (1, N, 3) — 采样点坐标
            sampled_label: (1, N) — 采样点 GT 标签
            occ_mask:   [可选] (1, N) — 额外 mask
        """
        # 兼容 GF 式的 inputs dict 调用
        if pred_occ is None and len(kwargs) == 1 and isinstance(
                list(kwargs.values())[0], dict):
            inputs = list(kwargs.values())[0]
            pred_occ = inputs.get('pred_occ')
            sampled_xyz = inputs.get('sampled_xyz')
            sampled_label = inputs.get('sampled_label')
            occ_mask = inputs.get('occ_mask')

        return self.weight * self.loss_voxel(
            pred_occ, sampled_xyz, sampled_label, occ_mask)

    # ========================================================================
    # 核心 loss 计算
    # ========================================================================

    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None):
        tot_loss = 0.

        if self.ignore_empty:
            empty_mask = sampled_label != self.empty_label
            occ_mask = empty_mask if occ_mask is None else \
                empty_mask & occ_mask.flatten(1)

        if occ_mask is not None:
            occ_mask = occ_mask.flatten(1)
            sampled_label = sampled_label[occ_mask][None]

        for semantics in pred_occ:
            if occ_mask is not None:
                semantics = semantics.transpose(1, 2)[occ_mask][None].transpose(1, 2)

            loss_dict = {}

            # --- CE loss ---
            if self.lovasz_use_softmax:
                loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * \
                    CE_ssc_loss(
                        semantics, sampled_label,
                        self.class_weights.type_as(semantics), ignore_index=255)
            else:
                loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * \
                    CE_wo_softmax(
                        semantics, sampled_label,
                        self.class_weights.type_as(semantics), ignore_index=255)

            # --- Lovasz softmax loss ---
            if self.use_lovasz_loss:
                if self.lovasz_use_softmax:
                    lovasz_input = torch.softmax(semantics, dim=1)
                else:
                    lovasz_input = semantics
                loss_dict['loss_voxel_lovasz'] = \
                    self.loss_voxel_lovasz_weight * lovasz_softmax(
                        lovasz_input.transpose(1, 2).flatten(0, 1),
                        sampled_label.flatten(),
                        ignore=self.lovasz_ignore)

            loss = 0.
            for v in loss_dict.values():
                loss = loss + v
            tot_loss = tot_loss + loss

        return tot_loss / max(len(pred_occ), 1)
