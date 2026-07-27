import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from ..bbox.utils import decode_points, encode_points

from .localagg_prob_fast import local_aggregate_prob_fast

from .loss import MultiLoss, OccupancyLoss  # noqa: F401

from ..utils import _unravel_index


def get_rotation_matrix(tensor):
    """四元数 → 旋转矩阵 3x3 (来自 GaussianFormer model/utils/utils.py)
    
    Args:
        tensor: (..., 4) — 四元数 quaternion (w, x, y, z)
    Returns:
        R_3x3: (..., 3, 3)
    """
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]

    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]

    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    R_4x4 = torch.matmul(mat1, mat2)  # (..., 4, 4)
    R_3x3 = R_4x4[..., :3, :3]  # (..., 3, 3)
    return R_3x3


@HEADS.register_module()
class GUSDHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 # Gaussian Splatting 参数
                 scale_range=[0.01, 5.4],
                 scale_max_safe=0.4,
                 warm_up_samples=0,
                 cuda_kwargs=dict(),
                 # Loss 配置
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 loss_gs_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 init_cfg=None,
                 gs_seed_scale=True,
                 output_all_gs=False,
                 # GS 正则化参数
                 scale_reg_weight=0.001,     # ℓ2-norm scale 方差正则化权重
                 scale_reg_tau=0,          # 允许的自然方差阈值 (世界坐标系, 米)
                 entropy_reg_weight=0.01,    # 最小熵正则化权重
                 entropy_min_ratio=0.5,      # 最小熵 = ratio * log(R)
                 # GS 渲染剪枝: 过滤 scale 和 opacity 都极小的无效高斯
                 pruning=False,               # 是否启用剪枝
                 prune_scale_thr=0.02,       # max(s_x,s_y,s_z) 低于此值的高斯可能被剪
                 prune_opa_thr=0.01,         # opacity 低于此值的高斯可能被剪
                 # loss_bin FN (假空) 加权
                 # 默认 None = 不启用 (保持原 reduction='mean' 行为)
                 # 设为正数 (如 20) 后, GT=occupied 但 pred=empty 的体素 BCE 权重 × FN_loss_mult
                 FN_loss_mult=None,
                 FN_loss_thresh=0.5,          # pred < thresh 视为预测为空
                 # 类别感知 scale floor (仅推理生效, 训练时不变)
                 # use_replace=True 时, 在 prepare_gaussian_args 构建 Cov 前
                 # 对特定类别高斯施加 scale 下限: scales = where(mask, max(s, floor), s)
                 # floor 作用于 Gaussian 局部坐标系对角元 (S=diag(scales))
                 # 被命中的高斯同时取消旋转 → 局部系=世界系, floor 的 xyz 才名副其实
                 # class_scale_floor: {class_idx: (sx, sy, sz)}, None=该轴不约束
                 # replace_opa_alpha: opa 衰减指数. 0=不衰减; 1.0=按体积膨胀倍数 r 线性降 opa
                 #   使 √det·opa 守恒 → 语义投票权恢复原值, 不抢邻居类(如 barrier)
                 #   bin_logit 只用 power 不含 opa, occupancy 填空洞效果保留
                 use_replace=None,
                 class_scale_floor=None,
                 replace_opa_alpha=0.0,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.output_all_gs = output_all_gs
        self.scale_reg_weight = scale_reg_weight
        self.scale_reg_tau = scale_reg_tau
        self.entropy_reg_weight = entropy_reg_weight
        self.entropy_min_ratio = entropy_min_ratio
        self.pruning = pruning
        self.prune_scale_thr = prune_scale_thr
        self.prune_opa_thr = prune_opa_thr
        self.FN_loss_mult = FN_loss_mult
        self.FN_loss_thresh = FN_loss_thresh
        self.use_replace = use_replace
        # 构建类别感知 scale floor buffer: (num_classes, 3), None=0 表示不约束
        # 仅推理时生效; __init__ 一次性构建, 随 .to(device) 自动迁移
        if use_replace and class_scale_floor:
            floor_tensor = torch.zeros(num_classes, 3, dtype=torch.float32)
            for cls_idx, floors in class_scale_floor.items():
                for ax in range(3):
                    if floors[ax] is not None:
                        floor_tensor[cls_idx, ax] = float(floors[ax])
            self.register_buffer('class_scale_floor', floor_tensor)
        else:
            self.use_replace = False
            self.class_scale_floor = None
        self.replace_opa_alpha = float(replace_opa_alpha)
        self.debug = train_cfg.get('debug', False)
        # self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.loss_gs_cls = build_loss(loss_gs_cls)
        if output_all_gs:
            transformer['output_all_gs'] = True
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        # Gaussian 属性维度
        self.ga_scale_dim = 3       # scale x, y, z
        self.ga_rot_dim = 4         # rotation quaternion w, x, y, z
        self.ga_opa_dim = 1         # opacity
        self.ga_geom_dim = self.ga_scale_dim + self.ga_rot_dim + self.ga_opa_dim  # = 8

        # Gaussian Aggregator
        self.scale_range = scale_range
        self.scale_max_safe = scale_max_safe
        self.warm_up_samples = warm_up_samples
        self.scale_max_current = scale_range[1] if warm_up_samples <= 0 else scale_max_safe

        # GaussianSeed 
        if gs_seed_scale:
            self.gs_seed_scales = [self.scale_range[1] / (r ** (1 / 3)) for r in self.transformer.num_refines]
        else:
            self.gs_seed_scales = [self.scale_range[1]] * len(self.transformer.num_refines)

        self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    # ========================================================================
    # Gaussian 参数准备 & 渲染
    # ========================================================================
    @force_fp32(apply_to=("gs_params",))
    def prepare_gaussian_args(self, refine_pts, gs_params, semantics_logits):
        """将 Transformer 输出转换为 aggregator 需要的 Gaussian 参数。

        Args:
            refine_pts:       (B, Q, R, 3) — 归一化坐标 (0~1)
            gs_params:        (B, Q, R, 8) — scale(3)+rot(4)+opa(1)
            semantics_logits: (B, Q, R, C) — 原始语义 logits (C=num_classes)

        Returns:
            means:    (B, G, 3)   — 世界坐标
            origi_opa: (B, G)     — 几何不透明度
            opacities: (B, G, C)  — 语义概率
            scales:   (B, G, 3)   — 各轴尺度
            Cov:      (B, G, 3, 3) — 协方差矩阵 (stable)
            CovInv:   (B, G, 3, 3) — 协方差逆矩阵
        """
        B, Q, R, _ = refine_pts.shape
        G = Q * R  # 展平 Q 和 R 维度

        # 1. 位置: decode 归一化坐标 → 世界坐标
        # clamp 到 [eps, 1-eps] 避免 decode 后恰好在 pc_range 边界，
        # 否则 aggregator 整数量化时 grid_index == grid_dim，触发 < H/W/D 断言失败
        eps = 1e-5
        refine_pts = refine_pts.clamp(min=eps, max=1 - eps)
        refine_pts = refine_pts.reshape(B, G, 3)
        means = decode_points(refine_pts, self.pc_range)  # (B, G, 3)

        # 2. Scale: sigmoid → 映射到 [scale_min, scale_max]
        gs_params = gs_params.reshape(B, G, self.ga_geom_dim)
        raw_scale = gs_params[..., 0:3]
        scales = torch.sigmoid(raw_scale)
        scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scales

        # 3. Rotation: L2 normalize 四元数
        rotations = F.normalize(gs_params[..., 3:7], dim=-1)  # (B, G, 4)

        # 4. Opacity: sigmoid
        origi_opa = torch.sigmoid(gs_params[..., 7:8]).squeeze(-1)  # (B, G)

        # 5. Semantics: softmax
        semantics_logits = semantics_logits.reshape(B, G, self.num_classes)
        opacities = F.softmax(semantics_logits, dim=-1)  # (B, G, C)
        # opacities = semantics_logits

        # 5.5 类别感知 scale floor (仅推理生效)
        #     floor 作用于 S=diag(scales) 对角元, 即 Gaussian 局部坐标系轴
        #     为让 floor 真正对应世界轴, 对被 floor 拉伸的 Gaussian
        #     同时取消其旋转 (quaternion → (1,0,0,0)), 局部系=世界系
        #     后续 Cov/CovInv 自动用 floored scales + identity R 构建, 天然一致
        if (not self.training) and self.use_replace and self.class_scale_floor is not None:
            gs_classes = opacities.argmax(dim=-1)        # (B, G)
            floor = self.class_scale_floor[gs_classes]   # (B, G, 3), 0=该轴不约束
            needs_floor = (floor > 0) & (scales < floor)  # (B, G, 3)
            needs_reset = needs_floor.any(dim=-1)         # (B, G) 任一轴命中
            # 1) scales: 命中的轴提至 floor, 未命中的轴保持原值
            scales_floored = torch.where(needs_floor, floor, scales)
            # 2) opa 衰减: 按体积膨胀倍数 r 降 opa, 让 √det·opa 守恒
            #    (bin_logit 只用 power, 不受影响 → occupancy 填空洞保留;
            #     out_logits 用 √det·opa·power → 语义投票权复原, 不抢邻居类)
            #    opa_new = opa_old / r^alpha, alpha=1.0 ⇒ det·opa 守恒
            if self.replace_opa_alpha > 0 and needs_reset.any():
                V_old = scales.prod(dim=-1).clamp(min=1e-6)                # (B, G)
                V_new = scales_floored.prod(dim=-1).clamp(min=1e-6)        # (B, G)
                r = (V_new / V_old).clamp(min=1.0)                         # >=1 仅膨胀
                opa_scale = 1.0 / (r ** self.replace_opa_alpha)
                origi_opa = torch.where(needs_reset,
                                        origi_opa * opa_scale,
                                        origi_opa)
            # 3) 提交 floored scales + 取消旋转
            scales = scales_floored
            if needs_reset.any():
                id_quat = scales.new_tensor([1.0, 0.0, 0.0, 0.0])  # (4,)
                rotations = torch.where(
                    needs_reset[..., None], id_quat, rotations)

        # 6. 构建协方差逆矩阵 CovInv
        S = torch.zeros(B, G, 3, 3, dtype=scales.dtype, device=scales.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations)  # (B, G, 3, 3)
        M = torch.matmul(S, R)              # (B, G, 3, 3)
        Cov = torch.matmul(M.transpose(-1, -2), M)  # (B, G, 3, 3)
        # 按 scale 比例加 jitter，避免大 scale 下协方差矩阵接近奇异 → pinv 出 NaN
        # jitter = max(1e-3, scale_max * 0.01)，即 jitter 随 scale 自适应增长
        diag_jitter = torch.clamp(scales.detach().amax(dim=-1, keepdim=True) * 0.01, min=1e-3)  # (B, G, 1)
        diag_jitter = diag_jitter.unsqueeze(-1)  # (B, G, 1, 1)
        I = torch.eye(3, device=Cov.device).view(1, 1, 3, 3)
        Cov_stable = Cov + diag_jitter * I
        # L = torch.linalg.cholesky(Cov_stable)
        # CovInv = torch.cholesky_inverse(L)  # (B, G, 3, 3)
        CovInv = torch.linalg.pinv(Cov_stable, rcond=1e-5)  # (B, G, 3, 3)

        return means, origi_opa, opacities, scales, Cov_stable, CovInv

    def generate_voxel_centers(self, batch_size, device):
        """生成所有体素中心的坐标 (世界坐标系).

        Returns:
            centers: (B, X*Y*Z, 3)
        """
        X, Y, Z = self.voxel_num.tolist()  # 如 [200, 200, 16]

        x = (torch.arange(X, device=device, dtype=torch.float32) + 0.5) / X * self.scene_size[0] + self.pc_range[0]
        y = (torch.arange(Y, device=device, dtype=torch.float32) + 0.5) / Y * self.scene_size[1] + self.pc_range[1]
        z = (torch.arange(Z, device=device, dtype=torch.float32) + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        centers = torch.stack([xx, yy, zz], dim=-1)  # (X, Y, Z, 3)
        centers = centers.reshape(-1, 3)              # (X*Y*Z, 3)
        centers = centers.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 3)
        return centers  # shape: (B, X*Y*Z, 3)

    # ========================================================================
    # Forward
    # ========================================================================

    def forward(self, mlvl_feats, img_metas):
        B, Q = mlvl_feats[0].shape[0], self.num_query

        # 初始化 query points
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        # Transformer forward
        pt_feats, refine_pts = self.transformer(
            init_points, query_feat, mlvl_feats, img_metas=img_metas)
        # pt_feats[i]: (gs_params, semantics) tuple 或 None
        # refine_pts[i]: (B, Q, R_i, 3)

        # 生成体素中心 (所有 batch 共享)
        voxel_centers = self.generate_voxel_centers(1, init_points.device)  # (1, X*Y*Z, 3)

        # 对每层进行 Gaussian 渲染
        all_gs_rendered = []
        all_gs_bin_logits = []
        all_gs_density = []
        all_gs = []
        all_gs_scales = []   # (B, Q, R, 3) decoded scales per layer
        all_gs_opas = []     # (B, Q, R)    decoded opacity per layer

        num_layers = len(refine_pts)
        for i in range(num_layers):
            if pt_feats[i] is None:
                all_gs_rendered.append(None)
                all_gs_bin_logits.append(None)
                all_gs_density.append(None)
                all_gs.append(None)
                all_gs_scales.append(None)
                all_gs_opas.append(None)
                continue

            gs_params, semantics = pt_feats[i]  # 解包

            _, Q_i, R_i, _ = refine_pts[i].shape
            G = Q_i * R_i
            gs_dim = 3 + self.ga_opa_dim + 1 + 9
            gaussians = refine_pts[i].new_zeros(B, G, gs_dim)

            # 逐 batch 调用 aggregator (aggregator 要求 batch=1)
            batch_rendered_sem = []
            batch_bin_logits = []
            batch_density = []
            batch_decoded_scales = []  # 收集解码后的 scales
            batch_decoded_opa = []     # 收集解码后的 opacity
            # 可以试试multi_apply等方法并行化
            # 但是算子只支持单batch
            for b in range(B):
                means, origi_opa, opacities, scales, Cov, CovInv = self.prepare_gaussian_args(
                    refine_pts[i][b:b+1], gs_params[b:b+1], semantics[b:b+1])

                batch_decoded_scales.append(scales)        # (1, G, 3)
                batch_decoded_opa.append(origi_opa)         # (1, G)

                gaussians[b] = torch.cat(
                    [
                        means,
                        origi_opa.unsqueeze(-1),
                        opacities.argmax(dim=-1, keepdim=True).float(),
                        Cov.reshape(1, G, 9),
                    ],
                    dim=-1,
                ).squeeze(0).detach()

                # 剪枝: 同时满足 scale_max < thr 且 opa < thr 的高斯跳过渲染
                # 保留所有高斯用于 loss 梯度, 仅对 aggregator 输入做过滤
                if self.pruning and self.training:
                    scale_max = scales.amax(dim=-1)                        # (1, G)
                    keep = (scale_max >= self.prune_scale_thr) | (origi_opa >= self.prune_opa_thr)
                    if not keep.all():
                        idx = keep.nonzero(as_tuple=True)[1]               # (K,)
                        means = means[:, idx, :]
                        origi_opa = origi_opa[:, idx]
                        opacities = opacities[:, idx, :]
                        scales = scales[:, idx, :]
                        CovInv = CovInv[:, idx, :, :]

                rendered = self.aggregator(
                    voxel_centers.clone().float(),
                    means,
                    origi_opa,
                    opacities,
                    scales,
                    CovInv
                )  # → (semantics[1,N,C], bin_logits[1,N], density[1,N])
                batch_rendered_sem.append(rendered[0])
                batch_bin_logits.append(rendered[1])
                batch_density.append(rendered[2])

            rendered_sem = torch.cat(batch_rendered_sem, dim=0)    # (B, N, C)
            bin_logits = torch.cat(batch_bin_logits, dim=0)        # (B, N)
            density = torch.cat(batch_density, dim=0)              # (B, N)

            # reshape 为 dense occ 格式
            X, Y, Z = self.voxel_num.tolist()
            rendered_sem = rendered_sem.reshape(B, X, Y, Z, self.num_classes).permute(0, 4, 1, 2, 3)  # (B, C, X, Y, Z)
            bin_logits = bin_logits.reshape(B, 1, X, Y, Z)  # (B, 1, X, Y, Z)
            density = density.reshape(B, 1, X, Y, Z)

            all_gs_rendered.append(rendered_sem)
            all_gs_bin_logits.append(bin_logits)
            all_gs_density.append(density)
            all_gs.append(gaussians)

            # 保存解码后的 scales & opacity (B, Q, R, ...) 供正则化复用
            decoded_scales = torch.cat(batch_decoded_scales, dim=0)   # (B, G, 3)
            decoded_opa = torch.cat(batch_decoded_opa, dim=0)          # (B, G)
            all_gs_scales.append(decoded_scales.reshape(B, Q_i, R_i, 3))
            all_gs_opas.append(decoded_opa.reshape(B, Q_i, R_i))

        return dict(
            init_points=init_points,
            all_refine_pts=refine_pts,
            all_gs_rendered=all_gs_rendered,
            all_gs_bin_logits=all_gs_bin_logits,
            all_gs_density=all_gs_density,
            all_gs=all_gs,
            all_gs_scales=all_gs_scales,  # list of (B, Q, R, 3) or None — decoded
            all_gs_opas=all_gs_opas,      # list of (B, Q, R)    or None — decoded
        )

    # ========================================================================
    # 点回归相关 (loss_pts, Chamfer distance) — 保持不变
    # ========================================================================

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    @torch.no_grad()
    def _get_regression_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)

        return gt_paired_idx, pred_paired_idx, gt_pts_weights

    def get_targets(self):
        # To instantiate the abstract method
        pass

    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask])  # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])

        return gt_points, gt_masks, gt_labels

    # ========================================================================
    # Loss 计算
    # ========================================================================

    def loss_pts_single(self, refine_pts, gt_points_list, gt_masks_list, gt_labels_list):
        """仅计算 Chamfer distance loss。"""
        num_imgs = refine_pts.size(0)  # B
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        # calculate loss pts
        gt_paired_idx_list, pred_paired_idx_list, gt_pts_weights = multi_apply(
            self._get_regression_target_single, refine_pts_list, gt_points_list,
            gt_masks_list, gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_pts

    def gs_rendered_loss_single(self, rendered_occ, bin_logits, gt_dense_occ):
        """对单层 Gaussian 渲染结果计算稀疏 loss。

        rendered_occ 是聚合后的概率分布 (sum=1), bin_logits 是占用概率 [0,1]。
        只对 GT 非空 或 Pred 非空 的体素计算 loss。

        Args:
            rendered_occ: (B, C, X, Y, Z) — aggregator 渲染的语义概率分布 (sums to 1)
            bin_logits:   (B, 1, X, Y, Z) — occupancy 概率 [0, 1]
            gt_dense_occ: (B, X, Y, Z)     — GT (empty_label 表示空)

        Returns:
            loss_gs: 标量
        """
        if rendered_occ is None:
            return torch.tensor(0., device=bin_logits.device if bin_logits is not None else 'cpu')

        # 1. 确定 GT 非空体素 和 Pred 非空体素
        gt_nonempty = (gt_dense_occ != self.empty_label)                      # (B, X, Y, Z)
        pred_occupied = (bin_logits.squeeze(1) > 0.5)                        # (B, X, Y, Z)

        # 2. 取并集（同时惩罚 false positive 和 false negative）
        valid_mask = gt_nonempty | pred_occupied  # (B, X, Y, Z)

        if valid_mask.sum() == 0:
            return rendered_occ.new_tensor(0.)

        # 3. 稀疏提取: 用 nonzero() 获取有效索引
        valid_indices = valid_mask.nonzero(as_tuple=False)  # (K, 4), [b, x, y, z]

        # 4. 在稀疏索引上提取预测值和 GT
        b_idx = valid_indices[:, 0]
        x_idx = valid_indices[:, 1]
        y_idx = valid_indices[:, 2]
        z_idx = valid_indices[:, 3]

        pred_at_valid = rendered_occ[b_idx, :, x_idx, y_idx, z_idx]   # (K, C) — 概率分布
        gt_at_valid = gt_dense_occ[b_idx, x_idx, y_idx, z_idx]         # (K,)

        # 5. 将概率分布转为 logits (inverse sigmoid), 适配 FocalLoss(use_sigmoid=True)
        eps = 1e-7
        pred_at_valid = pred_at_valid.clamp(min=eps, max=1 - eps)
        pred_logits = torch.log(pred_at_valid / (1 - pred_at_valid))   # (K, C) — log-odds
        # pred_logits = pred_at_valid  # 直接用概率分布，FocalLoss 内部会处理

        # 6. 计算语义 loss
        cls_weights = self.train_cfg.get('cls_weights', [1] * rendered_occ.shape[1])
        cls_weights = rendered_occ.new_tensor(cls_weights)
        cls_weights = cls_weights[None, :].expand(pred_logits.shape[0], -1)
        # 注意这里需要让cls_weights连续
        cls_weights = cls_weights.contiguous()
        avg_factor = max(gt_nonempty.sum(), 1)

        loss_gs = self.loss_gs_cls(
            pred_logits,
            gt_at_valid,
            weight=cls_weights,
            avg_factor=avg_factor)

        return loss_gs
    
    def _select_topk_occupied(self, bin_logits: torch.Tensor):
        B, _, X, Y, Z = bin_logits.shape
        bin_logits_topk, occupied_idx = torch.topk(
            bin_logits.flatten(start_dim=1),
            k=self.test_cfg.get('score_topk', 76800)
        )
        # (B, K), (B, K)
        kth_thresh = bin_logits_topk[:, -1:]  # (B, 1)
        occupied_idx = _unravel_index(occupied_idx, (X, Y, Z))  # (B, K, 3)
        return kth_thresh, occupied_idx

    # ========================================================================
    # 构建 GS loss 输入 (适配 OccupancyLoss / MultiLoss 接口)
    # ========================================================================

    def _build_gs_loss_inputs(self, rendered_occ, gt_dense_occ, mask_camera):
        """将 dense 格式的渲染结果转为 OccupancyLoss 需要的稀疏格式。

        Args:
            rendered_occ: (B, C, X, Y, Z) — 概率分布 (含 empty 通道)
            gt_dense_occ: (B, X, Y, Z)     — GT 标签 (empty_label=17)
            mask_camera:  (B, X, Y, Z)     — camera mask

        Returns:
            dict: {pred_occ, sampled_xyz, sampled_label, occ_mask}
        """
        gt_nonempty = (gt_dense_occ != self.empty_label)          # (B, X, Y, Z)
        valid_mask = gt_nonempty & mask_camera.bool()             # camera 内 & 非空

        B, C, X, Y, Z = rendered_occ.shape

        pred_occ_list = []
        sampled_xyz_list = []
        sampled_label_list = []
        occ_mask_list = []

        for b in range(B):
            vm = valid_mask[b]  # (X, Y, Z)
            if vm.sum() == 0:
                # 用 dummy 避免空 tensor
                vm[X // 2, Y // 2, Z // 2] = True

            indices = vm.nonzero(as_tuple=False)  # (K, 3) — [x, y, z]
            # 限制采样点数 (防止显存爆炸)
            # max_samples = self.train_cfg.get('max_gs_loss_samples', 50000)
            # if indices.shape[0] > max_samples:
            #     perm = torch.randperm(indices.shape[0], device=indices.device)
            #     indices = indices[perm[:max_samples]]

            x_idx, y_idx, z_idx = indices[:, 0], indices[:, 1], indices[:, 2]

            # pred_occ: list of (1, C, K) — GF OccupancyLoss 期望多个 layer
            sem = rendered_occ[b, :, x_idx, y_idx, z_idx]  # (C, K)
            pred_occ_list.append(sem.unsqueeze(0))          # (1, C, K)

            # sampled_xyz: (1, K, 3) — 采样点世界坐标
            pts = self._idx_to_world_coords(x_idx, y_idx, z_idx)  # (K, 3)
            sampled_xyz_list.append(pts.unsqueeze(0))               # (1, K, 3)

            # sampled_label: (1, K) — GT 标签
            sampled_label_list.append(
                gt_dense_occ[b, x_idx, y_idx, z_idx].unsqueeze(0))  # (1, K)

            # occ_mask: None (不使用)
            occ_mask_list.append(None)

        # GF OccupancyLoss 期望 pred_occ 是 list of (1, C, N) — 多层级
        pred_occ = [torch.cat([p[i:i+1] for p in pred_occ_list], dim=-1)
                    for i in range(len(pred_occ_list[0]))] if pred_occ_list else []
        # 实际上每层独立调用，所以 pred_occ 只有一个 element
        sampled_xyz = torch.cat(sampled_xyz_list, dim=1) if sampled_xyz_list else None
        sampled_label = torch.cat(sampled_label_list, dim=1) if sampled_label_list else None

        return dict(
            pred_occ=pred_occ,
            sampled_xyz=sampled_xyz,
            sampled_label=sampled_label,
            occ_mask=None,
        )

    def _idx_to_world_coords(self, x_idx, y_idx, z_idx):
        """将体素索引转为世界坐标（体素中心）."""
        X, Y, Z = self.voxel_num.tolist()
        x = (x_idx.float() + 0.5) / X * self.scene_size[0] + self.pc_range[0]
        y = (y_idx.float() + 0.5) / Y * self.scene_size[1] + self.pc_range[1]
        z = (z_idx.float() + 0.5) / Z * self.scene_size[2] + self.pc_range[2]
        return torch.stack([x, y, z], dim=-1)

    # ========================================================================
    # Loss 方法
    # ========================================================================

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        """计算 loss。

        Args:
            voxel_semantics: [B, X, Y, Z] GT dense occ (17=empty)
            mask_camera:     [B, X, Y, Z] camera mask
            preds_dicts:     forward() 的返回值
        """
        init_points = preds_dicts['init_points']
        all_refine_pts = preds_dicts['all_refine_pts']
        all_gs_rendered = preds_dicts['all_gs_rendered']
        all_gs_bin_logits = preds_dicts['all_gs_bin_logits']

        voxel_semantics = voxel_semantics.long()
        mask_camera = mask_camera.bool()

        num_dec_layers = len(all_refine_pts)

        # === A. 准备 GT sparse points (用于 loss_pts) ===
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        # === B. 计算每层的 loss_pts (Chamfer distance) ===
        losses_pts = []
        for i in range(num_dec_layers):
            loss_pts_i = self.loss_pts_single(
                all_refine_pts[i],
                all_gt_points_list[i],
                all_gt_masks_list[i],
                all_gt_labels_list[i])
            losses_pts.append(loss_pts_i)

        # === C. 计算每层的 gs_rendered_loss ===
        losses_gs = []
        for i in range(num_dec_layers):
            if all_gs_rendered[i] is None:
                losses_gs.append(torch.tensor(0., device=voxel_semantics.device))
            else:
                inputs = self._build_gs_loss_inputs(
                    all_gs_rendered[i], voxel_semantics, mask_camera)
                losses_gs.append(self.loss_gs_cls(**inputs))

        # === D. 计算BCE loss（二分类）===
        losses_bin = []
        gt_nonempty: torch.Tensor = (voxel_semantics != self.empty_label).float()
        for i in range(num_dec_layers):
            if all_gs_bin_logits[i] is None:
                losses_bin.append(torch.tensor(0., device=voxel_semantics.device))
            else:
                safe_input = all_gs_bin_logits[i].squeeze(1).clamp(min=1e-7, max=1 - 1e-7)
                if self.FN_loss_mult is not None:
                    # 加权: GT=occupied 但 pred < thresh (假空) 的体素 BCE 权重 × FN_loss_mult
                    pred_empty = (safe_input.detach() < self.FN_loss_thresh)
                    fn_mask = (gt_nonempty > 0.5) & pred_empty            # (B, X, Y, Z)
                    w = torch.ones_like(gt_nonempty)
                    w[fn_mask] = self.FN_loss_mult
                    losses_bin.append(F.binary_cross_entropy(safe_input,
                                                             gt_nonempty,
                                                             weight=w,
                                                             reduction='mean'))
                else:
                    losses_bin.append(F.binary_cross_entropy(safe_input,
                                                             gt_nonempty,
                                                             reduction='mean'))
                

        # === E. 组装 loss_dict ===
        loss_dict = dict()

        # init_points loss
        if init_points is not None:
            init_loss_pts = self.loss_pts_single(
                init_points, gt_points_list, gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # 最后一层（主监督）
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_gs'] = losses_gs[-1]
        loss_dict['loss_bin'] = losses_bin[-1]

        # 中间层（弱监督）
        for i in range(num_dec_layers - 1):
            loss_dict[f'd{i}.loss_pts'] = losses_pts[i]
            loss_dict[f'd{i}.loss_gs'] = losses_gs[i]
            loss_dict[f'd{i}.loss_bin'] = losses_bin[i]

        # === F. GS 正则化: scale 一致性 + 反坍塌 ===
        all_gs_scales = preds_dicts.get('all_gs_scales', None)
        all_gs_opas = preds_dicts.get('all_gs_opas', None)
        if all_gs_scales is not None and all_gs_opas is not None:
            loss_scale_reg, loss_entropy_reg = self._compute_gs_regularization(all_gs_scales, all_gs_opas)
            loss_dict['loss_scale_reg'] = loss_scale_reg
            loss_dict['loss_entropy_reg'] = loss_entropy_reg
            # # DEBUG: 验证正则化是否生效 (跑一轮后删除)
            # import torch.distributed as dist
            # if dist.is_initialized() and dist.get_rank() == 0:
            #     n_valid = sum(1 for s in all_gs_scales if s is not None)
            #     R_vals = [s.shape[2] for s in all_gs_scales if s is not None]
            #     print(f'[DEBUG reg] tau={self.scale_reg_tau} ratio={self.entropy_min_ratio} '
            #           f'n_valid={n_valid} R={R_vals} '
            #           f'scale_reg={loss_scale_reg.item():.8f} entropy_reg={loss_entropy_reg.item():.8f}',
            #           flush=True)

        return loss_dict

    # ========================================================================
    # GS 正则化 Loss
    # ========================================================================

    def _compute_gs_regularization(self, all_gs_scales, all_gs_opas):
        """计算两个 GS 正则化 loss（使用已解码的参数，无需重复 sigmoid）。

        1. Scale 一致性 (ℓ₂-norm 方差 + 软阈值):
           对每层每个 query 中 R 个高斯的 ℓ₂-norm scale 求方差，
           只惩罚超过 tau 的过散 query。

        2. 最小熵阈值 (反坍塌):
           确保同一 query 内 R 个高斯的 opacity 贡献不会坍塌到单个高斯上。

        Args:
            all_gs_scales: list of (B, Q, R, 3) or None — 已解码的世界坐标 scale
            all_gs_opas:   list of (B, Q, R)    or None — 已解码的 opacity (sigmoid 后)

        Returns:
            loss_scale_reg, loss_entropy_reg: 两个标量正则化 loss
        """
        loss_scale_total = 0.
        loss_entropy_total = 0.
        valid_layers = 0

        for scales, opa in zip(all_gs_scales, all_gs_opas):
            if scales is None or opa is None:
                continue
            # scales: (B, Q, R, 3), opa: (B, Q, R) — 均已解码
            B, Q, R, _ = scales.shape

            # ---- 1. Scale ℓ₂-norm 方差正则化 ----
            if self.scale_reg_weight > 0:
                eff_size = torch.norm(scales, dim=-1)                    # (B, Q, R)
                var_per_query = eff_size.var(dim=2, correction=0)       # (B, Q) — 有偏估计,避免 R=1 时除零
                over_spread = F.relu(var_per_query - self.scale_reg_tau ** 2)
                loss_scale_total += over_spread.mean()

            # ---- 2. 最小熵阈值正则化 ----
            if self.entropy_reg_weight > 0:
                importance = opa / (opa.sum(dim=2, keepdim=True).clamp(min=1e-8) + 1e-8)
                entropy = -(importance * (importance + 1e-8).log()).sum(dim=2)  # (B, Q)
                H_min = self.entropy_min_ratio * math.log(R)
                entropy_deficit = F.relu(H_min - entropy)
                loss_entropy_total += entropy_deficit.mean()

            valid_layers += 1

        if valid_layers == 0:
            return torch.tensor(0.), torch.tensor(0.)

        loss_scale_reg = self.scale_reg_weight * (loss_scale_total / valid_layers)
        loss_entropy_reg = self.entropy_reg_weight * (loss_entropy_total / valid_layers)
        return loss_scale_reg, loss_entropy_reg

    # ========================================================================
    # 推理
    # ========================================================================

    def get_occ(self, pred_dicts, img_metas, rescale=False):
        """从 Gaussian 渲染结果中提取占用预测（输出 dense occ）。

        Args:
            pred_dicts: forward() 的返回值
        Returns:
            result_list: list of dict {dense_occ: np.ndarray (X, Y, Z)}
        """
        rendered_occ: torch.Tensor = pred_dicts['all_gs_rendered'][-1]       # (B, C, X, Y, Z)
        bin_logits: torch.Tensor = pred_dicts['all_gs_bin_logits'][-1]       # (B, 1, X, Y, Z)
        batch_size = rendered_occ.shape[0]
        X, Y, Z = self.voxel_num.tolist()

        # 1. 语义: argmax → (B, X, Y, Z)
        semantics = rendered_occ.argmax(dim=1)

        # 2. TopK 占用过滤 → 构建 dense occ
        kth_thresh, occupied_idx = self._select_topk_occupied(bin_logits)
        dense_occ = torch.full(
            (batch_size, X, Y, Z), self.empty_label,
            dtype=torch.long, device=rendered_occ.device)
        for b in range(batch_size):
            x = occupied_idx[b, :, 0].long()
            y = occupied_idx[b, :, 1].long()
            z = occupied_idx[b, :, 2].long()
            dense_occ[b, x, y, z] = semantics[b, x, y, z]

        if self.test_cfg.get('padding', True):
            occupied_mask = (dense_occ != self.empty_label).float().unsqueeze(1)  # (B, 1, X, Y, Z)
            dilated = F.max_pool3d(occupied_mask, 3, stride=1, padding=1)
            eroded = -F.max_pool3d(-dilated, 3, stride=1, padding=1)
            closed_mask = (eroded > 0.5).squeeze(1)  # (B, X, Y, Z)
            # 保留原始有预测的位置，只填充新出现的空洞
            original_mask = (dense_occ != self.empty_label)
            newly_filled = closed_mask & ~original_mask
            dense_occ[newly_filled] = semantics[newly_filled]

        # 4. 返回 dense occ
        result_list = []
        for i in range(batch_size):
            result_list.append(dict(
                dense_occ=dense_occ[i].detach().cpu().numpy(),
            ))

        # === Debug: 诊断高斯分布 & 坍塌情况 ===
        if self.debug:
            all_refine_pts = pred_dicts['all_refine_pts']
            all_gs_scales = pred_dicts.get('all_gs_scales', None)
            all_gs_opas = pred_dicts.get('all_gs_opas', None)

            # 最后一层 (主监督)
            last_layer = -1
            refine_pts_last = all_refine_pts[last_layer]  # (B, Q, R, 3) normalized
            if refine_pts_last is not None:
                B, Q, R, _ = refine_pts_last.shape
                # 解码到世界坐标
                means = decode_points(
                    refine_pts_last.reshape(B, Q * R, 3), self.pc_range
                ).reshape(B, Q, R, 3)

                # 1. Intra-query 高斯弥散度 (防止坍塌到同一点)
                centroid = means.mean(dim=2, keepdim=True)  # (B, Q, 1, 3)
                dispersion = (means - centroid).norm(dim=-1).mean(dim=-1)  # (B, Q)
                avg_dispersion = dispersion.mean().item()
                max_dispersion = dispersion.max().item()
                min_dispersion = dispersion.min().item()

                # 2. Scale 分布
                scale_stats = ''
                if all_gs_scales is not None and all_gs_scales[last_layer] is not None:
                    scales_last = all_gs_scales[last_layer]  # (B, Q, R, 3)
                    eff_size = scales_last.norm(dim=-1)  # (B, Q, R)
                    scale_stats = (
                        f'scale_L2 avg={eff_size.mean().item():.4f} '
                        f'min={eff_size.min().item():.4f} max={eff_size.max().item():.4f} | '
                    )

                # 3. Opacity 分布
                opa_stats = ''
                if all_gs_opas is not None and all_gs_opas[last_layer] is not None:
                    opas_last = all_gs_opas[last_layer]  # (B, Q, R)
                    opa_stats = (
                        f'opa avg={opas_last.mean().item():.6f} '
                        f'min={opas_last.min().item():.6f} max={opas_last.max().item():.6f} '
                        f'frac<0.01={(opas_last < 0.01).float().mean().item():.2%}'
                    )

                logging.info(
                    f'[DEBUG] B={B} Q={Q} R_last={R} | '
                    f'{scale_stats}'
                    f'{opa_stats} | '
                    f'dispersion avg={avg_dispersion:.4f}m max={max_dispersion:.4f}m min={min_dispersion:.4f}m'
                )

        return result_list

    def get_gaussians(self, pred_dicts, img_metas, rescale=False):
        """从 forward 输出中提取 Gaussians 参数。

        Args:
            pred_dicts: forward() 的返回值
        Returns:
            result_list: list of dict {gaussians_0, gaussians_1, ...}
        """
        all_gs = pred_dicts['all_gs']
        batch_size = pred_dicts['all_refine_pts'][-1].shape[0]

        # 确定要输出的层索引
        if self.output_all_gs:
            layer_indices = [i for i, gs in enumerate(all_gs) if gs is not None]
        else:
            layer_indices = [len(all_gs) - 1] if all_gs[-1] is not None else []

        result_list = [dict() for _ in range(batch_size)]
        for layer_idx in layer_indices:
            gs = all_gs[layer_idx]  # (B, G, gs_dim)
            for b in range(batch_size):
                result_list[b][f'gaussians_{layer_idx}'] = gs[b].detach().cpu().numpy()

        return result_list
