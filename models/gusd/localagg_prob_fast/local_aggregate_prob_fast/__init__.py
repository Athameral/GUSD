#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import warnings

import torch
import torch.nn as nn

from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, points_int, means3D, means3D_int, opas, semantics,
                radii, cov3D, H, W, D, tile_size):
        result = _C.local_aggregate(
            pts, points_int, means3D, means3D_int, opas, semantics,
            radii, cov3D, H, W, D, tile_size)
        count, logits, bin_logits, density, probability, geom, bins, image = result
        ctx.num_rendered = count
        ctx.grid = (H, W, D)
        ctx.tile_size = tile_size
        ctx.save_for_backward(geom, bins, image, means3D, pts, points_int,
                              means3D_int, radii, cov3D, opas, semantics,
                              logits, bin_logits, density, probability)
        return logits, bin_logits, density

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):
        (geom, bins, image, means, pts, points_int, means_int, radii, cov,
         opas, semantics, logits, bin_logits, density, probability) = ctx.saved_tensors
        grads = _C.local_aggregate_backward(
            geom, bins, image, *ctx.grid, ctx.num_rendered, means, pts,
            points_int, means_int, radii, cov, opas, semantics, logits,
            bin_logits, density, probability, logits_grad, bin_logits_grad,
            density_grad, ctx.tile_size)
        means_grad, opas_grad, semantics_grad, cov_grad = grads
        return (None, None, means_grad, None, opas_grad, semantics_grad,
                None, cov_grad, None, None, None, None)


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1,
                 chunk_size=None, chunk_num=None, chunk_margin=None, tile_size=4):
        """Gaussian aggregation with runtime-selectable coarse spatial tiles.

        tile_size=2 prioritizes speed, 4 balances memory and speed, and 8
        prioritizes index memory. Legacy chunk options are accepted, but no
        longer perform spatial splitting or Gaussian-batch compositing.
        """
        super().__init__()
        if isinstance(tile_size, bool) or tile_size not in (2, 4, 8):
            raise ValueError('tile_size must be 2, 4 or 8')
        if chunk_size is not None or chunk_num is not None or chunk_margin is not None:
            warnings.warn('chunk_size/chunk_num/chunk_margin are deprecated and '
                          'ignored; use tile_size=2, 4 or 8 instead.',
                          FutureWarning, stacklevel=2)
        self.scale_multiplier = scale_multiplier
        self.H, self.W, self.D = H, W, D
        self.grid_size = grid_size
        self.radii_min = radii_min
        self.tile_size = int(tile_size)
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))

    def forward(self, pts, means3D, opas, semantics, scales, cov3D):
        assert pts.shape[0] == 1
        assert not pts.requires_grad
        pts = pts.squeeze(0).float().contiguous()
        means = means3D.squeeze(0).float().contiguous()
        opas = opas.squeeze(0).float().contiguous()
        semantics = semantics.squeeze(0).float().contiguous()
        cov = cov3D.squeeze(0).float()
        scales = scales.detach().squeeze(0).float()
        if pts.shape[0] == 0 or means.shape[0] == 0:
            link = (means.sum() + opas.sum() + semantics.sum() + cov.sum()) * 0
            logits = pts.new_zeros((pts.shape[0], semantics.shape[-1]))
            if means.shape[0] == 0 and semantics.shape[-1] > 1:
                logits[:, :-1] = 1.0 / (semantics.shape[-1] - 1)
            return logits + link, pts.new_zeros(pts.shape[0]) + link, pts.new_zeros(pts.shape[0]) + link
        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int).contiguous()
        means_int = ((means.detach() - self.pc_min) / self.grid_size).to(torch.int).contiguous()
        grid = points_int.new_tensor([self.H, self.W, self.D])
        assert ((points_int >= 0) & (points_int < grid)).all()
        assert ((means_int >= 0) & (means_int < grid)).all()
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min).contiguous()
        assert radii.min() >= 1
        cov_flat = cov.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()
        return _LocalAggregate.apply(pts, points_int, means, means_int, opas,
                                     semantics, radii, cov_flat, self.H,
                                     self.W, self.D, self.tile_size)
