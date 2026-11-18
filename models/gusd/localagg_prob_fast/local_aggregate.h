/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateCUDA(
	const torch::Tensor& pts,            // n, 3
	const torch::Tensor& points_int,
	const torch::Tensor& means3D,        // g, 3
	const torch::Tensor& means3D_int,
	const torch::Tensor& opas,
	const torch::Tensor& semantics,      // g, c
	const torch::Tensor& radii,          // g
	const torch::Tensor& cov3D,          // g, 6
	const int H, int W, int D, const int tile_size, const int render_mode);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateRenderCUDA(
	const torch::Tensor& linear_indices,
	const torch::Tensor& voxel_axes,
	const torch::Tensor& means3D,
	const torch::Tensor& means3D_int,
	const torch::Tensor& opas,
	const torch::Tensor& semantics,
	const torch::Tensor& radii,
	const torch::Tensor& cov3D,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int R, const int H, const int W, const int D, const int tile_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const torch::Tensor& means3D,
	const torch::Tensor& pts,
	const torch::Tensor& points_int,
	const torch::Tensor& linear_indices,
	const torch::Tensor& voxel_axes,
	const torch::Tensor& support_bits,
	const torch::Tensor& word_prefix,
	const torch::Tensor& means3D_int,
	const torch::Tensor& radii,
	const torch::Tensor& cov3D,
	const torch::Tensor& opas,
	const torch::Tensor& semantics,
	const torch::Tensor& logits,
	const torch::Tensor& bin_logits,
	const torch::Tensor& density,
	const torch::Tensor& probability,
	const torch::Tensor& logits_grad,
	const torch::Tensor& bin_logits_grad,
	const torch::Tensor& density_grad,
	const int tile_size);
