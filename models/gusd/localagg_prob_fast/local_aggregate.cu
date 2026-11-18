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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "src/config.h"
#include "src/aggregator.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const int H, int W, int D, const int tile_size, const int render_mode)
{
    TORCH_CHECK(tile_size == 2 || tile_size == 4 || tile_size == 8, "tile_size must be 2, 4 or 8");
	TORCH_CHECK(render_mode >= 0 && render_mode <= 2, "render_mode must be dense, active_tile or support_only");
  
	const int P = means3D.size(0);
	const int N = pts.size(0);
	TORCH_CHECK(render_mode != 1 || int64_t(N) == int64_t(H) * W * D, "active_tile requires the full ordered voxel grid");

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	const int output_n = render_mode == 2 ? 0 : N;
	torch::Tensor out_logits = torch::full({output_n, NUM_LOGITS_CHANNELS}, 0.0, float_opts);
	if (render_mode == 1) out_logits.slice(1, 0, NUM_LOGITS_CHANNELS - 1).fill_(1.0 / (NUM_LOGITS_CHANNELS - 1));
	torch::Tensor out_bin_logits = torch::full({output_n}, 0.0, float_opts);
	torch::Tensor out_density = torch::full({output_n}, 0.0, float_opts);
	torch::Tensor out_probability = torch::full({output_n}, 0.0, float_opts);
	torch::Tensor support_mask = torch::zeros({render_mode == 2 ? int64_t(H) * W * D : 0}, means3D.options().dtype(torch::kBool));
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	int rendered;
	rendered = LocalAggregator::Aggregator::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, N,
		pts.contiguous().data<float>(),
		points_int.contiguous().data<int>(),
		means3D.contiguous().data<float>(),
		means3D_int.contiguous().data<int>(),
		opas.contiguous().data<float>(),
		semantics.contiguous().data<float>(),
		cov3D.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		H, W, D,
		out_logits.contiguous().data<float>(),
		out_bin_logits.contiguous().data<float>(),
		out_density.contiguous().data<float>(),
		out_probability.contiguous().data<float>(),
		support_mask.data_ptr<bool>(),
		tile_size, render_mode);
	
	return std::make_tuple(rendered, out_logits, out_bin_logits, out_density, out_probability, support_mask, geomBuffer, binningBuffer, imgBuffer);
}

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
	const int R, const int H, const int W, const int D, const int tile_size)
{
	const int P = means3D.size(0);
	const int N = linear_indices.size(0);
	TORCH_CHECK(linear_indices.scalar_type() == torch::kInt64, "linear_indices must be int64");
	TORCH_CHECK(voxel_axes.numel() == H + W + D,
		"voxel_axes must contain H + W + D float32 centers");
	auto opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor logits = torch::zeros({N, NUM_LOGITS_CHANNELS}, opts);
	torch::Tensor bin_logits = torch::zeros({N}, opts);
	torch::Tensor density = torch::zeros({N}, opts);
	torch::Tensor probability = torch::zeros({N}, opts);
	LocalAggregator::Aggregator::renderOnly(
		P, R, N, H, W, D,
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		linear_indices.contiguous().data<int64_t>(),
		voxel_axes.contiguous().data<float>(),
		means3D.contiguous().data<float>(), means3D_int.contiguous().data<int>(),
		opas.contiguous().data<float>(), semantics.contiguous().data<float>(),
		cov3D.contiguous().data<float>(), radii.contiguous().data<int>(),
		logits.data_ptr<float>(), bin_logits.data_ptr<float>(),
		density.data_ptr<float>(), probability.data_ptr<float>(), tile_size);
	return std::make_tuple(logits, bin_logits, density, probability);
}

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
	const int tile_size)
{
    TORCH_CHECK(tile_size == 2 || tile_size == 4 || tile_size == 8, "tile_size must be 2, 4 or 8");
	const int P = means3D.size(0);
	const bool sparse_points = linear_indices.numel() > 0;
	const bool compressed_mapping = support_bits.numel() > 0;
	const int N = sparse_points ? linear_indices.size(0) : pts.size(0);
	
	torch::Tensor means3D_grad = torch::zeros({P, 3}, means3D.options());
	torch::Tensor opas_grad = torch::zeros({P}, means3D.options());
	torch::Tensor semantics_grad = torch::zeros({P, NUM_LOGITS_CHANNELS}, means3D.options());
	torch::Tensor cov3D_grad = torch::zeros({P, 6}, means3D.options());

	torch::Tensor voxel2pts = compressed_mapping
		? torch::empty({0}, means3D.options().dtype(torch::kInt32))
		: torch::full({H * W * D}, -1, means3D.options().dtype(torch::kInt32));
  
	LocalAggregator::Aggregator::backward(
		P, R, N,
		H, W, D,
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        points_int.contiguous().data<int>(),
		means3D_int.contiguous().data<int>(),
		radii.contiguous().data<int>(),
		voxel2pts.contiguous().data<int>(),
		compressed_mapping ? reinterpret_cast<const uint32_t*>(support_bits.contiguous().data_ptr<int32_t>()) : nullptr,
		compressed_mapping ? word_prefix.contiguous().data<int>() : nullptr,
		pts.contiguous().data<float>(),
		sparse_points ? linear_indices.contiguous().data<int64_t>() : nullptr,
		sparse_points ? voxel_axes.contiguous().data<float>() : nullptr,
		means3D.contiguous().data<float>(),
		cov3D.contiguous().data<float>(),
		opas.contiguous().data<float>(),
		semantics.contiguous().data<float>(),
		logits.contiguous().data<float>(),
		bin_logits.contiguous().data<float>(),
		density.contiguous().data<float>(),
		probability.contiguous().data<float>(),
		logits_grad.contiguous().data<float>(),
		bin_logits_grad.contiguous().data<float>(),
		density_grad.contiguous().data<float>(),
		means3D_grad.contiguous().data<float>(),
		opas_grad.contiguous().data<float>(),
		semantics_grad.contiguous().data<float>(),
		cov3D_grad.contiguous().data<float>(),
		tile_size, sparse_points, compressed_mapping);

	return std::make_tuple(means3D_grad, opas_grad, semantics_grad, cov3D_grad);
}
