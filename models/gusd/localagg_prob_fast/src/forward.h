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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
        const int P,
        const int* points_xyz,
        const int* radii,
        const dim3 grid,
        uint32_t* tiles_touched,
        const int tile_size);

	// Main rasterization method.
	void render(
        const int N,
        const float* pts,
        const int* points_int,
        const dim3 grid,
        const uint2* ranges,
        const uint32_t* point_list,
        const float* means3D,
        const float* cov3D,
        const float* opas,
        const float* semantic,
        float* out_logits,
        float* out_bin_logits,
        float* out_density,
        float* out_probability,
        const int* means_int,
        const int* radii,
        const int tile_size, const bool active_tile, const int tile_count,
        const uint32_t* active_tiles, const uint32_t* active_count);

	// Sparse variant: reconstruct integer/world-space centers from sorted
	// global voxel indices, avoiding Kx3 coordinate tensors.
	void renderSparse(
		const int N,
		const int64_t* linear_indices,
		const float* voxel_axes,
		const dim3 grid,
		const uint2* ranges,
		const uint32_t* point_list,
		const float* means3D,
		const float* cov3D,
		const float* opas,
		const float* semantic,
		float* out_logits,
		float* out_bin_logits,
		float* out_density,
		float* out_probability,
		const int* means_int,
		const int* radii,
		const int tile_size);

	void markSupport(
		const dim3 grid,
		const uint2* ranges,
		const uint32_t* point_list,
		const int* means_int,
		const int* radii,
		bool* support_mask,
		const uint32_t* active_tiles,
		const uint32_t* active_count,
		const int tile_size);
}


#endif
