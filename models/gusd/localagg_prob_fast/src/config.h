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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#ifndef NUM_LOGITS_CHANNELS
// #define NUM_LOGITS_CHANNELS 18 // Default 3, RGB
#error "Check setup.py: NUM_LOGITS_CHANNELS must be defined as a macro, e.g., -DNUM_LOGITS_CHANNELS=18"
#endif

#endif