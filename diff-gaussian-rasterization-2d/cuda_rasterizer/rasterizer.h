/*
 * Copyright (C) 2025, Fudan Zhang Vision Group
 * All rights reserved.
 * 
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static void markVisible(
			int P,
			float *means3D,
			float *viewmatrix,
			float *projmatrix,
			bool *present);

		static int forward(
			std::function<char *(size_t)> geometryBuffer,
			std::function<char *(size_t)> binningBuffer,
			std::function<char *(size_t)> imageBuffer,
			const int P, const int S, int D, int M,
			const float *background,
			const int width, int height,
			const float *means3D,
			const float *shs,
			const float *colors_precomp,
			const float *flows_precomp,
			const float *opacities,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const float *cov3D_precomp,
			const bool* mask,
			const float *viewmatrix,
			const float *projmatrix,
			const float *cam_pos,
			const float tan_fovx, 
			const float tan_fovy,
			const bool prefiltered,
			int* out_contrib,
			float *out_color,
			float *out_feature,
			float *out_depth,
			float *out_T,
			int *radii = nullptr,
			bool debug = false,
			const float vfov_min = -90,
			const float vfov_max = 90,
			const float hfov_min = -180,
			const float hfov_max = 180,
			const float scale_factor = 1.0);

		static void backward(
			const int P, const int S, int D, int M, int R,
			const float *background,
			const int width, int height,
			const float *means3D,
			const float *shs,
			const float *colors_precomp,
			const float *flows_2d,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const float *cov3D_precomp,
			const float *viewmatrix,
			const float *projmatrix,
			const float *campos,
			const float tan_fovx, float tan_fovy,
			const int *radii,
			char *geom_buffer,
			char *binning_buffer,
			char *image_buffer,
			const int* out_contrib,
			const float *dL_dpix,
			const float *dL_depths,
			const float *dL_masks,
			const float *dL_dpix_flow,
			float *dL_dmean2D,
			float *dL_dopacity,
			float *dL_dcolor,
			float *dL_dmean3D,
			float *dL_dcov3D,
			float *dL_dsh,
			float *dL_dflows,
			float *dL_dscale,
			float *dL_drot,
			float *dL_dtransMat,
			float *dL_dnormals,
			bool debug,
			const float vfov_min,
			const float vfov_max,
			const float hfov_min,
			const float hfov_max,
			const float scale_factor);
	};
};

#endif