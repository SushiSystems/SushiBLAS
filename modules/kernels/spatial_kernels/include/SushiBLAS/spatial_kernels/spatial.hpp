/**************************************************************************/
/* spatial.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file spatial.hpp
 * @brief Includes every spatial kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

// Every spatial kernel is private to this tree; see kernels/README.md.

#include <SushiBLAS/spatial_kernels/reference/im2col.hpp>
#include <SushiBLAS/spatial_kernels/reference/col2im.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_max.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_avg.hpp>
#include <SushiBLAS/spatial_kernels/reference/upsample.hpp>
#include <SushiBLAS/spatial_kernels/reference/channels.hpp>
#include <SushiBLAS/spatial_kernels/reference/broadcast.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot_weights.hpp>
#include <SushiBLAS/spatial_kernels/sycl/im2col.hpp>
#include <SushiBLAS/spatial_kernels/sycl/col2im.hpp>
#include <SushiBLAS/spatial_kernels/sycl/pool_max.hpp>
#include <SushiBLAS/spatial_kernels/sycl/pool_avg.hpp>
#include <SushiBLAS/spatial_kernels/sycl/upsample.hpp>
#include <SushiBLAS/spatial_kernels/sycl/channels.hpp>
#include <SushiBLAS/spatial_kernels/sycl/broadcast.hpp>
#include <SushiBLAS/spatial_kernels/sycl/window_dot.hpp>
#include <SushiBLAS/spatial_kernels/sycl/window_dot_weights.hpp>
