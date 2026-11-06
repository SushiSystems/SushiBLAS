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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
