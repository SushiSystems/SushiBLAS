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

#include "kernels/reference/spatial/im2col.hpp"
#include "kernels/reference/spatial/col2im.hpp"
#include "kernels/reference/spatial/pool_max.hpp"
#include "kernels/reference/spatial/pool_avg.hpp"
#include "kernels/reference/spatial/upsample.hpp"
#include "kernels/reference/spatial/channels.hpp"
#include "kernels/reference/spatial/broadcast.hpp"
#include "kernels/sycl/spatial/im2col.hpp"
#include "kernels/sycl/spatial/col2im.hpp"
#include "kernels/sycl/spatial/pool_max.hpp"
#include "kernels/sycl/spatial/pool_avg.hpp"
#include "kernels/sycl/spatial/upsample.hpp"
#include "kernels/sycl/spatial/channels.hpp"
#include "kernels/sycl/spatial/broadcast.hpp"
