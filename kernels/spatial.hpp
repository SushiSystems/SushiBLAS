/****************************************************************/
/* spatial.hpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
