/****************************************************************/
/* spatial.hpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

// Private to this tree, spelled with quoted repo-root-relative paths, for the
// reason kernels/blas.hpp states: a kernel a *public* seam instantiates in a
// consumer's translation unit has to live under include/, and nothing here is
// one. The spatial operations are reached only through SpatialOps, whose
// methods are compiled into this library.

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
