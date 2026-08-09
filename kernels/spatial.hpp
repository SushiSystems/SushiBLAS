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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
