/**************************************************************************/
/* blas.hpp                                                               */
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

// Level 3
#include "kernels/reference/blas/level3/gemm.hpp"
#include "kernels/sycl/blas/level3/gemm.hpp"
#include "kernels/reference/blas/level3/syrk.hpp"
#include "kernels/sycl/blas/level3/syrk.hpp"

// Level 2 (Planned)
// #include "kernels/reference/blas/level2/gemv.hpp"
// #include "kernels/sycl/blas/level2/gemv.hpp"
