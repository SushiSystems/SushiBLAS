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
 * @file blas.hpp
 * @brief Includes every BLAS kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

// A kernel a public seam names lives under modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/; see kernels/README.md.

// Level 1
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/level1.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotg.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotg.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotm.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotmg.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotmg.hpp>

// Level 2
#include <SushiBLAS/blas_kernels/reference/level2/gemv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/gemv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/ger.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/ger.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/symv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/syr.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/syr.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/syr2.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/syr2.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/trsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/gbmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/gbmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/sbmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/sbmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/spmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/spmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tbmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tpmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tpmv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tbsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbsv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tpsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tpsv.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/spr.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/spr.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/spr2.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/spr2.hpp>

// Level 3
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syrk.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syr2k.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syr2k.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/trsm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/trsm.hpp>
