/****************************************************************/
/* blas.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file blas.hpp
 * @brief Includes every BLAS kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

// A kernel a public seam names lives under include/SushiBLAS/kernels/; see kernels/README.md.

// Level 1
#include "kernels/reference/blas/level1/level1.hpp"
#include "kernels/sycl/blas/level1/level1.hpp"

// Level 2
#include "kernels/reference/blas/level2/gemv.hpp"
#include "kernels/sycl/blas/level2/gemv.hpp"
#include "kernels/reference/blas/level2/ger.hpp"
#include "kernels/sycl/blas/level2/ger.hpp"
#include "kernels/reference/blas/level2/symv.hpp"
#include "kernels/sycl/blas/level2/symv.hpp"
#include "kernels/reference/blas/level2/syr.hpp"
#include "kernels/sycl/blas/level2/syr.hpp"
#include "kernels/reference/blas/level2/syr2.hpp"
#include "kernels/sycl/blas/level2/syr2.hpp"
#include "kernels/reference/blas/level2/trmv.hpp"
#include "kernels/sycl/blas/level2/trmv.hpp"

// Level 3
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>
#include "kernels/reference/blas/level3/syrk.hpp"
#include "kernels/sycl/blas/level3/syrk.hpp"
#include "kernels/reference/blas/level3/trsm.hpp"
#include "kernels/sycl/blas/level3/trsm.hpp"
