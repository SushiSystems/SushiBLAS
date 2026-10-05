/****************************************************************/
/* blas.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

// Most kernels under kernels/ are private to this tree and spelled with a
// quoted, repo-root-relative path. GEMM's two are the exception and are spelled
// as angle-bracket <SushiBLAS/...> includes: they live under include/ and are
// installed, because `Dispatch::execute_gemm`
// (<SushiBLAS/engine/blas/dispatch.hpp>) instantiates them in a *consumer's*
// translation unit with the consumer's own epilogue functor. A template that a
// public seam instantiates cannot stay private, so the rule this file follows
// is: a kernel a public seam names lives under include/SushiBLAS/kernels/,
// mirroring the same reference/sycl split; everything else stays here.

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
