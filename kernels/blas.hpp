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
