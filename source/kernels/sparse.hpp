/**************************************************************************/
/* sparse.hpp                                                             */
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
 * @file sparse.hpp
 * @brief Includes every sparse kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

#include "kernels/sparse/csr_pattern.hpp"
#include "kernels/sparse/index_list.hpp"

#include "kernels/reference/sparse/structure.hpp"
#include "kernels/reference/sparse/level1.hpp"
#include "kernels/sycl/sparse/level1.hpp"
#include "kernels/sycl/sparse/row_blocks.hpp"
#include "kernels/reference/sparse/spmv.hpp"
#include "kernels/sycl/sparse/spmv.hpp"
#include "kernels/reference/sparse/spsymv.hpp"
#include "kernels/sycl/sparse/spsymv.hpp"
#include "kernels/reference/sparse/spmm.hpp"
#include "kernels/sycl/sparse/spmm.hpp"
#include "kernels/reference/sparse/sptrsv.hpp"
#include "kernels/sycl/sparse/sptrsv.hpp"
