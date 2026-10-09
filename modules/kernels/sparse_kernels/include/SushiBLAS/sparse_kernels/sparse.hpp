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

#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>

#include <SushiBLAS/sparse_kernels/reference/structure.hpp>
#include <SushiBLAS/sparse_kernels/reference/level1.hpp>
#include <SushiBLAS/sparse_kernels/sycl/level1.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spmv.hpp>
#include <SushiBLAS/sparse_kernels/reference/spsymv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spsymv.hpp>
#include <SushiBLAS/sparse_kernels/reference/spmm.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spmm.hpp>
#include <SushiBLAS/sparse_kernels/reference/sptrsv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/sptrsv.hpp>
