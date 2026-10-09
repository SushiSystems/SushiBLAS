/**************************************************************************/
/* random.hpp                                                             */
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
 * @file random.hpp
 * @brief Includes the random kernels: the SYCL ones and the reference orthonormalisation.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/random_kernels/sycl/philox.hpp>
#include <SushiBLAS/random_kernels/sycl/poisson.hpp>
#include <SushiBLAS/random_kernels/sycl/truncated_normal.hpp>
#include <SushiBLAS/random_kernels/sycl/orthonormalize.hpp>
#include <SushiBLAS/random_kernels/reference/orthonormalize.hpp>
