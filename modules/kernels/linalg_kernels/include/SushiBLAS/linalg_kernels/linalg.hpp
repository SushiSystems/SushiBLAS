/**************************************************************************/
/* linalg.hpp                                                             */
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
 * @file linalg.hpp
 * @brief Includes every dense linear algebra kernel, reference and SYCL.
 * @author Mustafa Garip
 */

#pragma once

// Every linear algebra kernel is private to this tree; see kernels/README.md.

#include <SushiBLAS/linalg_kernels/chunked_dot.hpp>
#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>
#include <SushiBLAS/linalg_kernels/reference/lu.hpp>
#include <SushiBLAS/linalg_kernels/reference/gmres.hpp>
#include <SushiBLAS/linalg_kernels/sycl/lu.hpp>
#include <SushiBLAS/linalg_kernels/sycl/gmres.hpp>
