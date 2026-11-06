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
