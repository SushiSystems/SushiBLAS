/**************************************************************************/
/* no_contraction.hpp                                                     */
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
 * @file no_contraction.hpp
 * @brief Declares SB_FP_CONTRACT_OFF, which forbids fusing a product into a sum in one block.
 * @author Mustafa Garip
 */

#pragma once

/**
 * @brief Forbids the compiler to fuse a * b + c in the enclosing compound statement.
 * @note Written first in a block; a fused step a kernel wants is sycl::fma or std::fma.
 * @see modules/foundation/arithmetic/README.md
 */
#if defined(__clang__)
#define SB_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define SB_FP_CONTRACT_OFF _Pragma("STDC FP_CONTRACT OFF")
#endif
