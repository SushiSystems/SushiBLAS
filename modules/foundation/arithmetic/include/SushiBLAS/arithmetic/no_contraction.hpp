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
