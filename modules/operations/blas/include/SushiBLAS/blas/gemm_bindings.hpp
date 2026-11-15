/**************************************************************************/
/* gemm_bindings.hpp                                                      */
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
 * @file gemm_bindings.hpp
 * @brief Declares GemmBindings, where A, B and C sit in the operands a GEMM task bound by handle.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/operand_pointers.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Names where A, B and C sit in the table of operands a GEMM task bound: slots 0, 1
         *        and 2, each with the offset of its first element in its allocation.
         * @see modules/operations/blas/README.md
         */
        struct GemmBindings
        {
            OperandBinding a; /**< The slot and offset of A. */
            OperandBinding b; /**< The slot and offset of B. */
            OperandBinding c; /**< The slot and offset of C. */
        };
    } // namespace Dispatch
} // namespace SushiBLAS
