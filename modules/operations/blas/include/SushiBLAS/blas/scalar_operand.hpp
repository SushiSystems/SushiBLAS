/**************************************************************************/
/* scalar_operand.hpp                                                     */
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
 * @file scalar_operand.hpp
 * @brief Declares the record-time check on a tensor that stands for one scalar of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Throws unless a tensor is one stored element of the expected dtype.
         * @throws std::invalid_argument on missing storage, another dtype or an element count
         *         other than one.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         * @see modules/operations/blas/README.md
         */
        inline void require_scalar_operand(const Tensor& t, Core::DataType expected,
                                           const char* op, const char* operand)
        {
            require_storage(t, op, operand);
            require_dtype(t, expected, op, operand);
            require_scalar(t, op, operand);
            require_device_dtype(t, op, operand);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
