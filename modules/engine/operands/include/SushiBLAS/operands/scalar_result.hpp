/**************************************************************************/
/* scalar_result.hpp                                                      */
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
 * @file scalar_result.hpp
 * @brief Declares the dtype and the shape check of a scalar result.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the dtype a norm or absolute-sum result holds for an operand of dtype.
         * @return The real component type of a complex dtype, dtype itself otherwise.
         */
        [[nodiscard]] inline constexpr Core::DataType real_result_dtype(Core::DataType dtype)
        {
            if (dtype == Core::DataType::Complex32)
                return Core::DataType::Float32;
            if (dtype == Core::DataType::Complex64)
                return Core::DataType::Float64;
            return dtype;
        }

        /**
         * @brief Throws unless a scalar result tensor has storage, the expected dtype and exactly
         *        one element.
         * @throws std::invalid_argument naming the first rule that fails.
         */
        inline void require_scalar_result(const Tensor& result, Core::DataType expected,
                                          const char* op)
        {
            Dispatch::require_storage(result, op, "result");
            Dispatch::require_dtype(result, expected, op, "result");
            if (result.num_elements != 1)
                Dispatch::Detail::throw_operand_error(op, "result",
                                                      "holds {} elements; a scalar result holds "
                                                      "exactly one",
                                                      result.num_elements);
        }
    } // namespace Internal
} // namespace SushiBLAS
