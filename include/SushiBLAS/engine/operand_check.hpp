/**************************************************************************/
/* operand_check.hpp                                                      */
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
 * @file operand_check.hpp
 * @brief Declares the record-time checks an operation runs on its tensor operands.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Logs and throws std::invalid_argument naming the operation, the operand and the rule.
             * @param fmt Format of the violated rule; args fill its placeholders.
             * @see include/SushiBLAS/engine/README.md
             */
            template<typename... Args>
            [[noreturn]] inline void throw_operand_error(const char* op, const char* operand,
                                                         const char* fmt, const Args&... args)
            {
                const std::string rule = Core::Detail::format(fmt, args...);
                const std::string msg = Core::Detail::format("{}: operand '{}' {}", op, operand, rule);
                SB_LOG_ERROR("Invalid operand: {}", msg);
                throw std::invalid_argument(msg);
            }
        } // namespace Detail

        /**
         * @brief Throws unless a tensor holding elements has storage.
         * @throws std::invalid_argument when num_elements > 0 and storage is null.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_storage(const Tensor& t, const char* op, const char* operand)
        {
            if (t.num_elements > 0 && !t.storage)
                Detail::throw_operand_error(op, operand, "holds {} elements but has no storage",
                                            t.num_elements);
        }

        /**
         * @brief Throws unless a tensor holds the expected dtype.
         * @throws std::invalid_argument naming both dtypes on a mismatch.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_dtype(const Tensor& t, Core::DataType expected, const char* op,
                                  const char* operand)
        {
            if (t.dtype != expected)
                Detail::throw_operand_error(op, operand, "has dtype {}, expected {}",
                                            Core::to_string(t.dtype), Core::to_string(expected));
        }

        /**
         * @brief Throws unless two tensors have the same rank and the same size in every dimension.
         * @throws std::invalid_argument naming the first rank or dimension that differs.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_same_shape(const Tensor& a, const Tensor& b, const char* op,
                                       const char* operand_a, const char* operand_b)
        {
            if (a.rank != b.rank)
                Detail::throw_operand_error(op, operand_a, "has rank {}, but '{}' has rank {}",
                                            a.rank, operand_b, b.rank);

            for (int32_t i = 0; i < a.rank; ++i)
            {
                if (a.shape[i] != b.shape[i])
                    Detail::throw_operand_error(op, operand_a,
                                                "has size {} in dimension {}, but '{}' has size {}",
                                                a.shape[i], i, operand_b, b.shape[i]);
            }
        }

        /**
         * @brief Throws unless two tensors of rank 2 or more share one memory layout.
         * @throws std::invalid_argument naming both layouts on a mismatch.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_same_layout(const Tensor& a, const Tensor& b, const char* op,
                                        const char* operand_a, const char* operand_b)
        {
            if (a.rank < 2 || b.rank < 2 || a.layout == b.layout)
                return;

            Detail::throw_operand_error(op, operand_a, "has layout {}, but '{}' has layout {}",
                                        a.layout == Core::Layout::ROW_MAJOR ? "row-major"
                                                                            : "column-major",
                                        operand_b,
                                        b.layout == Core::Layout::ROW_MAJOR ? "row-major"
                                                                            : "column-major");
        }

        /**
         * @brief Throws unless two tensors hold the same number of elements.
         * @throws std::invalid_argument naming both counts on a mismatch.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_same_element_count(const Tensor& a, const Tensor& b, const char* op,
                                               const char* operand_a, const char* operand_b)
        {
            if (a.num_elements != b.num_elements)
                Detail::throw_operand_error(op, operand_a, "holds {} elements, but '{}' holds {}",
                                            a.num_elements, operand_b, b.num_elements);
        }

        /**
         * @brief Throws unless a tensor holds at least min_elements elements.
         * @throws std::invalid_argument naming the count and the minimum.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_min_elements(const Tensor& t, int64_t min_elements, const char* op,
                                         const char* operand)
        {
            if (t.num_elements < min_elements)
                Detail::throw_operand_error(op, operand, "holds {} elements, at least {} required",
                                            t.num_elements, min_elements);
        }

        /**
         * @brief Throws unless a tensor's strides are the dense strides of its shape under its layout.
         * @throws std::invalid_argument on a zero or negative stride, a transposed view or a slice.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_contiguous(const Tensor& t, const char* op, const char* operand)
        {
            if (t.num_elements == 0)
                return;

            const auto expected = Tensor::layout_strides(
                SushiRuntime::span<const int64_t>(t.shape.data(), static_cast<size_t>(t.rank)),
                t.layout);

            for (int32_t i = 0; i < t.rank; ++i)
            {
                if (t.shape[i] == 1)
                    continue;

                if (t.strides[i] <= 0)
                    Detail::throw_operand_error(op, operand,
                                                "has stride {} in dimension {}; strides must be positive",
                                                t.strides[i], i);

                if (t.strides[i] != expected[i])
                    Detail::throw_operand_error(op, operand,
                                                "has stride {} in dimension {}, dense {} needs {}; "
                                                "the operation does not accept strided views",
                                                t.strides[i], i,
                                                t.layout == Core::Layout::ROW_MAJOR ? "row-major"
                                                                                    : "column-major",
                                                expected[i]);
            }
        }
    } // namespace Dispatch
} // namespace SushiBLAS
