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
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/device_policy.hpp>
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

            /**
             * @brief Logs and throws UnsupportedExecution naming the operation and the refusal.
             * @param fmt Format of what the mode or the device refuses; args fill its placeholders.
             * @see include/SushiBLAS/engine/README.md
             */
            template<typename... Args>
            [[noreturn]] inline void throw_execution_error(const char* op, const char* fmt,
                                                           const Args&... args)
            {
                const std::string refusal = Core::Detail::format(fmt, args...);
                const std::string msg = Core::Detail::format("{}: {}", op, refusal);
                SB_LOG_ERROR("Unsupported execution: {}", msg);
                throw UnsupportedExecution(msg);
            }

            /**
             * @brief Logs and throws std::invalid_argument naming an operation and the dtype it does not serve.
             * @param family What kind of operation this is, such as "Elementwise operation".
             * @see include/SushiBLAS/engine/README.md
             */
            [[noreturn]] inline void throw_dtype_error(const char* family, const char* name,
                                                       Core::DataType dtype)
            {
                const std::string msg = Core::Detail::format(
                    "{} '{}' does not support dtype {}. Complex operands need an ordering or a "
                    "real-only builtin they do not provide; integer operands are accepted only "
                    "where an integer element means something (copy, fill, the comparison and "
                    "logic family, and the index reductions).",
                    family, name, Core::to_string(dtype));
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

        /**
         * @brief Throws unless ok holds; the form for a rule no named check states.
         * @param fmt Format of the violated rule, read after "operand '<operand>' ".
         * @throws std::invalid_argument naming the operation, the operand and the rule.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename... Args>
        inline void require_operand(bool ok, const char* op, const char* operand, const char* fmt,
                                    const Args&... args)
        {
            if (!ok)
                Detail::throw_operand_error(op, operand, fmt, args...);
        }

        /**
         * @brief Throws unless a tensor has the expected rank.
         * @throws std::invalid_argument naming both ranks.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_rank(const Tensor& t, int32_t expected, const char* op,
                                 const char* operand)
        {
            if (t.rank != expected)
                Detail::throw_operand_error(op, operand, "has rank {}, expected {}", t.rank,
                                            expected);
        }

        /**
         * @brief Throws unless a tensor holds exactly one element.
         * @throws std::invalid_argument naming the count.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_scalar(const Tensor& t, const char* op, const char* operand)
        {
            if (t.num_elements != 1)
                Detail::throw_operand_error(op, operand,
                                            "holds {} elements; a scalar holds exactly one",
                                            t.num_elements);
        }

        /**
         * @brief Throws unless an extent of an operand equals the one another operand fixes.
         * @param what Names the extent in the message, such as "channels" or "rows".
         * @throws std::invalid_argument naming both extents.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_extent(int64_t actual, int64_t expected, const char* what,
                                   const char* op, const char* operand)
        {
            if (actual != expected)
                Detail::throw_operand_error(op, operand, "has {} {}, expected {}", actual, what,
                                            expected);
        }

        /**
         * @brief Throws unless the device a tensor lives on can compute in the tensor's dtype.
         * @throws UnsupportedExecution for FLOAT64 or COMPLEX64 storage on a device without the
         *         fp64 aspect, and for HALF storage on a device without the fp16 aspect.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_device_dtype(const Tensor& t, const char* op, const char* operand)
        {
            const bool needs_fp64 = t.dtype == Core::DataType::FLOAT64 ||
                                    t.dtype == Core::DataType::COMPLEX64;
            const bool needs_fp16 = t.dtype == Core::DataType::HALF;
            if ((!needs_fp64 && !needs_fp16) || !t.storage)
                return;

            const Device::Profile& profile = Device::profile_for(t.get_device());
            if (needs_fp64 && !profile.supports_fp64)
                Detail::throw_execution_error(op,
                                              "operand '{}' has dtype {} but lives on a device "
                                              "without the fp64 aspect",
                                              operand, Core::to_string(t.dtype));
            if (needs_fp16 && !profile.supports_fp16)
                Detail::throw_execution_error(op,
                                              "operand '{}' has dtype {} but lives on a device "
                                              "without the fp16 aspect",
                                              operand, Core::to_string(t.dtype));
        }
    } // namespace Dispatch
} // namespace SushiBLAS
