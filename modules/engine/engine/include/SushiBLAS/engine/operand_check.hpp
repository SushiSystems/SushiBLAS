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
 * @file operand_check.hpp
 * @brief Declares the record-time checks an operation runs on its tensor operands.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/engine_log_category.hpp>
#include <SushiBLAS/engine/tensor_profile.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/region/buffer_region.hpp>
#include <SushiLog/text/text_format.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Logs and throws std::invalid_argument naming the operation, the operand and the rule.
             * @param fmt Format of the violated rule; args fill its placeholders.
             * @see modules/engine/engine/README.md
             */
            template<typename... Args>
            [[noreturn]] inline void throw_operand_error(const char* op, const char* operand,
                                                         const char* fmt, const Args&... args)
            {
                const std::string rule = SushiLog::format_text(fmt, args...);
                const std::string msg = SushiLog::format_text("{}: operand '{}' {}", op, operand, rule);
                SB_LOG_ERROR(K_ENGINE_LOG_CATEGORY, "Invalid operand: {}", msg);
                throw std::invalid_argument(msg);
            }

            /**
             * @brief Logs and throws UnsupportedExecution naming the operation and the refusal.
             * @param fmt Format of what the mode or the device refuses; args fill its placeholders.
             * @see modules/engine/engine/README.md
             */
            template<typename... Args>
            [[noreturn]] inline void throw_execution_error(const char* op, const char* fmt,
                                                           const Args&... args)
            {
                const std::string refusal = SushiLog::format_text(fmt, args...);
                const std::string msg = SushiLog::format_text("{}: {}", op, refusal);
                SB_LOG_ERROR(K_ENGINE_LOG_CATEGORY, "Unsupported execution: {}", msg);
                throw UnsupportedExecution(msg);
            }

            /**
             * @brief Logs and throws std::invalid_argument naming an operation and the dtype it does not serve.
             * @param family What kind of operation this is, such as "Elementwise operation".
             * @see modules/engine/engine/README.md
             */
            [[noreturn]] inline void throw_dtype_error(const char* family, const char* name,
                                                       Core::DataType dtype)
            {
                const std::string msg = SushiLog::format_text(
                    "{} '{}' does not support dtype {}. Complex operands need an ordering or a "
                    "real-only builtin they do not provide; integer operands are accepted only "
                    "where an integer element means something (copy, fill, the comparison and "
                    "logic family, and the index reductions).",
                    family, name, Core::to_string(dtype));
                SB_LOG_ERROR(K_ENGINE_LOG_CATEGORY, "Invalid operand: {}", msg);
                throw std::invalid_argument(msg);
            }
        } // namespace Detail

        /**
         * @brief Holds what the storage of a tensor adds to the record-time checks, apart from
         *        its shape: whether it exists, the bytes it covers and what its device computes.
         *
         * A test builds one by hand, so the checks that read it run without a runtime.
         * @see modules/engine/engine/README.md
         */
        struct StorageFacts
        {
            bool present = false;                   /**< Whether the tensor has storage. */
            SushiRuntime::BufferRegion region{};    /**< The bytes the tensor covers; empty without storage. */
            bool supports_fp64 = true;              /**< Whether the device of the storage computes FLOAT64. */
            bool supports_fp16 = true;              /**< Whether the device of the storage computes HALF. */
        };

        /**
         * @brief Pairs a tensor, which holds the shape, with the facts of its storage.
         * @note The tensor is held by reference and must outlive the pair.
         * @see modules/engine/engine/README.md
         */
        struct PlacedTensor
        {
            const Tensor& tensor;   /**< The dtype, shape, strides and layout the checks read. */
            StorageFacts storage;   /**< The storage the checks read. */
        };

        /**
         * @brief Reads the facts of the storage of a tensor.
         * @return The facts; the device is asked only for a dtype that needs fp64 or fp16.
         * @see modules/engine/engine/README.md
         */
        inline StorageFacts storage_facts_of(const Tensor& t)
        {
            StorageFacts facts;
            facts.present = static_cast<bool>(t.storage);
            facts.region = t.region();

            const bool needs_fp64 = t.dtype == Core::DataType::Float64 ||
                                    t.dtype == Core::DataType::Complex64;
            const bool needs_fp16 = t.dtype == Core::DataType::Half;
            if (!facts.present || (!needs_fp64 && !needs_fp16))
                return facts;

            const Device::Profile& profile = profile_of(t);
            facts.supports_fp64 = profile.supports_fp64;
            facts.supports_fp16 = profile.supports_fp16;
            return facts;
        }

        /** @brief Returns t paired with the facts of its storage. */
        inline PlacedTensor placed(const Tensor& t)
        {
            return PlacedTensor{t, storage_facts_of(t)};
        }

        /**
         * @brief Throws unless a tensor holding elements has storage.
         * @param facts The storage of t.
         * @throws std::invalid_argument when num_elements > 0 and the storage is absent; the
         *         message says so when t is shape-only.
         * @see modules/engine/engine/README.md
         */
        inline void require_storage(const Tensor& t, const StorageFacts& facts, const char* op,
                                    const char* operand)
        {
            if (t.num_elements > 0 && !facts.present && t.shape_only)
                Detail::throw_operand_error(op, operand,
                                            "is a shape-only tensor of {} elements; this check "
                                            "needs memory and a shape-only tensor carries none",
                                            t.num_elements);
            if (t.num_elements > 0 && !facts.present)
                Detail::throw_operand_error(op, operand, "holds {} elements but has no storage",
                                            t.num_elements);
        }

        /**
         * @brief Throws unless a tensor holding elements has storage.
         * @throws std::invalid_argument when num_elements > 0 and storage is null.
         * @see modules/engine/engine/README.md
         */
        inline void require_storage(const Tensor& t, const char* op, const char* operand)
        {
            StorageFacts facts;
            facts.present = static_cast<bool>(t.storage);
            require_storage(t, facts, op, operand);
        }

        /**
         * @brief Throws unless the bytes of one operand share nothing with the bytes of another.
         * @param written The operand the message names.
         * @param other The operand it must not overlap; named in the message.
         * @throws std::invalid_argument naming both operands.
         * @see modules/engine/engine/README.md
         */
        inline void require_disjoint(const StorageFacts& written, const StorageFacts& other,
                                     const char* op, const char* written_name,
                                     const char* other_name)
        {
            if (written.region.overlaps(other.region))
                Detail::throw_operand_error(op, written_name, "overlaps '{}' in memory",
                                            other_name);
        }

        /**
         * @brief Throws unless a tensor holds the expected dtype.
         * @throws std::invalid_argument naming both dtypes on a mismatch.
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
         */
        inline void require_same_layout(const Tensor& a, const Tensor& b, const char* op,
                                        const char* operand_a, const char* operand_b)
        {
            if (a.rank < 2 || b.rank < 2 || a.layout == b.layout)
                return;

            Detail::throw_operand_error(op, operand_a, "has layout {}, but '{}' has layout {}",
                                        a.layout == Core::Layout::RowMajor ? "row-major"
                                                                            : "column-major",
                                        operand_b,
                                        b.layout == Core::Layout::RowMajor ? "row-major"
                                                                            : "column-major");
        }

        /**
         * @brief Throws unless two tensors hold the same number of elements.
         * @throws std::invalid_argument naming both counts on a mismatch.
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
                                                t.layout == Core::Layout::RowMajor ? "row-major"
                                                                                    : "column-major",
                                                expected[i]);
            }
        }

        /**
         * @brief Throws unless ok holds; the form for a rule no named check states.
         * @param fmt Format of the violated rule, read after "operand '<operand>' ".
         * @throws std::invalid_argument naming the operation, the operand and the rule.
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
         * @see modules/engine/engine/README.md
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
         * @param facts The storage of t, which says what its device computes.
         * @throws UnsupportedExecution for FLOAT64 or COMPLEX64 storage on a device without the
         *         fp64 aspect, and for HALF storage on a device without the fp16 aspect.
         * @see modules/engine/engine/README.md
         */
        inline void require_device_dtype(const Tensor& t, const StorageFacts& facts, const char* op,
                                         const char* operand)
        {
            const bool needs_fp64 = t.dtype == Core::DataType::Float64 ||
                                    t.dtype == Core::DataType::Complex64;
            const bool needs_fp16 = t.dtype == Core::DataType::Half;
            if ((!needs_fp64 && !needs_fp16) || !facts.present)
                return;

            if (needs_fp64 && !facts.supports_fp64)
                Detail::throw_execution_error(op,
                                              "operand '{}' has dtype {} but lives on a device "
                                              "without the fp64 aspect",
                                              operand, Core::to_string(t.dtype));
            if (needs_fp16 && !facts.supports_fp16)
                Detail::throw_execution_error(op,
                                              "operand '{}' has dtype {} but lives on a device "
                                              "without the fp16 aspect",
                                              operand, Core::to_string(t.dtype));
        }

        /**
         * @brief Throws unless the device a tensor lives on can compute in the tensor's dtype.
         * @throws UnsupportedExecution as the overload that takes the facts does.
         * @see modules/engine/engine/README.md
         */
        inline void require_device_dtype(const Tensor& t, const char* op, const char* operand)
        {
            require_device_dtype(t, storage_facts_of(t), op, operand);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
