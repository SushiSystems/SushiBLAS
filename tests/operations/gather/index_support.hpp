/**************************************************************************/
/* index_support.hpp                                                      */
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
 * @file index_support.hpp
 * @brief Declares the builders, the host oracles and the refusal check the row index tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace IndexSupport
        {
            /// @brief The value an output holds before its operation runs.
            constexpr double K_SENTINEL = 7.0;

            /**
             * @brief Names one element type and one index width a typed suite runs.
             * @tparam Value The element type of the data tensors.
             * @tparam Position The integer type of the index tensor.
             */
            template<typename Value, typename Position>
            struct Combo
            {
                using Element = Value;   /**< The element type of the data tensors. */
                using Index = Position;  /**< The integer type of the index tensor. */
            };

            /** @brief Returns whether the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                const sycl::device device = TestEnvironment::queue().get_device();

                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns value widened to double. */
            template<typename T>
            double widen(const T& value)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                    return static_cast<double>(static_cast<float>(value));
                else
                    return static_cast<double>(value);
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> elements_from(const std::vector<double>& values)
            {
                std::vector<T> elements(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                        elements[i] = sycl::half(static_cast<float>(values[i]));
                    else
                        elements[i] = static_cast<T>(values[i]);
                }
                return elements;
            }

            /** @brief Returns how many elements a tensor of dims holds. */
            inline std::size_t element_count(const std::vector<int64_t>& dims) noexcept
            {
                int64_t count = 1;
                for (const int64_t extent : dims)
                    count *= extent;
                return static_cast<std::size_t>(count);
            }

            /** @brief Returns positions as doubles, the form the tensor builders take. */
            inline std::vector<double> as_doubles(const std::vector<int64_t>& positions)
            {
                return std::vector<double>(positions.begin(), positions.end());
            }

            /** @brief Returns `((i * multiplier) % modulus) - shift` for every i below count. */
            inline std::vector<double> residue_pattern(std::size_t count, int64_t multiplier,
                                                       int64_t modulus, int64_t shift)
            {
                std::vector<double> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<double>(
                        (static_cast<int64_t>(i) * multiplier) % modulus - shift);
                return values;
            }

            /** @brief Returns `0.37 * v + 0.011` for every v; no element type holds it exactly. */
            inline std::vector<double> fractional(std::vector<double> values)
            {
                for (double& v : values)
                    v = 0.37 * v + 0.011;
                return values;
            }

            /** @brief Returns a dense tensor of dims holding values rounded to T. */
            template<typename T>
            Tensor tensor_from(Engine& engine, const std::vector<int64_t>& dims,
                               const std::vector<double>& values)
            {
                Tensor t = engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.data() + dims.size()),
                    sushiblas_dtype<T>());

                const std::vector<T> elements = elements_from<T>(values);
                if (t.num_elements > 0)
                    TestEnvironment::queue()
                        .memcpy(t.data(), elements.data(), elements.size() * sizeof(T))
                        .wait();
                return t;
            }

            /** @brief Returns a dense tensor of dims with every element at K_SENTINEL. */
            template<typename T>
            Tensor sentinel_tensor(Engine& engine, const std::vector<int64_t>& dims)
            {
                return tensor_from<T>(engine, dims,
                                      std::vector<double>(element_count(dims), K_SENTINEL));
            }

            /** @brief Returns the elements of a dense tensor, widened to double. */
            template<typename T>
            std::vector<double> values_of(const Tensor& t)
            {
                const std::vector<T> elements = read_tensor<T>(t);
                std::vector<double> values(elements.size());
                for (std::size_t i = 0; i < elements.size(); ++i)
                    values[i] = widen(elements[i]);
                return values;
            }

            /**
             * @brief Returns a rank-2 view of parent with the given extents and strides.
             * @param offset Elements past parent's first element where the view starts.
             */
            inline Tensor matrix_view(const Tensor& parent, int64_t rows, int64_t cols,
                                      int64_t row_stride, int64_t col_stride, int64_t offset)
            {
                const std::vector<int64_t> dims = {rows, cols};
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                strides[0] = row_stride;
                strides[1] = col_stride;

                return Tensor(parent.storage,
                              SushiRuntime::span<const int64_t>(dims.data(), dims.data() + 2),
                              strides, parent.dtype, parent.storage_offset + offset,
                              parent.layout);
            }

            /**
             * @brief Returns a rank-1 view of parent with count elements, step stride apart.
             * @param offset Elements past parent's first element where the view starts.
             */
            inline Tensor vector_view(const Tensor& parent, int64_t count, int64_t stride,
                                      int64_t offset)
            {
                const std::vector<int64_t> dims = {count};
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                strides[0] = stride;

                return Tensor(parent.storage,
                              SushiRuntime::span<const int64_t>(dims.data(), dims.data() + 1),
                              strides, parent.dtype, parent.storage_offset + offset,
                              parent.layout);
            }

            /** @brief Returns `table[index[r], d]` per row, zeros for an index outside `[0, V)`. */
            inline std::vector<double> host_take_rows(const std::vector<double>& table, int64_t V,
                                                      int64_t D,
                                                      const std::vector<int64_t>& index)
            {
                std::vector<double> rows(index.size() * static_cast<std::size_t>(D), 0.0);
                for (std::size_t r = 0; r < index.size(); ++r)
                    for (int64_t d = 0; d < D; ++d)
                        if (index[r] >= 0 && index[r] < V)
                            rows[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)] =
                                table[static_cast<std::size_t>(index[r] * D + d)];
                return rows;
            }

            /**
             * @brief Returns `beta * table` plus the rows each index selects, added in ascending r.
             * @param beta Exactly zero leaves the table out, as the operation does.
             */
            inline std::vector<double> host_add_rows(const std::vector<double>& rows, int64_t D,
                                                     const std::vector<int64_t>& index,
                                                     const std::vector<double>& table, int64_t V,
                                                     double beta)
            {
                std::vector<double> out(table.size(), 0.0);
                for (std::size_t i = 0; i < out.size(); ++i)
                    if (beta != 0.0)
                        out[i] = beta * table[i];

                for (std::size_t r = 0; r < index.size(); ++r)
                    for (int64_t d = 0; d < D; ++d)
                        if (index[r] >= 0 && index[r] < V)
                            out[static_cast<std::size_t>(index[r] * D + d)] +=
                                rows[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
                return out;
            }

            /** @brief Returns `values[r, index[r]]` per row, zero for an index outside `[0, C)`. */
            inline std::vector<double> host_take_along(const std::vector<double>& values,
                                                       int64_t C,
                                                       const std::vector<int64_t>& index)
            {
                std::vector<double> picked(index.size(), 0.0);
                for (std::size_t r = 0; r < index.size(); ++r)
                    if (index[r] >= 0 && index[r] < C)
                        picked[r] = values[r * static_cast<std::size_t>(C) +
                                           static_cast<std::size_t>(index[r])];
                return picked;
            }

            /** @brief Returns `values[r, c] = (c == index[r]) ? picked[r] : 0`. */
            inline std::vector<double> host_put_along(const std::vector<double>& picked, int64_t C,
                                                      const std::vector<int64_t>& index)
            {
                std::vector<double> values(index.size() * static_cast<std::size_t>(C), 0.0);
                for (std::size_t r = 0; r < index.size(); ++r)
                    for (int64_t c = 0; c < C; ++c)
                        if (c == index[r])
                            values[r * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)] =
                                picked[r];
                return values;
            }

            /**
             * @brief Expects record to throw std::invalid_argument naming operation and operand.
             * @param operand The operand the message names; empty when it names none.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(const char* operation, const char* operand, Record&& record)
            {
                try
                {
                    record();
                    ADD_FAILURE() << operation << " accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find(operation), std::string::npos) << text;
                    if (operand[0] != '\0')
                        EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                            << text;
                }
            }
        } // namespace IndexSupport
    } // namespace Test
} // namespace SushiBLAS
