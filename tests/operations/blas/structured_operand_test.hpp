/**************************************************************************/
/* structured_operand_test.hpp                                            */
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
 * @file structured_operand_test.hpp
 * @brief Declares StructuredOperandTest, the fixture that builds band, packed and scalar operands.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <string>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the two layouts every case runs under.
        inline constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::ColumnMajor,
                                                     Core::Layout::RowMajor};

        /**
         * @brief Builds the tensors the band, packed and scalar operand suites describe.
         */
        class StructuredOperandTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a dense tensor over a shared storage of its own. */
                [[nodiscard]] static Tensor
                make_tensor(std::initializer_list<int64_t> dims,
                            Core::Layout layout = Core::Layout::RowMajor,
                            Core::DataType dtype = Core::DataType::Float32)
                {
                    int64_t elements = 1;
                    for (const int64_t extent : dims)
                        elements *= extent;
                    return Tensor(make_storage(elements, dtype), dims, dtype, 0, layout);
                }

                /** @brief Returns a tensor of a shape that owns no storage. */
                [[nodiscard]] static Tensor
                make_tensor_without_storage(std::initializer_list<int64_t> dims,
                                            Core::Layout layout = Core::Layout::RowMajor)
                {
                    return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, Core::DataType::Float32,
                                  0, layout);
                }

                /**
                 * @brief Returns a rank-1 view of every second element of a storage.
                 * @param elements Elements the view holds.
                 */
                [[nodiscard]] static Tensor make_every_second_element(int64_t elements,
                                                                     Core::Layout layout)
                {
                    const int64_t dims[] = {elements};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(make_storage(2 * elements, Core::DataType::Float32),
                                  SushiRuntime::span<const int64_t>(dims, 1), strides,
                                  Core::DataType::Float32, 0, layout);
                }

                /** @brief Returns the text of the invalid_argument a check throws, or "". */
                template<typename Check>
                [[nodiscard]] static std::string message_of(const Check& check)
                {
                    try
                    {
                        check();
                    }
                    catch (const std::invalid_argument& error)
                    {
                        return error.what();
                    }

                    return std::string();
                }

            private:
                /** @brief Returns a shared storage of a count of elements of a dtype. */
                [[nodiscard]] static SushiRuntime::sushi_ptr<Storage>
                make_storage(int64_t elements, Core::DataType dtype)
                {
                    return SushiRuntime::make_sushi<Storage>(
                        TestEnvironment::runtime(),
                        static_cast<std::size_t>(elements) * Core::bytes_per_element(dtype),
                        SushiRuntime::Residency::Shared);
                }
        };
    } // namespace Test
} // namespace SushiBLAS
