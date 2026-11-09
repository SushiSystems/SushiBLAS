/**************************************************************************/
/* host_operands.hpp                                            */
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
 * @file host_operands.hpp
 * @brief Declares the free functions that build host-backed and storage-less operands for the band, packed and scalar operand suites.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the two layouts every case runs under.
        inline constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::ColumnMajor,
                                                     Core::Layout::RowMajor};

        /**
         * @brief Returns a dense tensor over host storage of its own, which needs no runtime.
         * @param dims The extents; none is zero.
         */
        [[nodiscard]] inline Tensor host_matrix(std::initializer_list<int64_t> dims,
                                                Core::Layout layout = Core::Layout::RowMajor,
                                                Core::DataType dtype = Core::DataType::Float32)
        {
            int64_t elements = 1;
            for (const int64_t extent : dims)
                elements *= extent;
            return Tensor(make_host_storage(static_cast<std::size_t>(elements) *
                                            Core::bytes_per_element(dtype)),
                          dims, dtype, 0, layout);
        }

        /**
         * @brief Returns a rank-1 view of every second element of a host storage.
         * @param elements Elements the view holds.
         */
        [[nodiscard]] inline Tensor host_every_second_element(int64_t elements, Core::Layout layout)
        {
            const int64_t dims[] = {elements};
            std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
            strides[0] = 2;
            return Tensor(make_host_storage(static_cast<std::size_t>(2 * elements) * sizeof(float)),
                          SushiRuntime::span<const int64_t>(dims, 1), strides, Core::DataType::Float32, 0,
                          layout);
        }

        /** @brief Returns a tensor of a shape that owns no storage, which needs no runtime. */
        [[nodiscard]] inline Tensor detached_matrix(std::initializer_list<int64_t> dims,
                                                    Core::Layout layout = Core::Layout::RowMajor)
        {
            return Tensor(SushiRuntime::sushi_ptr<Storage>(), dims, Core::DataType::Float32, 0, layout);
        }
    } // namespace Test
} // namespace SushiBLAS
