/**************************************************************************/
/* tensor_factory.cpp                                                     */
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
 * @file tensor_factory.cpp
 * @brief Defines the byte count of a dense tensor, which TensorFactory sizes its allocations by.
 * @author Mustafa Garip
 */

#include <SushiBLAS/tensor/tensor_factory.hpp>

namespace SushiBLAS
{
    std::size_t TensorFactory::allocation_bytes(SushiRuntime::span<const int64_t> dims,
                                                Core::DataType dtype)
    {
        SB_THROW_IF(dims.size() > Core::K_MAX_TENSOR_RANK, "Rank {} exceeds limit of {}",
                    dims.size(), Core::K_MAX_TENSOR_RANK);

        int64_t elements = 1;
        for (std::size_t axis = 0; axis < dims.size(); ++axis)
        {
            SB_THROW_IF(dims[axis] < 0, "Dimension {} is negative ({})", axis, dims[axis]);
            elements = Detail::checked_multiply(elements, dims[axis], "Element count");
        }

        const int64_t bpe = static_cast<int64_t>(Core::bytes_per_element(dtype));
        return static_cast<std::size_t>(Detail::checked_multiply(elements, bpe, "Byte size"));
    }
} // namespace SushiBLAS
