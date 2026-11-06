/**************************************************************************/
/* tensor_readback.hpp                                                    */
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
 * @file tensor_readback.hpp
 * @brief Declares read_tensor, which copies a dense tensor's elements into a host vector.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Copies a dense tensor's elements into a host vector.
         * @tparam T The tensor's element type.
         * @throws std::invalid_argument when T does not map to the tensor's dtype.
         */
        template <typename T>
        std::vector<T> read_tensor(const Tensor& tensor)
        {
            if (sushiblas_dtype<T>() != tensor.dtype)
                throw std::invalid_argument("read_tensor: T does not match the tensor's dtype " +
                                            std::string(Core::to_string(tensor.dtype)) + ".");

            const size_t size = static_cast<size_t>(tensor.num_elements);
            std::vector<T> host(size);
            TestEnvironment::queue()
                .memcpy(host.data(), tensor.data(), size * sizeof(T))
                .wait();
            return host;
        }
    } // namespace Test
} // namespace SushiBLAS
