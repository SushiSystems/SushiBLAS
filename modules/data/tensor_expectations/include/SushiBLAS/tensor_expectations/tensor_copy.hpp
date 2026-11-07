/**************************************************************************/
/* tensor_copy.hpp                                                        */
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
 * @file tensor_copy.hpp
 * @brief Declares copy_tensor_to_host and copy_host_to_tensor, the byte copies of a dense tensor.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <cstring>
#include <cstddef>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Copies bytes from a tensor's data into host memory and returns with the copy done.
         *
         * A tensor in host storage is copied by memcpy and reaches no queue; any other tensor is
         * copied through TestEnvironment::queue().
         * @param bytes Bytes to copy; at most the tensor's dense byte size.
         */
        inline void copy_tensor_to_host(void* destination, const Tensor& tensor, std::size_t bytes)
        {
            if (tensor.storage.get() != nullptr && tensor.storage->is_host())
                std::memcpy(destination, tensor.data(), bytes);
            else
                TestEnvironment::queue().memcpy(destination, tensor.data(), bytes).wait();
        }

        /**
         * @brief Copies bytes from host memory into a tensor's data and returns with the copy done.
         *
         * A tensor in host storage is written by memcpy and reaches no queue; any other tensor is
         * written through TestEnvironment::queue().
         * @param bytes Bytes to copy; at most the tensor's dense byte size.
         */
        inline void copy_host_to_tensor(Tensor& tensor, const void* source, std::size_t bytes)
        {
            if (tensor.storage.get() != nullptr && tensor.storage->is_host())
                std::memcpy(tensor.data(), source, bytes);
            else
                TestEnvironment::queue().memcpy(tensor.data(), source, bytes).wait();
        }
    } // namespace Test
} // namespace SushiBLAS
