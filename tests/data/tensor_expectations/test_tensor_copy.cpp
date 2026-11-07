/**************************************************************************/
/* test_tensor_copy.cpp                                                   */
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
 * @file test_tensor_copy.cpp
 * @brief Tests that copy_host_to_tensor and copy_tensor_to_host move the bytes of a host tensor by memcpy.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_copy.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that bytes written to a host tensor are the bytes read back, with no queue made. */
        TEST(Small_TensorCopy, BytesWrittenToAHostTensorAreReadBack)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const std::vector<std::int64_t> dims{3};
            Tensor tensor = factory.create(
                SushiRuntime::span<const std::int64_t>(dims.data(), dims.size()),
                Core::DataType::Int32);
            const std::vector<std::int32_t> written{7, -8, 9};
            std::vector<std::int32_t> read(written.size());

            copy_host_to_tensor(tensor, written.data(), written.size() * sizeof(std::int32_t));
            copy_tensor_to_host(read.data(), tensor, read.size() * sizeof(std::int32_t));

            EXPECT_EQ(read, written);
        }
    } // namespace Test
} // namespace SushiBLAS
