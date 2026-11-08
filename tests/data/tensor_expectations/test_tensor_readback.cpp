/**************************************************************************/
/* test_tensor_readback.cpp                                               */
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
 * @file test_tensor_readback.cpp
 * @brief Tests that read_tensor copies a tensor's elements out and refuses another element type.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tensor_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Reads tensors a subject's factory made on the host. */
        template<typename Subject>
        class Small_TensorReadback : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Reads tensors a subject's factory made on the shared runtime. */
        template<typename Subject>
        class Medium_TensorReadback : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSubjects = ::testing::Types<HostTensorSubject>;
        using RuntimeSubjects = ::testing::Types<RuntimeTensorSubject>;
        SB_TYPED_TEST_SUITE(Small_TensorReadback, HostSubjects);
        SB_TYPED_TEST_SUITE(Medium_TensorReadback, RuntimeSubjects);

        /// @brief Returns a three-element INT64 tensor of the subject's factory holding 5, -6, 7.
        template<typename Subject>
        static Tensor filled_tensor(const Subject& subject)
        {
            const int64_t dims[1] = {3};
            Tensor tensor = subject.factory.create(SushiRuntime::span<const int64_t>(dims, 1),
                                                   Core::DataType::Int64);
            fill_tensor(tensor, {5.0f, -6.0f, 7.0f});
            return tensor;
        }

        TYPED_TEST(Small_TensorReadback, CopiesTheElementsOut)
        {
            EXPECT_EQ(read_tensor<int64_t>(filled_tensor(this->subject)),
                      (std::vector<int64_t>{5, -6, 7}));
        }

        TYPED_TEST(Medium_TensorReadback, CopiesTheElementsOut)
        {
            EXPECT_EQ(read_tensor<int64_t>(filled_tensor(this->subject)),
                      (std::vector<int64_t>{5, -6, 7}));
        }

        TEST(Small_TensorReadbackRefusal, ATypeThatIsNotTheDtype)
        {
            const std::vector<int64_t> dims{3};
            const Tensor tensor = Tensor::make_shape_only(SushiRuntime::span<const int64_t>(
                dims.data(), dims.size()));

            EXPECT_THROW((void)read_tensor<double>(tensor), std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
