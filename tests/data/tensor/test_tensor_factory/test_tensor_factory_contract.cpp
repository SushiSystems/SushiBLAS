/**************************************************************************/
/* test_tensor_factory_contract.cpp                                       */
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
 * @file test_tensor_factory_contract.cpp
 * @brief Tests that the host and runtime adapters of TensorFactory keep the TensorFactory
 *        contract, and that a host factory has no runtime.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include "support/tensor_factory_contract.hpp"
#include "support/tensor_subjects.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the TensorFactory contract against a subject on the host. */
        template<typename Subject>
        class Small_TensorFactoryContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        /** @brief Runs the TensorFactory contract against a subject on the shared runtime. */
        template<typename Subject>
        class Medium_TensorFactoryContract : public ::testing::Test
        {
            protected:
                /// @brief The adapter under test.
                Subject subject;
        };

        using HostSubjects = ::testing::Types<HostTensorSubject>;
        using RuntimeSubjects = ::testing::Types<RuntimeTensorSubject>;
        SB_TYPED_TEST_SUITE(Small_TensorFactoryContract, HostSubjects);
        SB_TYPED_TEST_SUITE(Medium_TensorFactoryContract, RuntimeSubjects);

        TYPED_TEST(Small_TensorFactoryContract, ItKeepsItsLayout)
        {
            TensorFactoryContract::expect_keeps_its_layout(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, CreateGivesTheShapeAskedFor)
        {
            TensorFactoryContract::expect_create_gives_the_shape_asked_for(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, CreateGivesAlignedMemory)
        {
            TensorFactoryContract::expect_create_gives_aligned_memory(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, AnEmptyShapeGetsNoStorage)
        {
            TensorFactoryContract::expect_empty_shape_gets_no_storage(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, ANegativeDimensionIsRefused)
        {
            TensorFactoryContract::expect_negative_dimension_is_refused(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, ADeviceIndexOutOfRangeIsRefused)
        {
            TensorFactoryContract::expect_device_index_out_of_range_is_refused(this->subject);
        }

        TYPED_TEST(Small_TensorFactoryContract, CreatedTensorsDoNotOverlap)
        {
            TensorFactoryContract::expect_created_tensors_do_not_overlap(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, ItKeepsItsLayout)
        {
            TensorFactoryContract::expect_keeps_its_layout(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, CreateGivesTheShapeAskedFor)
        {
            TensorFactoryContract::expect_create_gives_the_shape_asked_for(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, CreateGivesAlignedMemory)
        {
            TensorFactoryContract::expect_create_gives_aligned_memory(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, AnEmptyShapeGetsNoStorage)
        {
            TensorFactoryContract::expect_empty_shape_gets_no_storage(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, ANegativeDimensionIsRefused)
        {
            TensorFactoryContract::expect_negative_dimension_is_refused(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, ADeviceIndexOutOfRangeIsRefused)
        {
            TensorFactoryContract::expect_device_index_out_of_range_is_refused(this->subject);
        }

        TYPED_TEST(Medium_TensorFactoryContract, CreatedTensorsDoNotOverlap)
        {
            TensorFactoryContract::expect_created_tensors_do_not_overlap(this->subject);
        }

        TEST(Small_HostTensorFactory, ItHasNoRuntime)
        {
            const TensorFactory factory(TensorFactory::Host{});

            EXPECT_THROW((void)factory.runtime(), UnsupportedExecution);
        }

        TEST(Small_HostTensorFactory, ItMakesHostStorageWhateverResidencyIsAsked)
        {
            const TensorFactory factory(TensorFactory::Host{});
            const std::int64_t dims[] = {4};

            const Tensor tensor = factory.create(SushiRuntime::span<const std::int64_t>(dims, 1),
                                                 Core::DataType::Float32,
                                                 SushiRuntime::Residency::Device);

            EXPECT_EQ(tensor.storage->block().queue(), nullptr);
            EXPECT_EQ(tensor.storage->residency(), SushiRuntime::Residency::Host);
        }

        TEST(Medium_TensorFactory, ARuntimeFactoryKeepsItsRuntime)
        {
            const TensorFactory factory(TestEnvironment::runtime());

            EXPECT_EQ(&factory.runtime(), &TestEnvironment::runtime());
        }
    } // namespace Test
} // namespace SushiBLAS
