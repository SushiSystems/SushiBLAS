/**************************************************************************/
/* tensor_factory_contract.hpp                                            */
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
 * @file tensor_factory_contract.hpp
 * @brief Declares the checks every TensorFactory adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the checks that define a TensorFactory; an adapter's suite calls each one. */
        struct TensorFactoryContract
        {
            /** @brief The factory reports the layout it was built with. */
            template<typename Subject>
            static void expect_keeps_its_layout(const Subject& subject)
            {
                EXPECT_EQ(subject.factory.layout(), Subject::K_LAYOUT);
            }

            /** @brief A created tensor has the dims, dtype and layout asked for and room for them. */
            template<typename Subject>
            static void expect_create_gives_the_shape_asked_for(const Subject& subject)
            {
                const std::vector<int64_t> dims{2, 3};
                const Tensor tensor = subject.factory.create(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                    Core::DataType::Complex32);

                EXPECT_EQ(tensor.rank, 2);
                EXPECT_EQ(tensor.shape[0], int64_t{2});
                EXPECT_EQ(tensor.shape[1], int64_t{3});
                EXPECT_EQ(tensor.dtype, Core::DataType::Complex32);
                EXPECT_EQ(tensor.layout, Subject::K_LAYOUT);
                EXPECT_FALSE(tensor.shape_only);
                EXPECT_GE(tensor.storage->size_bytes(), std::size_t{48});
            }

            /** @brief The memory of a created tensor starts on a K_STORAGE_ALIGNMENT boundary. */
            template<typename Subject>
            static void expect_create_gives_aligned_memory(const Subject& subject)
            {
                const std::vector<int64_t> dims{5, 7};
                const Tensor tensor = subject.factory.create(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()));

                EXPECT_TRUE(tensor.is_aligned());
            }

            /** @brief A shape with a zero dimension gets no storage. */
            template<typename Subject>
            static void expect_empty_shape_gets_no_storage(const Subject& subject)
            {
                const std::vector<int64_t> dims{4, 0};
                const Tensor tensor = subject.factory.create(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.size()));

                EXPECT_EQ(tensor.num_elements, int64_t{0});
                EXPECT_FALSE(static_cast<bool>(tensor.storage));
            }

            /** @brief A negative dimension is refused. */
            template<typename Subject>
            static void expect_negative_dimension_is_refused(const Subject& subject)
            {
                const std::vector<int64_t> dims{-1};

                EXPECT_THROW((void)subject.factory.create(
                                 SushiRuntime::span<const int64_t>(dims.data(), dims.size())),
                             std::runtime_error);
            }

            /** @brief A device index past the last device is refused. */
            template<typename Subject>
            static void expect_device_index_out_of_range_is_refused(const Subject& subject)
            {
                const std::vector<int64_t> dims{4};

                EXPECT_THROW((void)subject.factory.create(
                                 SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                                 Core::DataType::Float32, SushiRuntime::Residency::Shared,
                                 Subject::device_count()),
                             std::runtime_error);
            }

            /** @brief Two created tensors share no byte. */
            template<typename Subject>
            static void expect_created_tensors_do_not_overlap(const Subject& subject)
            {
                const std::vector<int64_t> dims{8};
                const SushiRuntime::span<const int64_t> span(dims.data(), dims.size());
                const Tensor first = subject.factory.create(span);
                const Tensor second = subject.factory.create(span);

                EXPECT_FALSE(first.region().overlaps(second.region()));
            }
        };
    } // namespace Test
} // namespace SushiBLAS
