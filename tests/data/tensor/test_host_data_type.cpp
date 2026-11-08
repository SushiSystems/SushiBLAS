/**************************************************************************/
/* test_host_data_type.cpp                                                */
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
 * @file test_host_data_type.cpp
 * @brief Tests that HostDataType gives each host element type its DataType.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/host_data_type.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief A host element type and the DataType it maps to.
        template<typename Host, Core::DataType Expected>
        struct HostMapping
        {
            /// @brief The host element type.
            using HostType = Host;
            /// @brief The DataType the host type must map to.
            static constexpr Core::DataType K_EXPECTED = Expected;
        };

        template<typename Mapping>
        class Small_HostDataType : public ::testing::Test
        {
        };

        using Mappings = ::testing::Types<
            HostMapping<float, Core::DataType::Float32>,
            HostMapping<double, Core::DataType::Float64>,
            HostMapping<sycl::half, Core::DataType::Half>,
            HostMapping<std::complex<float>, Core::DataType::Complex32>,
            HostMapping<std::complex<double>, Core::DataType::Complex64>,
            HostMapping<int32_t, Core::DataType::Int32>,
            HostMapping<int64_t, Core::DataType::Int64>,
            HostMapping<uint32_t, Core::DataType::UInt32>>;
        SB_TYPED_TEST_SUITE(Small_HostDataType, Mappings);

        TYPED_TEST(Small_HostDataType, EachHostTypeHasItsDataType)
        {
            EXPECT_EQ(Detail::HostDataType<typename TypeParam::HostType>::K_VALUE,
                      TypeParam::K_EXPECTED);
        }

        TYPED_TEST(Small_HostDataType, TheDataTypeHasTheSizeOfItsHostType)
        {
            EXPECT_EQ(Core::bytes_per_element(Detail::HostDataType<typename TypeParam::HostType>::K_VALUE),
                      sizeof(typename TypeParam::HostType));
        }
    } // namespace Test
} // namespace SushiBLAS
