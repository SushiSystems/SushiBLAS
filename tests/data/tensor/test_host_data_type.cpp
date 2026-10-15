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

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_HostDataType, EachHostTypeHasItsDataType)
        {
            EXPECT_EQ(Detail::HostDataType<float>::K_VALUE, Core::DataType::Float32);
            EXPECT_EQ(Detail::HostDataType<double>::K_VALUE, Core::DataType::Float64);
            EXPECT_EQ(Detail::HostDataType<sycl::half>::K_VALUE, Core::DataType::Half);
            EXPECT_EQ(Detail::HostDataType<std::complex<float>>::K_VALUE,
                      Core::DataType::Complex32);
            EXPECT_EQ(Detail::HostDataType<std::complex<double>>::K_VALUE,
                      Core::DataType::Complex64);
            EXPECT_EQ(Detail::HostDataType<int32_t>::K_VALUE, Core::DataType::Int32);
            EXPECT_EQ(Detail::HostDataType<int64_t>::K_VALUE, Core::DataType::Int64);
        }

        TEST(Medium_HostDataType, TheDataTypeHasTheSizeOfItsHostType)
        {
            EXPECT_EQ(Core::bytes_per_element(Detail::HostDataType<sycl::half>::K_VALUE),
                      sizeof(sycl::half));
            EXPECT_EQ(Core::bytes_per_element(Detail::HostDataType<std::complex<double>>::K_VALUE),
                      sizeof(std::complex<double>));
            EXPECT_EQ(Core::bytes_per_element(Detail::HostDataType<int64_t>::K_VALUE),
                      sizeof(int64_t));
        }
    } // namespace Test
} // namespace SushiBLAS
