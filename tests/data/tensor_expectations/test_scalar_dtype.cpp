/**************************************************************************/
/* test_scalar_dtype.cpp                                                  */
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
 * @file test_scalar_dtype.cpp
 * @brief Tests that sushiblas_dtype names the DataType of each element type a test uses.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_ScalarDtype, NamesEveryElementType)
        {
            EXPECT_EQ(sushiblas_dtype<sycl::half>(), Core::DataType::Half);
            EXPECT_EQ(sushiblas_dtype<float>(), Core::DataType::Float32);
            EXPECT_EQ(sushiblas_dtype<double>(), Core::DataType::Float64);
            EXPECT_EQ(sushiblas_dtype<std::complex<float>>(), Core::DataType::Complex32);
            EXPECT_EQ(sushiblas_dtype<std::complex<double>>(), Core::DataType::Complex64);
            EXPECT_EQ(sushiblas_dtype<int32_t>(), Core::DataType::Int32);
            EXPECT_EQ(sushiblas_dtype<int64_t>(), Core::DataType::Int64);
        }
    } // namespace Test
} // namespace SushiBLAS
