/**************************************************************************/
/* test_tensor_fill.cpp                                                   */
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
 * @file test_tensor_fill.cpp
 * @brief Tests that fill_tensor converts its values to each dtype and that fill_random repeats.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a rank-one tensor of four elements of dtype.
        Tensor four_of(Core::DataType dtype)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {4};
            return factory.create(SushiRuntime::span<const int64_t>(dims, 1), dtype);
        }

        TEST(Unit_TensorFill, ConvertsTheValuesToTheTensorsDtype)
        {
            const std::vector<float> values{1.5f, -2.0f, 0.25f, 7.0f};
            Tensor doubles = four_of(Core::DataType::Float64);
            Tensor complexes = four_of(Core::DataType::Complex32);
            Tensor integers = four_of(Core::DataType::Int32);

            fill_tensor(doubles, values);
            fill_tensor(complexes, values);
            fill_tensor(integers, values);

            EXPECT_EQ(read_tensor<double>(doubles), (std::vector<double>{1.5, -2.0, 0.25, 7.0}));
            EXPECT_EQ(read_tensor<std::complex<float>>(complexes)[0],
                      std::complex<float>(1.5f, 0.0f));
            EXPECT_EQ(read_tensor<int32_t>(integers), (std::vector<int32_t>{1, -2, 0, 7}));
        }

        TEST(Unit_TensorFill, FillRandomStaysInRangeAndRepeatsFromItsSeed)
        {
            Tensor first = four_of(Core::DataType::Float32);
            Tensor second = four_of(Core::DataType::Float32);

            reset_fill_random_generator();
            fill_random<float>(first, 2.0f, 3.0f);
            reset_fill_random_generator();
            fill_random<float>(second, 2.0f, 3.0f);

            const std::vector<float> values = read_tensor<float>(first);
            EXPECT_EQ(values, read_tensor<float>(second));
            for (float value : values)
            {
                EXPECT_GE(value, 2.0f);
                EXPECT_LT(value, 3.0f);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
