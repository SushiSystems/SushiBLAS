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
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_TensorReadback, CopiesTheElementsOut)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {3};
            Tensor tensor = factory.create(SushiRuntime::span<const int64_t>(dims, 1),
                                           Core::DataType::Int64);
            const std::vector<int64_t> values{5, -6, 7};
            TestEnvironment::queue()
                .memcpy(tensor.data(), values.data(), values.size() * sizeof(int64_t))
                .wait();

            EXPECT_EQ(read_tensor<int64_t>(tensor), values);
        }

        TEST(Medium_TensorReadback, RefusesATypeThatIsNotTheDtype)
        {
            const TensorFactory factory(TestEnvironment::runtime());
            const int64_t dims[1] = {3};
            const Tensor tensor = factory.create(SushiRuntime::span<const int64_t>(dims, 1));

            EXPECT_THROW((void)read_tensor<double>(tensor), std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
