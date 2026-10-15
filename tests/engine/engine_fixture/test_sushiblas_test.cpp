/**************************************************************************/
/* test_sushiblas_test.cpp                                                */
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
 * @file test_sushiblas_test.cpp
 * @brief Tests that SushiBLASTest gives each test a row-major engine and initialised tensors.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the fixture under its own suite name. */
        class Unit_SushiBLASTest : public SushiBLASTest
        {

        };

        TEST_F(Unit_SushiBLASTest, GivesARowMajorEngine)
        {
            ASSERT_TRUE(engine);
            EXPECT_EQ(engine->layout(), Core::Layout::RowMajor);
        }

        TEST_F(Unit_SushiBLASTest, ReinitRebuildsTheEngineInAnotherLayout)
        {
            reinit_engine(Core::Layout::ColumnMajor);

            EXPECT_EQ(engine->layout(), Core::Layout::ColumnMajor);
        }

        TEST_F(Unit_SushiBLASTest, AnInitialisedTensorHoldsTheHostValues)
        {
            const std::vector<int32_t> values{4, -5, 6};
            const Tensor tensor = create_initialized_tensor<int32_t>({3}, values);

            EXPECT_EQ(tensor.dtype, Core::DataType::Int32);
            EXPECT_EQ(read_tensor<int32_t>(tensor), values);
        }
    } // namespace Test
} // namespace SushiBLAS
