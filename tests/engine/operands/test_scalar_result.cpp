/**************************************************************************/
/* test_scalar_result.cpp                                                 */
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
 * @file test_scalar_result.cpp
 * @brief Tests the real dtype of a scalar result and the check on the tensor that receives it.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds result tensors on the fixture's engine. */
        class Medium_ScalarResult : public SushiBLASTest
        {

        };

        TEST_F(Medium_ScalarResult, AComplexDtypeHasTheRealDtypeOfItsParts)
        {
            static_assert(Internal::real_result_dtype(Core::DataType::Complex32) ==
                              Core::DataType::Float32,
                          "COMPLEX32 has FLOAT32 parts.");

            EXPECT_EQ(Internal::real_result_dtype(Core::DataType::Complex64),
                      Core::DataType::Float64);
            EXPECT_EQ(Internal::real_result_dtype(Core::DataType::Half), Core::DataType::Half);
        }

        TEST_F(Medium_ScalarResult, AResultHoldsOneElementOfTheExpectedDtype)
        {
            EXPECT_NO_THROW(Internal::require_scalar_result(engine->create_tensor({1}),
                                                            Core::DataType::Float32, "op"));
            EXPECT_THROW(Internal::require_scalar_result(engine->create_tensor({2}),
                                                         Core::DataType::Float32, "op"),
                         std::invalid_argument);
            EXPECT_THROW(Internal::require_scalar_result(engine->create_tensor({1}),
                                                         Core::DataType::Float64, "op"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
