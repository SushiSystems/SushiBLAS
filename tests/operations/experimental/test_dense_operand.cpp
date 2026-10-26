/**************************************************************************/
/* test_dense_operand.cpp                                                 */
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
 * @file test_dense_operand.cpp
 * @brief Tests the record-time operand checks the transforms and dense solvers share.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/dense_operand.hpp>

#include <string>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds the tensors the checks read. */
        class Medium_DenseOperand : public SushiBLASTest
        {

        };

        TEST_F(Medium_DenseOperand, AcceptsADenseTensorOfTheStatedCount)
        {
            const Tensor t = engine->create_tensor({2, 3}, Core::DataType::Float32);

            EXPECT_NO_THROW(Internal::require_dense_elements(t, 6, "op", "t"));
        }

        TEST_F(Medium_DenseOperand, RefusesAnotherCountNamingTheOperand)
        {
            const Tensor t = engine->create_tensor({2, 3}, Core::DataType::Float32);

            try
            {
                Internal::require_dense_elements(t, 5, "op", "t");
                ADD_FAILURE() << "a count of 5 was accepted for 6 elements";
            }
            catch (const std::invalid_argument& error)
            {
                EXPECT_NE(std::string(error.what()).find("'t'"), std::string::npos)
                    << error.what();
            }
        }

        TEST_F(Medium_DenseOperand, AdmitsOnlyFloatAndDoubleAsReal)
        {
            const Tensor single = engine->create_tensor({1}, Core::DataType::Float32);
            const Tensor twin = engine->create_tensor({1}, Core::DataType::Float64);
            const Tensor half = engine->create_tensor({1}, Core::DataType::Half);

            EXPECT_NO_THROW(Internal::require_float_dtype(single, "op", "t"));
            EXPECT_NO_THROW(Internal::require_float_dtype(twin, "op", "t"));
            EXPECT_THROW(Internal::require_float_dtype(half, "op", "t"), std::invalid_argument);
        }

        TEST_F(Medium_DenseOperand, AdmitsOnlyTheComplexDtypesAsComplex)
        {
            const Tensor complex = engine->create_tensor({1}, Core::DataType::Complex64);
            const Tensor real = engine->create_tensor({1}, Core::DataType::Float64);

            EXPECT_NO_THROW(Internal::require_complex_dtype(complex, "op", "t"));
            EXPECT_THROW(Internal::require_complex_dtype(real, "op", "t"), std::invalid_argument);
        }

        TEST(Small_DenseOperand, MapsAComplexDtypeToTheDtypeOfItsParts)
        {
            EXPECT_EQ(Internal::scalar_dtype_of(Core::DataType::Complex32),
                      Core::DataType::Float32);
            EXPECT_EQ(Internal::scalar_dtype_of(Core::DataType::Complex64),
                      Core::DataType::Float64);
            EXPECT_EQ(Internal::scalar_dtype_of(Core::DataType::Float32), Core::DataType::Float32);
        }
    } // namespace Test
} // namespace SushiBLAS
