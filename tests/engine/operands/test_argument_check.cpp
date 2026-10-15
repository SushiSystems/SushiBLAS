/**************************************************************************/
/* test_argument_check.cpp                                                */
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
 * @file test_argument_check.cpp
 * @brief Tests the record-time checks on a flag, a scalar, a dtype and a refused transpose.
 * @author Mustafa Garip
 */

#include <complex>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds tensors for the dtype check on the fixture's engine. */
        class Medium_ArgumentCheck : public SushiBLASTest
        {

        };

        TEST_F(Medium_ArgumentCheck, AFlagMustBeAnEnumerator)
        {
            EXPECT_NO_THROW(Dispatch::require_flag(Core::Uplo::Lower, "op", "uplo"));
            EXPECT_THROW(Dispatch::require_flag(Core::Uplo{}, "op", "uplo"),
                         std::invalid_argument);
        }

        TEST_F(Medium_ArgumentCheck, ARealTensorTakesOnlyARealScalar)
        {
            const BLASScalar real(2.0);
            const BLASScalar complex(std::complex<double>(1.0, 1.0));

            EXPECT_NO_THROW(Dispatch::require_real_scalar(real, Core::DataType::Float32, "op",
                                                          "alpha"));
            EXPECT_NO_THROW(Dispatch::require_real_scalar(complex, Core::DataType::Complex64,
                                                          "op", "alpha"));
            EXPECT_THROW(Dispatch::require_real_scalar(complex, Core::DataType::Float64, "op",
                                                       "alpha"),
                         std::invalid_argument);
        }

        TEST_F(Medium_ArgumentCheck, ARealRoutineRefusesAComplexTensor)
        {
            const Tensor real = engine->create_tensor({2});
            const Tensor complex = engine->create_tensor({2}, Core::DataType::Complex32);

            EXPECT_NO_THROW(Dispatch::require_real_dtype(real, "op", "x", "the H routine"));
            EXPECT_THROW(Dispatch::require_real_dtype(complex, "op", "x", "the H routine"),
                         std::invalid_argument);
        }

        TEST_F(Medium_ArgumentCheck, ARefusedTransposeIsNamed)
        {
            EXPECT_NO_THROW(Dispatch::require_transpose_not(Core::Transpose::Trans,
                                                            Core::Transpose::ConjTrans, "op",
                                                            "sibling"));
            EXPECT_THROW(Dispatch::require_transpose_not(Core::Transpose::ConjTrans,
                                                         Core::Transpose::ConjTrans, "op",
                                                         "sibling"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
