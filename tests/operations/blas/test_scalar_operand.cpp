/**************************************************************************/
/* test_scalar_operand.cpp                                                */
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
 * @file test_scalar_operand.cpp
 * @brief Tests the record-time checks on a one-element scalar operand.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas/scalar_operand.hpp>
#include "structured_operand_test.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;

            /// @brief Names the suite of the scalar operand cases.
            using Medium_ScalarOperand = StructuredOperandTest;
        } // namespace

        /**
         * @brief Tests that one stored element is a scalar whatever the rank of its tensor.
         */
        TEST_F(Medium_ScalarOperand, OneElementOfAnyRankIsAccepted)
        {
            const Tensor rank0 = make_tensor(std::initializer_list<int64_t>{});
            ASSERT_EQ(0, rank0.rank);
            ASSERT_EQ(1, rank0.num_elements);
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank0, DataType::Float32, "op", "alpha"));

            const Tensor rank1 = make_tensor({1});
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank1, DataType::Float32, "op", "alpha"));

            const Tensor rank2 = make_tensor({1, 1});
            EXPECT_NO_THROW(
                Dispatch::require_scalar_operand(rank2, DataType::Float32, "op", "alpha"));
        }

        /**
         * @brief Tests that a tensor of another element count is no scalar.
         */
        TEST_F(Medium_ScalarOperand, AnotherElementCountIsRefused)
        {
            const Tensor two = make_tensor({2});
            EXPECT_THROW(Dispatch::require_scalar_operand(two, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);

            const Tensor none = make_tensor_without_storage({0});
            EXPECT_THROW(Dispatch::require_scalar_operand(none, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that a scalar operand must own storage.
         */
        TEST_F(Medium_ScalarOperand, AMissingStorageIsRefused)
        {
            const Tensor t = make_tensor_without_storage({1});
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float32, "op", "alpha"),
                         std::invalid_argument);
        }

        /**
         * @brief Tests that a scalar operand must hold the dtype the call expects.
         */
        TEST_F(Medium_ScalarOperand, AnotherDtypeIsRefused)
        {
            const Tensor t = make_tensor({1});
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Float64, "op", "alpha"),
                         std::invalid_argument);
            EXPECT_THROW(Dispatch::require_scalar_operand(t, DataType::Int32, "op", "alpha"),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
