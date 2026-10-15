/**************************************************************************/
/* test_dispatch.cpp                                                      */
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
 * @file test_dispatch.cpp
 * @brief Tests the logic predicates and a unary and a binary logic task in both execution modes.
 * @author Mustafa Garip
 */

#include <complex>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/logic/dispatch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives one where an element is not zero and zero elsewhere. */
        struct NotZeroIndicator
        {
            /** @brief Returns the indicator of one element. */
            template<typename T>
            T operator()(T x) const
            {
                return Dispatch::logic_not_zero(x) ? T(1) : T(0);
            }
        };

        /** @brief Gives one where two elements are equal and zero elsewhere. */
        struct EqualIndicator
        {
            /** @brief Returns the indicator of one pair of elements. */
            template<typename T>
            T operator()(T x, T y) const
            {
                return Dispatch::logic_equal(x, y) ? T(1) : T(0);
            }
        };

        /** @brief Records logic tasks on the fixture's engine. */
        class Medium_LogicDispatch : public SushiBLASTest
        {

        };

        TEST_F(Medium_LogicDispatch, ThePredicatesCompareRealAndComplexValues)
        {
            EXPECT_TRUE(Dispatch::logic_not_zero(std::complex<float>(0.0f, 1.0f)));
            EXPECT_FALSE(Dispatch::logic_not_zero(0.0));
            EXPECT_TRUE(Dispatch::logic_equal(std::complex<double>(1, 2),
                                              std::complex<double>(1, 2)));
            EXPECT_TRUE(Dispatch::logic_less(1, 2));
            EXPECT_TRUE(Dispatch::logic_greater_equal(2, 2));
            EXPECT_FALSE(Dispatch::logic_less_equal(3, 2));
            EXPECT_TRUE(Dispatch::logic_greater(3, 2));
        }

        TEST_F(Medium_LogicDispatch, UnaryAndBinaryTasksWriteOneOrZeroInBothModes)
        {
            const Tensor a = create_initialized_tensor<float>({4}, {0, 2, -1, 2});
            const Tensor b = create_initialized_tensor<float>({4}, {0, 1, -1, 3});

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                Tensor not_zero = create_initialized_tensor<float>({4}, {9, 9, 9, 9});
                Tensor equal = create_initialized_tensor<float>({4}, {9, 9, 9, 9});
                engine->recorder().set_mode(mode);

                Dispatch::execute_logic_unary(engine->recorder(), a, not_zero, "not_zero",
                                              SushiRuntime::OperationID(), NotZeroIndicator{});
                Dispatch::execute_logic_binary<Dispatch::Domain::All>(
                    engine->recorder(), a, b, equal, "equal", SushiRuntime::OperationID(),
                    EqualIndicator{});
                ASSERT_TRUE(engine->execute().has_value());

                EXPECT_EQ(read_tensor<float>(not_zero), (std::vector<float>{0, 1, 1, 1}));
                EXPECT_EQ(read_tensor<float>(equal), (std::vector<float>{1, 0, 1, 0}));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
