/**************************************************************************/
/* test_late_count_hold.cpp                                               */
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
 * @file test_late_count_hold.cpp
 * @brief Tests LateCountHold: the token it makes holds every storage until release().
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Builds a runtime and an engine over it for each test. */
        class Medium_LateCountHold : public ::testing::Test
        {
            protected:
                /// @brief The runtime the hold borrows counts from.
                SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();

                /// @brief The engine the tensors come from.
                Engine engine{runtime};
        };

        /**
         * @brief Tests that a kept token holds the operand and count storages and the borrowed
         *        count's owner, and that release() lets all of them go.
         */
        TEST_F(Medium_LateCountHold, ReleaseLetsGoOfTheStoragesAndKillsTheToken)
        {
            LateCountHold hold(runtime);
            Tensor out = engine.create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine.create_tensor({1}, Core::DataType::Int32,
                                                SushiRuntime::Residency::Host);
            SushiRuntime::LivenessObserver watch;
            {
                OperandSet operands;
                operands.writes(out);
                LateCount late = hold.borrow(operands, count);
                watch = late.owner;
                EXPECT_EQ(late.words.size(), count.storage->size_bytes() / sizeof(std::uint32_t));
                hold.keep(std::move(late.owner));
            }

            EXPECT_EQ(hold.held_count(), std::size_t{1});
            EXPECT_FALSE(watch.expired());
            EXPECT_GT(out.storage->ref_count(), 1);
            EXPECT_GT(count.storage->ref_count(), 1);

            hold.release();
            EXPECT_EQ(hold.held_count(), std::size_t{0});
            EXPECT_TRUE(watch.expired());
            EXPECT_EQ(out.storage->ref_count(), 1);
            EXPECT_EQ(count.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
