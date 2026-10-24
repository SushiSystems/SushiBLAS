/**************************************************************************/
/* test_update_binding.cpp                                                */
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
 * @file test_update_binding.cpp
 * @brief Tests bind_updates, the bundle that declares the tensors an element functor reads and writes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/write_binding.hpp>
#include <SushiBLAS/functor_map/update_binding.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the operation name every check of this file reports under.
            constexpr const char* K_OPERATION = "update_binding_test";

            /// @brief Holds the element count of every dense tensor of this file.
            constexpr std::int64_t K_ELEMENTS = 4;

            /** @brief Declares a bundle's reads, updates and writes in the order a seam does. */
            template<typename Bound>
            void declare_all(const Bound& bound, OperandSet& operands)
            {
                Dispatch::Detail::declare_reads(bound, operands, K_OPERATION);
                Dispatch::Detail::declare_updates(bound, operands, K_OPERATION);
                Dispatch::Detail::declare_writes(bound, operands, K_OPERATION);
            }
        } // namespace

        /** @brief Gives each test the fixture's engine. */
        class Medium_UpdateBinding : public SushiBLASTest
        {

        };

        TEST_F(Medium_UpdateBinding, UpdatePointersSitBetweenTheReadAndTheWritePointers)
        {
            const Tensor r = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor w = engine->create_tensor({K_ELEMENTS}, Core::DataType::Int64);

            const float* seen_read = nullptr;
            float* seen_update = nullptr;
            std::int64_t* seen_write = nullptr;
            const auto fn = [&](std::size_t, const float* read, float* update, std::int64_t* write)
            {
                seen_read = read;
                seen_update = update;
                seen_write = write;
            };

            const auto bound = Dispatch::bind_writes(
                Dispatch::bind_updates(Dispatch::bind_reads(fn, Dispatch::read_as<float>(r)),
                                       Dispatch::update_as_element(u)),
                Dispatch::write_as<std::int64_t>(w));
            const auto kernel = Dispatch::Detail::kernel_op(Dispatch::ElementTag<float>{},
                                                            Dispatch::Detail::task_functor(bound));
            kernel(std::size_t{0});

            EXPECT_EQ(seen_read, r.data_as<float>());
            EXPECT_EQ(seen_update, u.data_as<float>());
            EXPECT_EQ(seen_write, w.data_as<std::int64_t>());
            EXPECT_EQ(Dispatch::Detail::element_dtype_of(bound), Core::DataType::Float32);
        }

        TEST_F(Medium_UpdateBinding, AnUpdateIsDeclaredAsOneReadAndOneWrite)
        {
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto bound = Dispatch::bind_updates([](std::size_t, float*) {},
                                                      Dispatch::update_as<float>(u));

            OperandSet operands;
            declare_all(bound, operands);

            ASSERT_EQ(operands.read_regions().size(), std::size_t{1});
            ASSERT_EQ(operands.write_regions().size(), std::size_t{1});
            EXPECT_TRUE(operands.read_regions()[0].overlaps(u.region()));
            EXPECT_TRUE(operands.write_regions()[0].overlaps(u.region()));
            EXPECT_EQ(operands.retained().size(), std::size_t{1});
        }

        TEST_F(Medium_UpdateBinding, AnUpdateThatIsAlsoReadIsRefused)
        {
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto bound = Dispatch::bind_updates(
                Dispatch::bind_reads([](std::size_t, const float*, float*) {},
                                     Dispatch::read_as<float>(u)),
                Dispatch::update_as<float>(u));

            OperandSet operands;
            EXPECT_THROW(declare_all(bound, operands), std::invalid_argument);
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Medium_UpdateBinding, AWriteThatOverlapsAnUpdateIsRefused)
        {
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor head = u.slice(0, 0, 2);
            const auto bound = Dispatch::bind_writes(
                Dispatch::bind_updates([](std::size_t, float*, float*) {},
                                       Dispatch::update_as<float>(u)),
                Dispatch::write_as<float>(head));

            OperandSet operands;
            EXPECT_THROW(declare_all(bound, operands), std::invalid_argument);
            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
        }

        TEST_F(Medium_UpdateBinding, TwoOverlappingUpdatesAreRefused)
        {
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto bound = Dispatch::bind_updates([](std::size_t, float*, float*) {},
                                                      Dispatch::update_as<float>(u),
                                                      Dispatch::update_as<float>(u));

            OperandSet operands;
            EXPECT_THROW(declare_all(bound, operands), std::invalid_argument);
            EXPECT_TRUE(operands.read_regions().empty());
        }

        TEST_F(Medium_UpdateBinding, AnUpdateOfAnotherDtypeIsRefused)
        {
            Tensor u = engine->create_tensor({K_ELEMENTS}, Core::DataType::Int32);
            const auto bound = Dispatch::bind_updates([](std::size_t, float*) {},
                                                      Dispatch::update_as<float>(u));

            EXPECT_THROW(Dispatch::Detail::require_declared_updates(bound, Core::DataType::Float32,
                                                                    K_OPERATION),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
