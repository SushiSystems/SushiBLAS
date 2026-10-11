/**************************************************************************/
/* test_read_binding.cpp                                                  */
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
 * @file test_read_binding.cpp
 * @brief Tests bind_reads, the bundle that declares the tensors an element functor reads.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/SushiRuntime.h>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the operation name every check of this file reports under.
            constexpr const char* K_OPERATION = "read_binding_test";

            /// @brief Holds the element count of every dense tensor of this file.
            constexpr std::int64_t K_ELEMENTS = 4;

            /// @brief Holds the tolerance of a value handed through unchanged: it is exact.
            constexpr double K_EXACT = 0.0;
        } // namespace

        /** @brief Gives each test the fixture's engine and the text of a refused read. */
        class Unit_ReadBinding : public SushiBLASTest
        {
            protected:
                /** @brief Returns the text of the std::invalid_argument a check throws, or "". */
                template<typename Check>
                [[nodiscard]] static std::string message_of(const Check& check)
                {
                    try
                    {
                        check();
                    }
                    catch (const std::invalid_argument& error)
                    {
                        return error.what();
                    }

                    return std::string();
                }

                /** @brief Reports whether message names the operand `read <index>`. */
                [[nodiscard]] static bool names_read(const std::string& message, std::size_t index)
                {
                    const std::string operand = "'read " + std::to_string(index) + "'";
                    return message.find(operand) != std::string::npos;
                }
        };

        TEST_F(Unit_ReadBinding, PointersFollowTheElementsInBoundOrder)
        {
            const Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const Tensor b = engine->create_tensor({K_ELEMENTS}, Core::DataType::Int64);

            double seen_element = 0.0;
            const float* seen_first = nullptr;
            const std::int64_t* seen_second = nullptr;
            const auto fn = [&seen_element, &seen_first, &seen_second](
                                double element, const float* first, const std::int64_t* second)
            {
                seen_element = element;
                seen_first = first;
                seen_second = second;
            };

            const auto kernel = Dispatch::bind_reads(fn, Dispatch::read_as<float>(a),
                                                     Dispatch::read_as<std::int64_t>(b))
                                    .addressed()
                                    .typed(Dispatch::ElementTag<double>{});
            kernel(1.5);

            EXPECT_NEAR(seen_element, 1.5, K_EXACT);
            EXPECT_EQ(seen_first, a.data_as<float>());
            EXPECT_EQ(seen_second, b.data_as<std::int64_t>());
        }

        TEST_F(Unit_ReadBinding, AnElementReadTakesTheOperationsType)
        {
            const Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            const auto fn = [](auto element, auto pointer) -> const void*
            {
                static_assert(std::is_same_v<decltype(pointer), const decltype(element)*>,
                              "An element read arrives as a pointer to the element type.");
                return pointer;
            };

            const auto addressed =
                Dispatch::bind_reads(fn, Dispatch::read_as_element(a)).addressed();
            const void* as_float = addressed.typed(Dispatch::ElementTag<float>{})(1.0f);
            const void* as_double = addressed.typed(Dispatch::ElementTag<double>{})(1.0);

            EXPECT_EQ(as_float, a.data());
            EXPECT_EQ(as_double, a.data());
        }

        TEST_F(Unit_ReadBinding, BoundScalarsComeAfterThePointers)
        {
            const Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            float seen_element = 0.0f;
            const float* seen_pointer = nullptr;
            float seen_scalar = 0.0f;
            const auto fn = [&seen_element, &seen_pointer, &seen_scalar](
                                float element, const float* pointer, float scalar)
            {
                seen_element = element;
                seen_pointer = pointer;
                seen_scalar = scalar;
            };

            const auto kernel =
                Dispatch::bind_reads(Dispatch::bind_scalars(fn, Dispatch::ElementScalar(2.0)),
                                     Dispatch::read_as_element(a))
                    .addressed()
                    .typed(Dispatch::ElementTag<float>{});
            kernel(3.0f);

            EXPECT_NEAR(seen_element, 3.0f, K_EXACT);
            EXPECT_EQ(seen_pointer, a.data_as<float>());
            EXPECT_NEAR(seen_scalar, 2.0f, K_EXACT);
        }

        TEST_F(Unit_ReadBinding, TheKernelFunctorHoldsNoStorage)
        {
            const auto fn = [](float element, const float*) { return element; };
            using Fn = std::decay_t<decltype(fn)>;

            static_assert(
                std::is_trivially_copyable_v<Dispatch::AddressedOp<Fn, Dispatch::ElementRead>>,
                "A task closure copies an AddressedOp byte by byte.");
            static_assert(
                std::is_trivially_copyable_v<Dispatch::ReadingOp<Fn, float, Dispatch::ElementRead>>,
                "A kernel copies a ReadingOp byte by byte.");

            SushiRuntime::Execution::RuntimeContext& context = TestEnvironment::get_context();
            const std::size_t live_baseline = context.memory().live_count();

            std::optional<Tensor> a(engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32));
            ASSERT_EQ(context.memory().live_count(), live_baseline + 1);

            const auto addressed =
                Dispatch::bind_reads(fn, Dispatch::read_as_element(*a)).addressed();
            a.reset();

            static_assert(
                std::is_same_v<decltype(addressed),
                               const Dispatch::AddressedOp<Fn, Dispatch::ElementRead>>,
                "addressed() hands the task closure an AddressedOp, which holds no Tensor.");
            EXPECT_EQ(context.memory().live_count(), live_baseline);
        }

        TEST_F(Unit_ReadBinding, AReadIsCheckedAsAnOperand)
        {
            const auto fn = [](float element, const float*) { return element; };
            const auto fn_of_two = [](float element, const float*, const float*)
            {
                return element;
            };

            const Tensor dense = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const Tensor empty = engine->create_tensor({0}, Core::DataType::Float32);
            const Tensor wide = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float64);
            const Tensor strided =
                engine->create_tensor({K_ELEMENTS, K_ELEMENTS}, Core::DataType::Float32)
                    .slice(1, 0, 2);

            EXPECT_NO_THROW(Dispatch::bind_reads(fn, Dispatch::read_as_element(dense))
                                .require(Core::DataType::Float32, K_OPERATION));
            EXPECT_NO_THROW(Dispatch::bind_reads(fn, Dispatch::read_as<float>(dense))
                                .require(Core::DataType::Int64, K_OPERATION));

            EXPECT_TRUE(names_read(message_of([&]()
            {
                Dispatch::bind_reads(fn, Dispatch::read_as_element(empty))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_read(message_of([&]()
            {
                Dispatch::bind_reads(fn, Dispatch::read_as<float>(wide))
                    .require(Core::DataType::Float64, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_read(message_of([&]()
            {
                Dispatch::bind_reads(fn, Dispatch::read_as_element(dense))
                    .require(Core::DataType::Int32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_read(message_of([&]()
            {
                Dispatch::bind_reads(fn, Dispatch::read_as_element(strided))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_read(message_of([&]()
            {
                Dispatch::bind_reads(fn_of_two, Dispatch::read_as_element(dense),
                                     Dispatch::read_as_element(empty))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 1));
        }

        TEST_F(Unit_ReadBinding, AReadMayNotOverlapAWrittenRegion)
        {
            const auto fn = [](float element, const float*) { return element; };
            const Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto bound = Dispatch::bind_reads(fn, Dispatch::read_as_element(a));

            OperandSet writing;
            writing.writes(a);
            EXPECT_TRUE(names_read(message_of([&]()
            {
                bound.declare_into(writing, K_OPERATION);
            }), 0));
            EXPECT_TRUE(writing.read_regions().empty());

            OperandSet reading;
            reading.reads(a);
            ASSERT_EQ(reading.read_regions().size(), std::size_t{1});
            bound.declare_into(reading, K_OPERATION);

            EXPECT_EQ(reading.read_regions().size(), std::size_t{2});
            EXPECT_TRUE(reading.write_regions().empty());
        }

        TEST_F(Unit_ReadBinding, APlainFunctorPassesThroughUntouched)
        {
            const auto fn = [](float element) { return element; };
            OperandSet operands;

            Dispatch::Detail::require_declared_reads(fn, Core::DataType::Float32, K_OPERATION);
            Dispatch::Detail::declare_reads(fn, operands, K_OPERATION);

            EXPECT_EQ(&Dispatch::Detail::task_functor(fn), &fn);
            EXPECT_EQ(&Dispatch::Detail::kernel_op(Dispatch::ElementTag<float>{}, fn), &fn);
            EXPECT_TRUE(operands.read_regions().empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
