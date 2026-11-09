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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_read_binding.cpp
 * @brief Tests bind_reads, the bundle that declares the tensors an element functor reads.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
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
        } // namespace

        /** @brief Gives each test the fixture's engine and the text of a refused read. */
        class Medium_ReadBinding : public SushiBLASTest
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

        TEST_F(Medium_ReadBinding, PointersFollowTheElementsInBoundOrder)
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

            EXPECT_EQ(seen_element, 1.5);
            EXPECT_EQ(seen_first, a.data_as<float>());
            EXPECT_EQ(seen_second, b.data_as<std::int64_t>());
        }

        TEST_F(Medium_ReadBinding, AnElementReadTakesTheOperationsType)
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

        TEST_F(Medium_ReadBinding, BoundScalarsComeAfterThePointers)
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

            EXPECT_EQ(seen_element, 3.0f);
            EXPECT_EQ(seen_pointer, a.data_as<float>());
            EXPECT_EQ(seen_scalar, 2.0f);
        }

        TEST_F(Medium_ReadBinding, TheKernelFunctorHoldsNoStorage)
        {
            const auto fn = [](float element, const float*) { return element; };
            using Fn = std::decay_t<decltype(fn)>;

            static_assert(
                std::is_trivially_copyable_v<Dispatch::AddressedOp<Fn, Dispatch::ElementRead>>,
                "A task closure copies an AddressedOp byte by byte.");
            static_assert(
                std::is_trivially_copyable_v<Dispatch::ReadingOp<Fn, float, Dispatch::ElementRead>>,
                "A kernel copies a ReadingOp byte by byte.");

            const std::size_t live_baseline = TestEnvironment::live_allocation_count();

            std::optional<Tensor> a(engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32));
            ASSERT_EQ(TestEnvironment::live_allocation_count(), live_baseline + 1);

            const auto addressed =
                Dispatch::bind_reads(fn, Dispatch::read_as_element(*a)).addressed();
            a.reset();

            static_assert(
                std::is_same_v<decltype(addressed),
                               const Dispatch::AddressedOp<Fn, Dispatch::ElementRead>>,
                "addressed() hands the task closure an AddressedOp, which holds no Tensor.");
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline);
        }

        namespace
        {
            /** @brief Names one read bundle and whether its check accepts it or refuses the read at an index. */
            struct ReadCheckRow
            {
                const char* name;                    /**< The row name a failure shows. */
                void (*require)(Engine&);            /**< Binds the reads and runs the operand check. */
                bool refused;                        /**< Whether the check refuses the bundle. */
                std::size_t refused_index;           /**< The read the refusal names. */
            };

            /** @brief Returns one row per accepted and per refused read. */
            std::vector<ReadCheckRow> read_check_rows()
            {
                return {
                    {"AnElementReadIsAccepted",
                     [](Engine& engine)
                     {
                         const Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as_element(dense))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     false, 0},
                    {"ATypedReadIsAcceptedWhateverTheElementType",
                     [](Engine& engine)
                     {
                         const Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as<float>(dense))
                             .require(Core::DataType::Int64, K_OPERATION);
                     },
                     false, 0},
                    {"AnEmptyReadIsRefused",
                     [](Engine& engine)
                     {
                         const Tensor empty = engine.create_tensor({0}, Core::DataType::Float32);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as_element(empty))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 0},
                    {"AReadOfAnotherDtypeIsRefused",
                     [](Engine& engine)
                     {
                         const Tensor wide = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float64);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as<float>(wide))
                             .require(Core::DataType::Float64, K_OPERATION);
                     },
                     true, 0},
                    {"AnElementReadOfAnotherElementTypeIsRefused",
                     [](Engine& engine)
                     {
                         const Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as_element(dense))
                             .require(Core::DataType::Int32, K_OPERATION);
                     },
                     true, 0},
                    {"AStridedReadIsRefused",
                     [](Engine& engine)
                     {
                         const Tensor strided =
                             engine.create_tensor({K_ELEMENTS, K_ELEMENTS}, Core::DataType::Float32)
                                 .slice(1, 0, 2);
                         Dispatch::bind_reads([](float e, const float*) { return e; }, Dispatch::read_as_element(strided))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 0},
                    {"TheSecondOfTwoReadsIsNamedByItsIndex",
                     [](Engine& engine)
                     {
                         const Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         const Tensor empty = engine.create_tensor({0}, Core::DataType::Float32);
                         Dispatch::bind_reads([](float e, const float*, const float*) { return e; },
                                               Dispatch::read_as_element(dense),
                                               Dispatch::read_as_element(empty))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 1}};
            }
        } // namespace

        /** @brief Runs one row of read_check_rows(). */
        class Medium_ReadBindingCheck : public Medium_ReadBinding,
                                         public ::testing::WithParamInterface<ReadCheckRow>
        {
        };

        TEST_P(Medium_ReadBindingCheck, AReadIsCheckedAsAnOperand)
        {
            const ReadCheckRow& row = GetParam();
            const std::string message = message_of([&] { row.require(*engine); });

            if (row.refused)
                EXPECT_TRUE(names_read(message, row.refused_index)) << message;
            else
                EXPECT_TRUE(message.empty()) << message;
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_ReadBindingCheck, ::testing::ValuesIn(read_check_rows()),
            [](const ::testing::TestParamInfo<ReadCheckRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Medium_ReadBinding, AReadMayNotOverlapAWrittenRegion)
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
        }

        TEST_F(Medium_ReadBinding, AReadDeclaresItsRegion)
        {
            const auto fn = [](float element, const float*) { return element; };
            const Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto bound = Dispatch::bind_reads(fn, Dispatch::read_as_element(a));

            OperandSet reading;
            reading.reads(a);
            ASSERT_EQ(reading.read_regions().size(), std::size_t{1});
            bound.declare_into(reading, K_OPERATION);

            EXPECT_EQ(reading.read_regions().size(), std::size_t{2});
            EXPECT_TRUE(reading.write_regions().empty());
        }

        TEST(Small_ReadBinding, APlainFunctorPassesThroughUntouched)
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
