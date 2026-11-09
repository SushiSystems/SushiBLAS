/**************************************************************************/
/* test_write_binding.cpp                                                 */
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
 * @file test_write_binding.cpp
 * @brief Tests bind_writes, the bundle that declares the tensors an element functor writes.
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
#include <SushiBLAS/functor_map/write_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the operation name every check of this file reports under.
            constexpr const char* K_OPERATION = "write_binding_test";

            /// @brief Holds the element count of every dense tensor of this file.
            constexpr std::int64_t K_ELEMENTS = 4;
        } // namespace

        /** @brief Gives each test the fixture's engine and the text of a refused write. */
        class Medium_WriteBinding : public SushiBLASTest
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

                /** @brief Reports whether message names the operand `write <index>`. */
                [[nodiscard]] static bool names_write(const std::string& message, std::size_t index)
                {
                    const std::string operand = "'write " + std::to_string(index) + "'";
                    return message.find(operand) != std::string::npos;
                }
        };

        TEST_F(Medium_WriteBinding, WritePointersFollowTheReadPointersInBoundOrder)
        {
            const Tensor r = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor first = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor second = engine->create_tensor({K_ELEMENTS}, Core::DataType::Int64);

            double seen_element = 0.0;
            const float* seen_read = nullptr;
            float* seen_first = nullptr;
            std::int64_t* seen_second = nullptr;
            const auto fn = [&](double element, const float* read, float* first_out,
                                std::int64_t* second_out)
            {
                seen_element = element;
                seen_read = read;
                seen_first = first_out;
                seen_second = second_out;
            };

            const auto kernel =
                Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as<float>(r)),
                                      Dispatch::write_as<float>(first),
                                      Dispatch::write_as<std::int64_t>(second))
                    .addressed()
                    .typed(Dispatch::ElementTag<double>{});
            kernel(1.5);

            EXPECT_EQ(seen_element, 1.5);
            EXPECT_EQ(seen_read, r.data_as<float>());
            EXPECT_EQ(seen_first, first.data_as<float>());
            EXPECT_EQ(seen_second, second.data_as<std::int64_t>());
        }

        TEST_F(Medium_WriteBinding, AnElementWriteTakesTheOperationsType)
        {
            Tensor a = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            const auto fn = [](auto element, auto pointer) -> void*
            {
                static_assert(std::is_same_v<decltype(pointer), decltype(element)*>,
                              "An element write arrives as a pointer to the element type.");
                return pointer;
            };

            const auto addressed =
                Dispatch::bind_writes(fn, Dispatch::write_as_element(a)).addressed();
            void* as_float =
                Dispatch::Detail::kernel_op(Dispatch::ElementTag<float>{}, addressed)(1.0f);
            void* as_double =
                Dispatch::Detail::kernel_op(Dispatch::ElementTag<double>{}, addressed)(1.0);

            EXPECT_EQ(as_float, a.data());
            EXPECT_EQ(as_double, a.data());
        }

        TEST_F(Medium_WriteBinding, BoundScalarsComeAfterTheWrites)
        {
            const Tensor r = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor w = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            const float* seen_read = nullptr;
            float* seen_write = nullptr;
            float seen_scalar = 0.0f;
            const auto fn = [&](float, const float* read, float* write, float scalar)
            {
                seen_read = read;
                seen_write = write;
                seen_scalar = scalar;
            };

            const auto kernel =
                Dispatch::bind_writes(
                    Dispatch::bind_reads(Dispatch::bind_scalars(fn, Dispatch::ElementScalar(2.0)),
                                         Dispatch::read_as_element(r)),
                    Dispatch::write_as_element(w))
                    .addressed()
                    .typed(Dispatch::ElementTag<float>{});
            kernel(3.0f);

            EXPECT_EQ(seen_read, r.data_as<float>());
            EXPECT_EQ(seen_write, w.data_as<float>());
            EXPECT_EQ(seen_scalar, 2.0f);
        }

        TEST_F(Medium_WriteBinding, TheKernelFunctorHoldsNoStorage)
        {
            const auto fn = [](float, const float*, float*) {};
            using Fn = std::decay_t<decltype(fn)>;
            using Wrapped = Dispatch::WriteAddressedOp<Fn, Dispatch::ElementWrite>;
            using Task = Dispatch::AddressedOp<Wrapped, Dispatch::ElementRead>;

            static_assert(std::is_trivially_copyable_v<Wrapped>,
                          "A task closure copies a WriteAddressedOp byte by byte.");
            static_assert(std::is_trivially_copyable_v<Task>,
                          "A task closure copies an AddressedOp byte by byte.");
            static_assert(std::is_trivially_copyable_v<
                              Dispatch::WritingOp<Fn, float, Dispatch::ElementWrite>>,
                          "A kernel copies a WritingOp byte by byte.");

            const std::size_t live_baseline = TestEnvironment::live_allocation_count();

            std::optional<Tensor> r(engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32));
            std::optional<Tensor> w(engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32));
            ASSERT_EQ(TestEnvironment::live_allocation_count(), live_baseline + 2);

            const auto addressed =
                Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as_element(*r)),
                                      Dispatch::write_as_element(*w))
                    .addressed();
            r.reset();
            w.reset();

            static_assert(std::is_same_v<decltype(addressed), const Task>,
                          "addressed() hands the task closure an AddressedOp, which holds no Tensor.");
            EXPECT_EQ(TestEnvironment::live_allocation_count(), live_baseline);
        }

        namespace
        {
            /** @brief Names one write bundle and whether its check accepts it or refuses the write at an index. */
            struct WriteCheckRow
            {
                const char* name;                    /**< The row name a failure shows. */
                void (*require)(Engine&);            /**< Binds the writes and runs the operand check. */
                bool refused;                        /**< Whether the check refuses the bundle. */
                std::size_t refused_index;           /**< The write the refusal names. */
            };

            /** @brief Returns one row per accepted and per refused write. */
            std::vector<WriteCheckRow> write_check_rows()
            {
                return {
                    {"AnElementWriteIsAccepted",
                     [](Engine& engine)
                     {
                         Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as_element(dense))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     false, 0},
                    {"ATypedWriteIsAcceptedWhateverTheElementType",
                     [](Engine& engine)
                     {
                         Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as<float>(dense))
                             .require(Core::DataType::Int64, K_OPERATION);
                     },
                     false, 0},
                    {"AnEmptyWriteIsRefused",
                     [](Engine& engine)
                     {
                         Tensor empty = engine.create_tensor({0}, Core::DataType::Float32);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as_element(empty))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 0},
                    {"AWriteOfAnotherDtypeIsRefused",
                     [](Engine& engine)
                     {
                         Tensor wide = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float64);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as<float>(wide))
                             .require(Core::DataType::Float64, K_OPERATION);
                     },
                     true, 0},
                    {"AnElementWriteOfAnotherElementTypeIsRefused",
                     [](Engine& engine)
                     {
                         Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as_element(dense))
                             .require(Core::DataType::Int32, K_OPERATION);
                     },
                     true, 0},
                    {"AStridedWriteIsRefused",
                     [](Engine& engine)
                     {
                         Tensor strided =
                             engine.create_tensor({K_ELEMENTS, K_ELEMENTS}, Core::DataType::Float32)
                                 .slice(1, 0, 2);
                         Dispatch::bind_writes([](float, float*) {}, Dispatch::write_as_element(strided))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 0},
                    {"TheSecondOfTwoWritesIsNamedByItsIndex",
                     [](Engine& engine)
                     {
                         Tensor dense = engine.create_tensor({K_ELEMENTS}, Core::DataType::Float32);
                         Tensor empty = engine.create_tensor({0}, Core::DataType::Float32);
                         Dispatch::bind_writes([](float, float*, float*) {},
                                               Dispatch::write_as_element(dense),
                                               Dispatch::write_as_element(empty))
                             .require(Core::DataType::Float32, K_OPERATION);
                     },
                     true, 1}};
            }
        } // namespace

        /** @brief Runs one row of write_check_rows(). */
        class Medium_WriteBindingCheck : public Medium_WriteBinding,
                                         public ::testing::WithParamInterface<WriteCheckRow>
        {
        };

        TEST_P(Medium_WriteBindingCheck, AWriteIsCheckedAsAnOperand)
        {
            const WriteCheckRow& row = GetParam();
            const std::string message = message_of([&] { row.require(*engine); });

            if (row.refused)
                EXPECT_TRUE(names_write(message, row.refused_index)) << message;
            else
                EXPECT_TRUE(message.empty()) << message;
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_WriteBindingCheck, ::testing::ValuesIn(write_check_rows()),
            [](const ::testing::TestParamInfo<WriteCheckRow>& info)
            { return std::string(info.param.name); });

        TEST_F(Medium_WriteBinding, ATensorReadAndWrittenIsDeclaredOnceAsEach)
        {
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto fn = [](float, const float*, float*) {};
            const auto bound =
                Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as_element(t)),
                                      Dispatch::write_as_element(t));

            OperandSet operands;
            Dispatch::Detail::declare_reads(bound, operands, K_OPERATION);
            Dispatch::Detail::declare_writes(bound, operands, K_OPERATION);

            EXPECT_EQ(operands.read_regions().size(), std::size_t{1});
            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
            EXPECT_EQ(operands.retained().size(), std::size_t{1});
        }

        TEST_F(Medium_WriteBinding, ATensorNamedTwiceAsAWriteIsDeclaredOnce)
        {
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto fn = [](float, float*, float*) {};
            const auto bound = Dispatch::bind_writes(fn, Dispatch::write_as_element(t),
                                                     Dispatch::write_as<float>(t));

            OperandSet operands;
            Dispatch::Detail::declare_writes(bound, operands, K_OPERATION);

            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
            EXPECT_TRUE(operands.read_regions().empty());
        }

        TEST_F(Medium_WriteBinding, AWriteMayNotOverlapAReadThroughAnotherView)
        {
            Tensor whole = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor head = whole.slice(0, 0, 2);
            const auto fn = [](float, const float*, float*) {};
            const auto bound =
                Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as_element(whole)),
                                      Dispatch::write_as_element(head));

            OperandSet operands;
            Dispatch::Detail::declare_reads(bound, operands, K_OPERATION);
            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::Detail::declare_writes(bound, operands, K_OPERATION);
            }), 0));
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST(Small_WriteBinding, APlainFunctorDeclaresNoWrites)
        {
            const auto fn = [](float element) { return element; };
            OperandSet operands;

            Dispatch::Detail::require_declared_writes(fn, Core::DataType::Float32, K_OPERATION);
            Dispatch::Detail::declare_writes(fn, operands, K_OPERATION);

            EXPECT_TRUE(operands.read_regions().empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
