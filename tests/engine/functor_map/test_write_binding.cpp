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
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/write_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
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

            /// @brief Holds the tolerance of a value handed through unchanged: it is exact.
            constexpr double K_EXACT = 0.0;

            /**
             * @brief Records one task the way a per-index seam does: checks and declares the
             *        reads of the bundle, then its writes, and records a body that does no work.
             * @param bound A bind_writes bundle, or a bare functor.
             */
            template<typename Bound>
            TaskHandle record_task(TaskRecorder& recorder, const Bound& bound, const char* name)
            {
                Dispatch::Detail::require_declared_reads(bound, Core::DataType::Float32, name);
                Dispatch::Detail::require_declared_writes(bound, Core::DataType::Float32, name);

                OperandSet operands;
                Dispatch::Detail::declare_reads(bound, operands, name);
                Dispatch::Detail::declare_writes(bound, operands, name);

                TaskDescription description;
                description.name = name;
                return recorder.record_pair(
                    description, operands, K_OPERATION,
                    [](const KernelPairDispatcher&, sycl::queue&, const std::vector<sycl::event>&)
                    {
                        return sycl::event{};
                    });
            }
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

            EXPECT_NEAR(seen_element, 1.5, K_EXACT);
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
            EXPECT_NEAR(seen_scalar, 2.0f, K_EXACT);
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

        TEST_F(Medium_WriteBinding, AWriteIsCheckedAsAnOperand)
        {
            const auto fn = [](float, float*) {};
            const auto fn_of_two = [](float, float*, float*) {};

            Tensor dense = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor empty = engine->create_tensor({0}, Core::DataType::Float32);
            Tensor wide = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float64);
            Tensor strided = engine->create_tensor({K_ELEMENTS, K_ELEMENTS}, Core::DataType::Float32)
                                 .slice(1, 0, 2);

            EXPECT_NO_THROW(Dispatch::bind_writes(fn, Dispatch::write_as_element(dense))
                                .require(Core::DataType::Float32, K_OPERATION));
            EXPECT_NO_THROW(Dispatch::bind_writes(fn, Dispatch::write_as<float>(dense))
                                .require(Core::DataType::Int64, K_OPERATION));

            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::bind_writes(fn, Dispatch::write_as_element(empty))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::bind_writes(fn, Dispatch::write_as<float>(wide))
                    .require(Core::DataType::Float64, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::bind_writes(fn, Dispatch::write_as_element(dense))
                    .require(Core::DataType::Int32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::bind_writes(fn, Dispatch::write_as_element(strided))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 0));

            EXPECT_TRUE(names_write(message_of([&]()
            {
                Dispatch::bind_writes(fn_of_two, Dispatch::write_as_element(dense),
                                      Dispatch::write_as_element(empty))
                    .require(Core::DataType::Float32, K_OPERATION);
            }), 1));
        }

        TEST_F(Medium_WriteBinding, ALaterReaderWaitsForTheTaskThatWritesTheTensor)
        {
            GraphProbe probe;
            const Tensor r = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor w = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            Tensor out = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            const auto fn = [](float, const float*, float*) {};

            record_task(probe.recorder(),
                        Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as_element(r)),
                                              Dispatch::write_as_element(w)),
                        "writer");
            record_task(probe.recorder(),
                        Dispatch::bind_writes(Dispatch::bind_reads(fn, Dispatch::read_as_element(w)),
                                              Dispatch::write_as_element(out)),
                        "reader");

            const SushiRuntime::Result<std::vector<RecordedTask>> tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks);
            ASSERT_EQ(tasks->size(), std::size_t{2});
            EXPECT_EQ((*tasks)[0].dependency_count, std::size_t{0});
            EXPECT_EQ((*tasks)[1].dependency_count, std::size_t{1});
            EXPECT_EQ((*tasks)[0].reads.size(), std::size_t{1});
            EXPECT_EQ((*tasks)[0].writes.size(), std::size_t{1});
        }

        TEST_F(Medium_WriteBinding, ATensorReadAndWrittenIsDeclaredOnceAsEach)
        {
            GraphProbe probe;
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

            record_task(probe.recorder(), bound, "update");
            const SushiRuntime::Result<std::vector<RecordedTask>> tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks);
            ASSERT_EQ(tasks->size(), std::size_t{1});
            EXPECT_EQ((*tasks)[0].reads.size(), std::size_t{1});
            EXPECT_EQ((*tasks)[0].writes.size(), std::size_t{1});
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

        TEST_F(Medium_WriteBinding, APlainFunctorDeclaresNoWrites)
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
