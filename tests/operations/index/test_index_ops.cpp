/**************************************************************************/
/* test_index_ops.cpp                                                     */
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

/**
 * @file test_index_ops.cpp
 * @brief Tests IndexOps::for_each with a fixed count and with a count a previous node writes.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/index/index_ops.hpp>
#include <SushiRuntime/core/determinism.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief The vertices of the chain the scatter tests walk.
            constexpr std::int64_t K_VERTICES = 33;

            /// @brief The colours of the chain: even edges and odd edges.
            constexpr std::size_t K_COLOURS = 2;

            /// @brief The capacity of the late-count tests.
            constexpr std::int64_t K_CAPACITY = 16;

            /// @brief The count a node writes in the late-count tests.
            constexpr std::int32_t K_WRITTEN = 5;

            /// @brief Returns the bits of a float, so two results compare bit for bit.
            std::uint32_t bits_of(float value) noexcept
            {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }

            /// @brief Returns the bits of every element, in order.
            std::vector<std::uint32_t> bits_of(const std::vector<float>& values)
            {
                std::vector<std::uint32_t> bits;
                bits.reserve(values.size());
                for (const float value : values)
                    bits.push_back(bits_of(value));
                return bits;
            }

            /// @brief Returns the first vertex of edge j of the chain; the other is the next one.
            std::int32_t edge_origin(std::int64_t j) noexcept
            {
                return static_cast<std::int32_t>(j);
            }

            /// @brief Returns the weight of edge j; a multiple of 0.25, so every sum is exact.
            float edge_weight(std::int64_t j) noexcept
            {
                return 0.25f * static_cast<float>((j * 7) % 5 + 1);
            }

            /// @brief Returns the value vertex v starts with.
            float start_value(std::int64_t v) noexcept
            {
                return 0.5f * static_cast<float>(v % 4);
            }

            /// @brief Returns the edges of one colour: the even edges of the chain, then the odd ones.
            std::vector<std::int64_t> edges_of_colour(std::size_t colour)
            {
                std::vector<std::int64_t> edges;
                for (std::int64_t j = static_cast<std::int64_t>(colour); j + 1 < K_VERTICES; j += 2)
                    edges.push_back(j);
                return edges;
            }

            /// @brief Applies both colours on the host, in order; the expectation the tests compare to.
            std::vector<float> scattered_on_the_host()
            {
                std::vector<float> x;
                for (std::int64_t v = 0; v < K_VERTICES; ++v)
                    x.push_back(start_value(v));

                for (std::size_t colour = 0; colour < K_COLOURS; ++colour)
                {
                    for (const std::int64_t j : edges_of_colour(colour))
                    {
                        const std::size_t a = static_cast<std::size_t>(edge_origin(j));
                        x[a] += edge_weight(j);
                        x[a + 1] -= 2.0f * edge_weight(j);
                    }
                }

                return x;
            }
        } // namespace

        /** @brief Gives each test the fixture's engine and the per-index view over its recorder. */
        class Medium_IndexOps : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Records the two colours of the chain scatter, runs them under mode and
                 *        returns the vertices.
                 */
                std::vector<float> scatter_under(ExecutionMode mode)
                {
                    set_mode(mode);
                    IndexOps index(recorder());

                    Tensor x = engine->create_tensor({K_VERTICES}, Core::DataType::Float32);
                    float* values = x.data_as<float>();
                    for (std::int64_t v = 0; v < K_VERTICES; ++v)
                        values[v] = start_value(v);

                    std::array<Tensor, K_COLOURS> origins;
                    std::array<Tensor, K_COLOURS> weights;
                    for (std::size_t colour = 0; colour < K_COLOURS; ++colour)
                    {
                        const std::vector<std::int64_t> edges = edges_of_colour(colour);
                        const std::int64_t count = static_cast<std::int64_t>(edges.size());
                        origins[colour] = engine->create_tensor({count}, Core::DataType::Int32);
                        weights[colour] = engine->create_tensor({count}, Core::DataType::Float32);
                        for (std::int64_t k = 0; k < count; ++k)
                        {
                            origins[colour].data_as<std::int32_t>()[k] = edge_origin(edges[k]);
                            weights[colour].data_as<float>()[k] = edge_weight(edges[k]);
                        }
                    }

                    for (std::size_t colour = 0; colour < K_COLOURS; ++colour)
                    {
                        const auto scatter = [](std::size_t e, const std::int32_t* origin,
                                                const float* weight, auto* target)
                        {
                            using Element = std::remove_pointer_t<decltype(target)>;
                            const std::size_t a = static_cast<std::size_t>(origin[e]);
                            const float w = weight[e];
                            target[a] = static_cast<Element>(static_cast<float>(target[a]) + w);
                            target[a + 1] =
                                static_cast<Element>(static_cast<float>(target[a + 1]) - 2.0f * w);
                        };

                        index.for_each(
                            static_cast<std::size_t>(origins[colour].num_elements),
                            Dispatch::bind_writes(
                                Dispatch::bind_reads(scatter,
                                                     Dispatch::read_as<std::int32_t>(origins[colour]),
                                                     Dispatch::read_as<float>(weights[colour])),
                                Dispatch::write_as_element(x)),
                            OperationOptions{"scatter_colour", SushiRuntime::Determinism::Bitwise});
                    }

                    EXPECT_TRUE(recording->run_succeeds());
                    return read_tensor<float>(x);
                }

                /**
                 * @brief Records a node that writes written into a count tensor of Count, then a
                 *        task over K_CAPACITY indices that marks index i with 1, runs both under
                 *        mode and returns the marks.
                 * @tparam Count The element type of the count tensor: std::int32_t or std::uint32_t.
                 */
                template<typename Count>
                std::vector<float> marks_under_count(ExecutionMode mode, Count written)
                {
                    set_mode(mode);
                    IndexOps index(recorder());

                    Tensor count = engine->create_tensor({1}, Detail::HostDataType<Count>::K_VALUE,
                                                         SushiRuntime::Residency::Host);
                    count.data_as<Count>()[0] = static_cast<Count>(K_CAPACITY + 1);
                    Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32);
                    for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                        marks.data_as<float>()[i] = 0.0f;

                    index.for_each(
                        1,
                        Dispatch::bind_writes(
                            [written](std::size_t, Count* slot) { slot[0] = written; },
                            Dispatch::write_as<Count>(count)),
                        OperationOptions{"write_count", SushiRuntime::Determinism::Bitwise});

                    index.for_each(
                        static_cast<std::size_t>(K_CAPACITY), count,
                        Dispatch::bind_writes(
                            [](std::size_t i, float* out) { out[i] = 1.0f; },
                            Dispatch::write_as<float>(marks)),
                        OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});

                    EXPECT_TRUE(recording->run_succeeds());
                    return read_tensor<float>(marks);
                }

                /** @brief Runs marks_under_count over an INT32 count. */
                std::vector<float> marks_under(ExecutionMode mode, std::int32_t written)
                {
                    return marks_under_count<std::int32_t>(mode, written);
                }
        };

        TEST_F(Medium_IndexOps, AColouredScatterMatchesItsForceReferenceRunBitForBit)
        {
            const std::vector<float> by_reference = scatter_under(ExecutionMode::ForceReference);
            const std::vector<float> by_sycl = scatter_under(ExecutionMode::ForceSYCL);

            EXPECT_EQ(bits_of(by_sycl), bits_of(by_reference));
        }

        TEST_F(Medium_IndexOps, AColouredScatterEqualsTheHandOrderedHostLoop)
        {
            const std::vector<float> by_reference = scatter_under(ExecutionMode::ForceReference);

            EXPECT_EQ(bits_of(by_reference), bits_of(scattered_on_the_host()));
        }

        TEST_F(Medium_IndexOps, ALateCountRunsOnlyTheCountAPreviousNodeWrote)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                const std::vector<float> marks = marks_under(mode, K_WRITTEN);

                for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                    EXPECT_EQ(marks[static_cast<std::size_t>(i)], i < K_WRITTEN ? 1.0f : 0.0f)
                        << "index " << i;
            }
        }

        TEST_F(Medium_IndexOps, ALateCountOfTypeUInt32RunsOnlyTheCountAPreviousNodeWrote)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                const std::vector<float> marks =
                    marks_under_count<std::uint32_t>(mode, static_cast<std::uint32_t>(K_WRITTEN));

                for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                    EXPECT_EQ(marks[static_cast<std::size_t>(i)], i < K_WRITTEN ? 1.0f : 0.0f)
                        << "index " << i;
            }
        }

        TEST_F(Medium_IndexOps, ALateCountOfTypeUInt32WrittenOnTheHostIsRead)
        {
            Tensor count = engine->create_tensor({1}, Core::DataType::UInt32,
                                                 SushiRuntime::Residency::Host);
            count.data_as<std::uint32_t>()[0] = static_cast<std::uint32_t>(K_WRITTEN);
            Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32);
            for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                marks.data_as<float>()[i] = 0.0f;

            set_mode(ExecutionMode::ForceSYCL);
            IndexOps(recorder()).for_each(
                static_cast<std::size_t>(K_CAPACITY), count,
                Dispatch::bind_writes([](std::size_t i, float* out) { out[i] = 1.0f; },
                                      Dispatch::write_as<float>(marks)),
                OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});

            EXPECT_TRUE(recording->run_succeeds());
            EXPECT_EQ(marks.data_as<float>()[K_WRITTEN - 1], 1.0f);
            EXPECT_EQ(marks.data_as<float>()[K_WRITTEN], 0.0f);
        }

        TEST_F(Medium_IndexOps, ALateCountOfZeroRunsNothing)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                const std::vector<float> marks = marks_under(mode, 0);

                for (const float mark : marks)
                    EXPECT_EQ(mark, 0.0f);
            }
        }

        TEST_F(Medium_IndexOps, ALateCountOfTheCapacityRunsEveryIndex)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                const std::vector<float> marks =
                    marks_under(mode, static_cast<std::int32_t>(K_CAPACITY));

                for (const float mark : marks)
                    EXPECT_EQ(mark, 1.0f);
            }
        }

        TEST_F(Medium_IndexOps, ALateCountOutsideTheCapacityFailsTheRun)
        {
            for (const std::int32_t written :
                 {static_cast<std::int32_t>(K_CAPACITY + 5), std::int32_t{-3}})
            {
                set_mode(ExecutionMode::ForceSYCL);
                IndexOps index(recorder());
                Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                     SushiRuntime::Residency::Host);
                count.data_as<std::int32_t>()[0] = written;
                Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32);

                index.for_each(static_cast<std::size_t>(K_CAPACITY), count,
                               Dispatch::bind_writes(
                                   [](std::size_t i, float* out) { out[i] = 1.0f; },
                                   Dispatch::write_as<float>(marks)),
                               OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});

                EXPECT_FALSE(recording->run_succeeds()) << "count " << written;
            }
        }

        TEST_F(Medium_IndexOps, ACountOfZeroRecordsNothing)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

            const TaskHandle handle = index.for_each(
                0, Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                         Dispatch::write_as<float>(out)),
                OperationOptions{"empty", SushiRuntime::Determinism::Bitwise});

            EXPECT_FALSE(handle.is_valid());
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
        }

        TEST_F(Medium_IndexOps, TheNodeCarriesTheOptionsNameAndOrdersALaterReader)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor sink = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(out)),
                           OperationOptions{"fill_out", SushiRuntime::Determinism::Tolerant});
            index.for_each(4,
                           Dispatch::bind_writes(
                               Dispatch::bind_reads([](std::size_t i, const float* in, float* p)
                                                    { p[i] = in[i]; },
                                                    Dispatch::read_as<float>(out)),
                               Dispatch::write_as<float>(sink)),
                           OperationOptions{"copy_out", SushiRuntime::Determinism::Cosmetic});

            const SushiRuntime::Result<std::vector<RecordedTask>> tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks);
            ASSERT_EQ(tasks->size(), std::size_t{2});
            EXPECT_EQ((*tasks)[0].name, "fill_out");
            EXPECT_EQ((*tasks)[0].operation, "index.for_each"_op);
            EXPECT_EQ((*tasks)[0].dependency_count, std::size_t{0});
            EXPECT_EQ((*tasks)[1].name, "copy_out");
            EXPECT_EQ((*tasks)[1].dependency_count, std::size_t{1});
        }

        TEST_F(Medium_IndexOps, AnUpdateIsOrderedAfterAnEarlierWriterAndBeforeALaterReader)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor sink = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(x)),
                           OperationOptions{"fill_x", SushiRuntime::Determinism::Bitwise});
            index.for_each(4,
                           Dispatch::bind_updates([](std::size_t i, float* p) { p[i] += 1.0f; },
                                                  Dispatch::update_as<float>(x)),
                           OperationOptions{"bump_x", SushiRuntime::Determinism::Bitwise});
            index.for_each(4,
                           Dispatch::bind_writes(
                               Dispatch::bind_reads([](std::size_t i, const float* in, float* p)
                                                    { p[i] = in[i]; },
                                                    Dispatch::read_as<float>(x)),
                               Dispatch::write_as<float>(sink)),
                           OperationOptions{"copy_x", SushiRuntime::Determinism::Bitwise});

            const SushiRuntime::Result<std::vector<RecordedTask>> tasks = probe.recorded_tasks();
            ASSERT_TRUE(tasks);
            ASSERT_EQ(tasks->size(), std::size_t{3});
            EXPECT_EQ((*tasks)[1].name, "bump_x");
            EXPECT_EQ((*tasks)[1].reads.size(), std::size_t{1});
            EXPECT_EQ((*tasks)[1].writes.size(), std::size_t{1});
            EXPECT_EQ((*tasks)[0].dependency_count, std::size_t{0});
            EXPECT_EQ((*tasks)[1].dependency_count, std::size_t{1});
            EXPECT_EQ((*tasks)[2].dependency_count, std::size_t{1});
        }

        TEST_F(Medium_IndexOps, AnUpdateHandsTheFunctorThePreviousContentsBetweenReadsAndWrites)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                set_mode(mode);
                IndexOps index(recorder());
                Tensor w = engine->create_tensor({4}, Core::DataType::Float32);
                Tensor x = engine->create_tensor({4}, Core::DataType::Float32);
                Tensor before = engine->create_tensor({4}, Core::DataType::Float32);
                for (std::int64_t i = 0; i < 4; ++i)
                {
                    w.data_as<float>()[i] = 0.25f;
                    x.data_as<float>()[i] = 0.5f * static_cast<float>(i);
                    before.data_as<float>()[i] = -1.0f;
                }

                const auto step = [](std::size_t i, const float* weight, auto* value, float* old)
                {
                    using Element = std::remove_pointer_t<decltype(value)>;
                    const float current = static_cast<float>(value[i]);
                    old[i] = current;
                    value[i] = static_cast<Element>(2.0f * current + weight[i]);
                };
                index.for_each(4,
                               Dispatch::bind_writes(
                                   Dispatch::bind_updates(
                                       Dispatch::bind_reads(step, Dispatch::read_as<float>(w)),
                                       Dispatch::update_as_element(x)),
                                   Dispatch::write_as<float>(before)),
                               OperationOptions{"step_x", SushiRuntime::Determinism::Bitwise});

                EXPECT_TRUE(recording->run_succeeds());
                EXPECT_EQ(bits_of(read_tensor<float>(before)),
                          bits_of(std::vector<float>{0.0f, 0.5f, 1.0f, 1.5f}));
                EXPECT_EQ(bits_of(read_tensor<float>(x)),
                          bits_of(std::vector<float>{0.25f, 1.25f, 2.25f, 3.25f}));
            }
        }

        TEST_F(Medium_IndexOps, AnUpdateThatOverlapsAnotherOperandIsRefused)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor head = x.slice(0, 0, 2);
            const OperationOptions options{"overlap", SushiRuntime::Determinism::Bitwise};

            EXPECT_THROW(index.for_each(4,
                                        Dispatch::bind_updates(
                                            Dispatch::bind_reads(
                                                [](std::size_t, const float*, float*) {},
                                                Dispatch::read_as<float>(x)),
                                            Dispatch::update_as<float>(x)),
                                        options),
                         std::invalid_argument);
            EXPECT_THROW(index.for_each(4,
                                        Dispatch::bind_writes(
                                            Dispatch::bind_updates(
                                                [](std::size_t, float*, float*) {},
                                                Dispatch::update_as<float>(x)),
                                            Dispatch::write_as<float>(x)),
                                        options),
                         std::invalid_argument);
            EXPECT_THROW(index.for_each(2,
                                        Dispatch::bind_writes(
                                            Dispatch::bind_updates(
                                                [](std::size_t, float*, float*) {},
                                                Dispatch::update_as<float>(x)),
                                            Dispatch::write_as<float>(head)),
                                        options),
                         std::invalid_argument);
            EXPECT_THROW(index.for_each(2,
                                        Dispatch::bind_updates(
                                            [](std::size_t, float*, float*) {},
                                            Dispatch::update_as<float>(x),
                                            Dispatch::update_as<float>(head)),
                                        options),
                         std::invalid_argument);
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
        }

        TEST_F(Medium_IndexOps, ALateCountTaskWaitsForTheNodeThatWritesTheCount)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                GraphProbe probe(mode);
                IndexOps index(probe.recorder());
                Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                     SushiRuntime::Residency::Host);
                Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

                index.for_each(1,
                               Dispatch::bind_writes(
                                   [](std::size_t, std::int32_t* slot) { slot[0] = 2; },
                                   Dispatch::write_as<std::int32_t>(count)),
                               OperationOptions{"write_count", SushiRuntime::Determinism::Bitwise});
                index.for_each(4, count,
                               Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                     Dispatch::write_as<float>(out)),
                               OperationOptions{"use_count", SushiRuntime::Determinism::Bitwise});

                const SushiRuntime::Result<std::vector<std::size_t>> counts =
                    probe.dependency_counts();
                ASSERT_TRUE(counts);
                EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
            }
        }

        TEST_F(Medium_IndexOps, TheRuntimePlanReadsTheDeterminismClassOfEachNode)
        {
            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                set_mode(mode);
                IndexOps index(recorder());
                Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                     SushiRuntime::Residency::Host);
                Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

                index.for_each(1,
                               Dispatch::bind_writes(
                                   [](std::size_t, std::int32_t* slot) { slot[0] = 2; },
                                   Dispatch::write_as<std::int32_t>(count)),
                               OperationOptions{"fixed", SushiRuntime::Determinism::Tolerant});
                index.for_each(4, count,
                               Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                     Dispatch::write_as<float>(out)),
                               OperationOptions{"late", SushiRuntime::Determinism::Cosmetic});

                SushiRuntime::Result<SushiRuntime::PlanView> plan = recording->plan();
                ASSERT_TRUE(plan);
                ASSERT_EQ(plan->nodes.size(), std::size_t{2});
                EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Tolerant);
                EXPECT_EQ(plan->nodes[1].determinism, SushiRuntime::Determinism::Cosmetic);

                EXPECT_TRUE(recording->run_succeeds());
            }
        }

        TEST_F(Medium_IndexOps, ANodeThatStatesNoClassReadsUnstated)
        {
            set_mode(ExecutionMode::ForceReference);
            IndexOps index(recorder());
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(out)),
                           OperationOptions{"silent"});

            SushiRuntime::Result<SushiRuntime::PlanView> plan = recording->plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{1});
            EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Unstated);
        }

        TEST_F(Medium_IndexOps, AnEmptyNameIsRefused)
        {
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            IndexOps index(recorder());

            EXPECT_THROW(index.for_each(4,
                                        Dispatch::bind_writes(
                                            [](std::size_t i, float* p) { p[i] = 1.0f; },
                                            Dispatch::write_as<float>(out)),
                                        OperationOptions{"", SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
        }

        TEST_F(Medium_IndexOps, ACountTensorThatIsNeitherUInt32NorInt32InHostMemoryIsRefused)
        {
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            IndexOps index(recorder());

            const std::array<Tensor, 4> refused = {
                engine->create_tensor({1}, Core::DataType::Float32, SushiRuntime::Residency::Host),
                engine->create_tensor({1}, Core::DataType::Int64, SushiRuntime::Residency::Host),
                engine->create_tensor({1}, Core::DataType::Int32, SushiRuntime::Residency::Shared),
                engine->create_tensor({1}, Core::DataType::UInt32, SushiRuntime::Residency::Shared)};

            for (const ExecutionMode mode :
                 {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})
            {
                set_mode(mode);
                for (const Tensor& count : refused)
                {
                    EXPECT_THROW(index.for_each(4, count,
                                                Dispatch::bind_writes(
                                                    [](std::size_t i, float* p) { p[i] = 1.0f; },
                                                    Dispatch::write_as<float>(out)),
                                                OperationOptions{"bad_count",
                                                             SushiRuntime::Determinism::Bitwise}),
                                 std::invalid_argument);
                }
            }
        }

        TEST_F(Medium_IndexOps, ATaskThatWritesItsOwnCountIsRefused)
        {
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            IndexOps index(recorder());

            EXPECT_THROW(index.for_each(4, count,
                                        Dispatch::bind_writes(
                                            [](std::size_t, std::int32_t* slot) { slot[0] = 1; },
                                            Dispatch::write_as<std::int32_t>(count)),
                                        OperationOptions{"self_count",
                                                     SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
        }

        /** @brief Runs the IndexOps tests that guard who owns a late count's storages. */
        class Medium_Regression_IndexOps : public Medium_IndexOps
        {

        };

        /**
         * @brief Tests that a late for_each recorded through a recorder gone before the graph
         *        runs replays twice on the live count while its caller keeps the hold, and that
         *        releasing the hold leaves the tensors the only owners of their storages.
         */
        TEST_F(Medium_Regression_IndexOps, AReplayedLateForEachOutlivesItsRecorder)
        {
            SushiRuntime::Runtime& runtime = engine->runtime();
            SushiRuntime::Graph graph = runtime.graph();
            ScratchPool scratch_pool(runtime);
            LateCountHold late_counts(runtime);
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32,
                                                 SushiRuntime::Residency::Shared);
            {
                TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts,
                                      ExecutionMode::ForceSYCL);
                IndexOps(recorder).for_each(
                    static_cast<std::size_t>(K_CAPACITY), count,
                    Dispatch::bind_writes([](std::size_t i, float* out) { out[i] = 1.0f; },
                                          Dispatch::write_as<float>(marks)),
                    OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});
            }

            for (const std::int32_t live : {3, 5})
            {
                count.data_as<std::int32_t>()[0] = live;
                for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                    marks.data_as<float>()[i] = 0.0f;
                SushiRuntime::RunReport report;
                const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);

                ASSERT_TRUE(ran) << "run over a count of " << live;
                EXPECT_EQ(marks.data_as<float>()[live - 1], 1.0f) << "count " << live;
                EXPECT_EQ(marks.data_as<float>()[live], 0.0f) << "count " << live;
            }

            graph.clear();
            late_counts.release();
            EXPECT_EQ(late_counts.held_count(), std::size_t{0});
            EXPECT_EQ(count.storage->ref_count(), 1);
            EXPECT_EQ(marks.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
