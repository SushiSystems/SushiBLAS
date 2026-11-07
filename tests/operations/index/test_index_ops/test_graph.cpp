/**************************************************************************/
/* test_graph.cpp                                                         */
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

/**
 * @file test_graph.cpp
 * @brief Tests IndexOps::for_each with a fixed count and with a count a previous node writes.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
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
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>

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

            /// @brief Returns the marks of K_CAPACITY indices when the first count of them ran.
            std::vector<float> marks_of_the_first(std::int64_t count)
            {
                std::vector<float> marks(static_cast<std::size_t>(K_CAPACITY), 0.0f);
                for (std::int64_t i = 0; i < count; ++i)
                    marks[static_cast<std::size_t>(i)] = 1.0f;
                return marks;
            }

            /// @brief Returns the name a test instance carries for an execution mode.
            std::string mode_name(ExecutionMode mode)
            {
                return mode == ExecutionMode::ForceReference ? "ForceReference" : "ForceSYCL";
            }

            /// @brief Says who sets the count tensor a late-count task reads.
            enum class CountSource
            {
                WrittenByANode, ///< A recorded node writes it before the task runs.
                WrittenOnTheHost ///< The caller stores it before the graph runs.
            };

            /// @brief One late count: the mode, the count tensor's type and source, and its value.
            struct LateCountRow
            {
                const char* name;
                ExecutionMode mode;
                Core::DataType type;
                CountSource source;
                std::int32_t written;
            };

            /// @brief Returns the late counts that run exactly the indices below the count.
            std::vector<LateCountRow> late_count_rows()
            {
                return {
                    {"ReferenceInt32FromANode", ExecutionMode::ForceReference, Core::DataType::Int32,
                     CountSource::WrittenByANode, K_WRITTEN},
                    {"SyclInt32FromANode", ExecutionMode::ForceSYCL, Core::DataType::Int32,
                     CountSource::WrittenByANode, K_WRITTEN},
                    {"ReferenceUInt32FromANode", ExecutionMode::ForceReference,
                     Core::DataType::UInt32, CountSource::WrittenByANode, K_WRITTEN},
                    {"SyclUInt32FromANode", ExecutionMode::ForceSYCL, Core::DataType::UInt32,
                     CountSource::WrittenByANode, K_WRITTEN},
                    {"SyclUInt32FromTheHost", ExecutionMode::ForceSYCL, Core::DataType::UInt32,
                     CountSource::WrittenOnTheHost, K_WRITTEN},
                    {"ReferenceZero", ExecutionMode::ForceReference, Core::DataType::Int32,
                     CountSource::WrittenByANode, 0},
                    {"SyclZero", ExecutionMode::ForceSYCL, Core::DataType::Int32,
                     CountSource::WrittenByANode, 0},
                    {"ReferenceTheCapacity", ExecutionMode::ForceReference, Core::DataType::Int32,
                     CountSource::WrittenByANode, static_cast<std::int32_t>(K_CAPACITY)},
                    {"SyclTheCapacity", ExecutionMode::ForceSYCL, Core::DataType::Int32,
                     CountSource::WrittenByANode, static_cast<std::int32_t>(K_CAPACITY)}};
            }

            /// @brief One count the runtime refuses to run: the capacity and the count read.
            struct OutOfRangeRow
            {
                const char* name;
                std::size_t capacity;
                std::int32_t written;
            };

            /// @brief Returns the counts outside [0, capacity].
            std::vector<OutOfRangeRow> out_of_range_rows()
            {
                return {{"AboveTheCapacity", static_cast<std::size_t>(K_CAPACITY),
                         static_cast<std::int32_t>(K_CAPACITY + 5)},
                        {"Negative", static_cast<std::size_t>(K_CAPACITY), std::int32_t{-3}},
                        {"AboveACapacityOfZero", std::size_t{0}, std::int32_t{2}}};
            }

            /// @brief One count tensor in shared memory for_each refuses: what it holds.
            struct CountRefusalRow
            {
                std::string name;
                ExecutionMode mode;
                Core::DataType type;
                SushiRuntime::Residency residency;
                std::size_t capacity;
            };

            /// @brief Returns the refused count tensors that live in shared memory, in both modes.
            std::vector<CountRefusalRow> count_refusal_rows()
            {
                struct Refused
                {
                    const char* name;
                    Core::DataType type;
                    SushiRuntime::Residency residency;
                };
                const std::array<Refused, 2> refused = {
                    {{"Int32InSharedMemory", Core::DataType::Int32, SushiRuntime::Residency::Shared},
                     {"UInt32InSharedMemory", Core::DataType::UInt32,
                      SushiRuntime::Residency::Shared}}};

                std::vector<CountRefusalRow> rows;
                for (const Refused& kind : refused)
                {
                    for (const ExecutionMode mode : K_BOTH_MODES)
                        rows.push_back({std::string(kind.name) + "Under" + mode_name(mode), mode,
                                        kind.type, kind.residency, std::size_t{4}});
                }
                return rows;
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
                 * @brief Records a count tensor of Count and a task over K_CAPACITY indices that
                 *        marks index i with 1, runs them under mode and returns the marks.
                 * @tparam Count The element type of the count tensor: std::int32_t or std::uint32_t.
                 * @param source Whether a recorded node writes written into the count or the host
                 *        stores it before the run.
                 */
                template<typename Count>
                std::vector<float> marks_under_count(ExecutionMode mode, Count written,
                                                     CountSource source)
                {
                    set_mode(mode);
                    IndexOps index(recorder());

                    Tensor count = engine->create_tensor({1}, ::SushiBLAS::Detail::HostDataType<Count>::K_VALUE,
                                                         SushiRuntime::Residency::Host);
                    count.data_as<Count>()[0] = source == CountSource::WrittenByANode
                                                    ? static_cast<Count>(K_CAPACITY + 1)
                                                    : written;
                    Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32);
                    for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                        marks.data_as<float>()[i] = 0.0f;

                    if (source == CountSource::WrittenByANode)
                    {
                        index.for_each(
                            1,
                            Dispatch::bind_writes(
                                [written](std::size_t, Count* slot) { slot[0] = written; },
                                Dispatch::write_as<Count>(count)),
                            OperationOptions{"write_count", SushiRuntime::Determinism::Bitwise});
                    }

                    index.for_each(
                        static_cast<std::size_t>(K_CAPACITY), count,
                        Dispatch::bind_writes(
                            [](std::size_t i, float* out) { out[i] = 1.0f; },
                            Dispatch::write_as<float>(marks)),
                        OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});

                    EXPECT_TRUE(recording->run_succeeds());
                    return read_tensor<float>(marks);
                }

                /** @brief Runs marks_under_count over the count type and source the row names. */
                std::vector<float> marks_under(const LateCountRow& row)
                {
                    if (row.type == Core::DataType::UInt32)
                    {
                        return marks_under_count<std::uint32_t>(
                            row.mode, static_cast<std::uint32_t>(row.written), row.source);
                    }
                    return marks_under_count<std::int32_t>(row.mode, row.written, row.source);
                }

                /**
                 * @brief Records a late for_each over count and marks through recorder, then lets
                 *        the recorder go before the graph runs.
                 */
                void record_late_for_each_and_drop_its_recorder(SushiRuntime::Graph& graph,
                                                                ScratchPool& scratch_pool,
                                                                LateCountHold& late_counts,
                                                                const Tensor& count,
                                                                Tensor& marks)
                {
                    TaskRecorder recorder(graph.recorder(), scratch_pool, late_counts,
                                          ExecutionMode::ForceSYCL);
                    IndexOps(recorder).for_each(
                        static_cast<std::size_t>(K_CAPACITY), count,
                        Dispatch::bind_writes([](std::size_t i, float* out) { out[i] = 1.0f; },
                                              Dispatch::write_as<float>(marks)),
                        OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});
                }

                /** @brief Stores live in count, clears marks, runs graph once and returns the marks. */
                std::vector<float> run_over_live_count(SushiRuntime::Graph& graph, Tensor& count,
                                                       Tensor& marks, std::int32_t live)
                {
                    count.data_as<std::int32_t>()[0] = live;
                    for (std::int64_t i = 0; i < K_CAPACITY; ++i)
                        marks.data_as<float>()[i] = 0.0f;

                    SushiRuntime::RunReport report;
                    const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);
                    EXPECT_TRUE(ran) << "run over a count of " << live;
                    return read_tensor<float>(marks);
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

        TEST_F(Medium_IndexOps, ATaskThatReadsWhatAnEarlierNodeWroteWaitsForIt)
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

            const SushiRuntime::Result<std::vector<std::size_t>> counts = probe.dependency_counts();
            ASSERT_TRUE(counts);
            EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
        }

        TEST_F(Medium_IndexOps, AnUpdateWaitsForAnEarlierWriter)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);

            index.for_each(4,
                           Dispatch::bind_writes([](std::size_t i, float* p) { p[i] = 1.0f; },
                                                 Dispatch::write_as<float>(x)),
                           OperationOptions{"fill_x", SushiRuntime::Determinism::Bitwise});
            index.for_each(4,
                           Dispatch::bind_updates([](std::size_t i, float* p) { p[i] += 1.0f; },
                                                  Dispatch::update_as<float>(x)),
                           OperationOptions{"bump_x", SushiRuntime::Determinism::Bitwise});

            const SushiRuntime::Result<std::vector<std::size_t>> counts = probe.dependency_counts();
            ASSERT_TRUE(counts);
            EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
        }

        TEST_F(Medium_IndexOps, ALaterReaderWaitsForAnUpdate)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            IndexOps index(probe.recorder());
            Tensor x = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor sink = engine->create_tensor({4}, Core::DataType::Float32);

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

            const SushiRuntime::Result<std::vector<std::size_t>> counts = probe.dependency_counts();
            ASSERT_TRUE(counts);
            EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
        }

        /** @brief Runs one late count of the table of late_count_rows(). */
        class Medium_IndexOpsLateCount : public Medium_IndexOps,
                                         public ::testing::WithParamInterface<LateCountRow>
        {

        };

        TEST_P(Medium_IndexOpsLateCount, RunsOnlyTheIndicesBelowTheCountItReads)
        {
            const LateCountRow row = GetParam();

            EXPECT_EQ(marks_under(row), marks_of_the_first(row.written));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IndexOpsLateCount, ::testing::ValuesIn(late_count_rows()),
            [](const ::testing::TestParamInfo<LateCountRow>& info)
            { return std::string(info.param.name); });

        /** @brief Runs one count outside the capacity of the table of out_of_range_rows(). */
        class Medium_IndexOpsLateCountOutOfRange
            : public Medium_IndexOps, public ::testing::WithParamInterface<OutOfRangeRow>
        {

        };

        TEST_P(Medium_IndexOpsLateCountOutOfRange, FailsTheRun)
        {
            const OutOfRangeRow row = GetParam();
            set_mode(ExecutionMode::ForceSYCL);
            IndexOps index(recorder());
            Tensor count =
                engine->create_tensor({1}, Core::DataType::Int32, SushiRuntime::Residency::Host);
            count.data_as<std::int32_t>()[0] = row.written;
            Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32);

            index.for_each(row.capacity, count,
                           Dispatch::bind_writes([](std::size_t i, float* out) { out[i] = 1.0f; },
                                                 Dispatch::write_as<float>(marks)),
                           OperationOptions{"mark", SushiRuntime::Determinism::Bitwise});

            EXPECT_FALSE(recording->run_succeeds());
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IndexOpsLateCountOutOfRange, ::testing::ValuesIn(out_of_range_rows()),
            [](const ::testing::TestParamInfo<OutOfRangeRow>& info)
            { return std::string(info.param.name); });

        /** @brief Runs one count tensor of the table of count_refusal_rows(). */
        class Medium_IndexOpsCountRefusal : public Medium_IndexOps,
                                            public ::testing::WithParamInterface<CountRefusalRow>
        {

        };

        TEST_P(Medium_IndexOpsCountRefusal, IsRefusedWhenItIsNotAHostInt32OrUInt32)
        {
            const CountRefusalRow row = GetParam();
            set_mode(row.mode);
            Tensor out = engine->create_tensor({4}, Core::DataType::Float32);
            Tensor count = engine->create_tensor({1}, row.type, row.residency);
            IndexOps index(recorder());

            EXPECT_THROW(index.for_each(row.capacity, count,
                                        Dispatch::bind_writes(
                                            [](std::size_t i, float* p) { p[i] = 1.0f; },
                                            Dispatch::write_as<float>(out)),
                                        OperationOptions{"bad_count",
                                                         SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IndexOpsCountRefusal, ::testing::ValuesIn(count_refusal_rows()),
            [](const ::testing::TestParamInfo<CountRefusalRow>& info) { return info.param.name; });

        /** @brief Runs one test in each execution mode. */
        class Medium_IndexOpsInEachMode : public Medium_IndexOps,
                                          public ::testing::WithParamInterface<ExecutionMode>
        {

        };

        TEST_P(Medium_IndexOpsInEachMode, AnUpdateHandsTheFunctorThePreviousContentsBetweenReadsAndWrites)
        {
            set_mode(GetParam());
            IndexOps index(recorder());
            Tensor w = create_initialized_tensor<float>({4}, {0.25f, 0.25f, 0.25f, 0.25f});
            Tensor x = create_initialized_tensor<float>({4}, {0.0f, 0.5f, 1.0f, 1.5f});
            Tensor before = create_initialized_tensor<float>({4}, {-1.0f, -1.0f, -1.0f, -1.0f});

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

        TEST_P(Medium_IndexOpsInEachMode, ALateCountTaskWaitsForTheNodeThatWritesTheCount)
        {
            GraphProbe probe(GetParam());
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

            const SushiRuntime::Result<std::vector<std::size_t>> counts = probe.dependency_counts();
            ASSERT_TRUE(counts);
            EXPECT_EQ(*counts, (std::vector<std::size_t>{0, 1}));
        }

        TEST_P(Medium_IndexOpsInEachMode, TheRuntimePlanReadsTheDeterminismClassOfEachNode)
        {
            set_mode(GetParam());
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

        INSTANTIATE_TEST_SUITE_P(
            , Medium_IndexOpsInEachMode, ::testing::ValuesIn(K_BOTH_MODES),
            [](const ::testing::TestParamInfo<ExecutionMode>& info) { return mode_name(info.param); });

        /** @brief Runs the IndexOps tests that guard who owns a late count's storages. */
        class Medium_Regression_IndexOps : public Medium_IndexOps
        {

        };

        /**
         * @brief Tests that a late for_each recorded through a recorder gone before the graph
         *        runs replays twice on the live count while its caller keeps the hold.
         */
        TEST_F(Medium_Regression_IndexOps, AReplayedLateForEachRunsOnTheLiveCountOfEachRun)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            SushiRuntime::Graph graph = runtime.graph();
            RuntimeScratchSource scratch_pool_source(runtime);
            ScratchPool scratch_pool(scratch_pool_source);
            LateCountHold late_counts(runtime);
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32,
                                                 SushiRuntime::Residency::Shared);
            record_late_for_each_and_drop_its_recorder(graph, scratch_pool, late_counts, count,
                                                       marks);

            EXPECT_EQ(run_over_live_count(graph, count, marks, 3), marks_of_the_first(3));
            EXPECT_EQ(run_over_live_count(graph, count, marks, 5), marks_of_the_first(5));
        }

        /**
         * @brief Tests that releasing the hold of a late for_each leaves the tensors the only
         *        owners of their storages.
         */
        TEST_F(Medium_Regression_IndexOps, ReleasingTheHoldOfALateForEachLeavesTheTensorsTheirOwners)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            SushiRuntime::Graph graph = runtime.graph();
            RuntimeScratchSource scratch_pool_source(runtime);
            ScratchPool scratch_pool(scratch_pool_source);
            LateCountHold late_counts(runtime);
            Tensor count = engine->create_tensor({1}, Core::DataType::Int32,
                                                 SushiRuntime::Residency::Host);
            Tensor marks = engine->create_tensor({K_CAPACITY}, Core::DataType::Float32,
                                                 SushiRuntime::Residency::Shared);
            record_late_for_each_and_drop_its_recorder(graph, scratch_pool, late_counts, count,
                                                       marks);

            graph.clear();
            late_counts.release();

            EXPECT_EQ(late_counts.held_count(), std::size_t{0});
            EXPECT_EQ(count.storage->ref_count(), 1);
            EXPECT_EQ(marks.storage->ref_count(), 1);
        }
    } // namespace Test
} // namespace SushiBLAS
