/**************************************************************************/
/* test_profiler_parts.cpp                                                */
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
 * @file test_profiler_parts.cpp
 * @brief Tests that register_profiler_parts adds SushiBLAS to a registry once and keeps it there.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <map>
#include <utility>
#include <optional>
#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/recording_graph/recording_graph.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/profiling/ceilings_file.hpp>
#include <SushiRuntime/profiling/register_profiler_parts.hpp>
#include <SushiProfiler/session/session.hpp>
#include <SushiProfiler/analysis/table.hpp>
#include <SushiBLAS/profiling/profiler_parts.hpp>
#include <SushiProfiler/registry/registry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns no device, for a registry that needs no ceilings. */
            std::vector<Profiling::DeviceFacts> no_devices()
            {
                return {};
            }

        } // namespace

        TEST(Small_RegisterProfilerParts, AddsTheContributorToTheRegistry)
        {
            SushiProfiler::Registry registry;

            const Profiling::SushiBLASProfilerParts parts =
                Profiling::register_profiler_parts(registry, no_devices(), "");

            ASSERT_EQ(registry.contributors().size(), std::size_t{1});
            EXPECT_EQ(registry.contributors().front(), &parts.environment());
        }

        TEST(Small_RegisterProfilerParts, RefusesASecondRegistration)
        {
            SushiProfiler::Registry registry;
            const Profiling::SushiBLASProfilerParts first =
                Profiling::register_profiler_parts(registry, no_devices(), "");

            EXPECT_THROW(static_cast<void>(
                             Profiling::register_profiler_parts(registry, no_devices(), "")),
                         std::invalid_argument);
            EXPECT_EQ(registry.contributors().size(), std::size_t{1});
        }

        TEST(Small_RegisterProfilerParts, TwoRegistriesTakeOneSetEach)
        {
            SushiProfiler::Registry first_registry;
            SushiProfiler::Registry second_registry;

            const Profiling::SushiBLASProfilerParts first =
                Profiling::register_profiler_parts(first_registry, no_devices(), "");
            const Profiling::SushiBLASProfilerParts second =
                Profiling::register_profiler_parts(second_registry, no_devices(), "");

            EXPECT_EQ(first_registry.contributors().size(), std::size_t{1});
            EXPECT_EQ(second_registry.contributors().size(), std::size_t{1});
        }

        namespace
        {
            /** @brief Runs two steps of a GEMM and an elementwise add under a profiler session and keeps what it recorded. */
            class Medium_ProfilingEndToEnd : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        // The cases run as parallel processes, so each one has its own folder.
                        const std::string name =
                            ::testing::UnitTest::GetInstance()->current_test_info()->name();
                        folder_ = std::filesystem::temp_directory_path() /
                                  ("sushiblas_profiling_end_to_end_" + name);
                        std::filesystem::remove_all(folder_);
                        std::filesystem::create_directories(folder_);
                    }

                    void TearDown() override
                    {
                        std::filesystem::remove_all(folder_);
                    }

                    /** @brief Writes ceilings for the first device and records the given number of periods of the two operations. */
                    void record(std::uint32_t steps, Core::DataType dtype = Core::DataType::Float32,
                                bool with_add = true)
                    {
                        SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
                        const std::string device = runtime.advanced().devices().front().name;
                        Profiling::DeviceCeilings ceilings;
                        ceilings.device_name = device;
                        ceilings.backend = "test";
                        ceilings.driver = "1";
                        ceilings.smoke_only = true;
                        ceilings.method = {1, 3, 1.0, 1.0};
                        ceilings.compute = {{"f32", 1.0e12, 0.0, 3}, {"f16", 4.0e12, 0.0, 3}};
                        ceilings.bandwidth = {1.0e11, 0.0, 3};
                        ceilings.launch_floor_ns = 1000.0;
                        std::ofstream(folder_ / Profiling::ceilings_file_name(device), std::ios::binary)
                            << Profiling::write_ceilings(ceilings);

                        SushiProfiler::Registry registry;
                        SushiRuntime::RuntimeProfilerParts runtime_parts(registry, runtime);
                        const Profiling::SushiBLASProfilerParts blas_parts =
                            Profiling::register_profiler_parts(registry, runtime, folder_.string());

                        SushiProfiler::ProfilerConfig config;
                        config.timeline = "blas.step";
                        config.window.warmup = 0U;
                        config.window.active = steps;
                        config.sources = {SushiProfiler::NamedSettings{"sushiruntime.device",
                                                                      SushiProfiler::JsonValue::object()}};
                        config.analyses = {"roofline"};
                        config.buffers.linger_ms = 0U;
                        SushiProfiler::SessionStart started =
                            SushiProfiler::Session::start(config, registry);
                        ASSERT_TRUE(started.ok()) << started.error;

                        Engine engine(runtime);
                        RecordingGraph graph(runtime, engine.scratch_pool(), engine.mode());
                        const Tensor a = engine.create_tensor({64, 64}, dtype);
                        const Tensor b = engine.create_tensor({64, 64}, dtype);
                        Tensor c = engine.create_tensor({64, 64}, dtype);
                        Tensor d = engine.create_tensor({64, 64}, dtype);
                        SushiProfiler::PeriodTimeline& timeline = started.session->timeline("blas.step");
                        for (std::uint32_t step = 0; step < steps; ++step)
                        {
                            timeline.mark();
                            BLASOps(graph.recorder()).gemm(a, b, c);
                            if (with_add)
                                ElementwiseOps(graph.recorder()).add(c, c, d);
                            graph.run().value();
                        }
                        timeline.mark();
                        SushiProfiler::SessionStop stopped = started.session->stop();
                        record_ = std::move(stopped.record);
                        tables_ = started.session->report(record_).tables;
                    }

                    /** @brief Returns the integer attribute named key of a span row, or -1 when the span has none. */
                    std::int64_t attribute_of(const SushiProfiler::PeriodRecord& period,
                                              std::size_t row, const char* key) const
                    {
                        const SushiProfiler::AttributeColumn* column =
                            period.spans.attribute(record_.find_name(key));
                        if (column == nullptr || column->present[row] == 0)
                            return -1;
                        return column->integers[row];
                    }

                    /** @brief Returns the counts of periods, spans and dropped spans the record holds. */
                    std::string record_summary() const
                    {
                        std::size_t spans = 0;
                        for (const SushiProfiler::PeriodRecord& period : record_.periods)
                            spans += period.spans.size();
                        return std::to_string(record_.periods.size()) + " periods, " +
                               std::to_string(spans) + " spans, " +
                               std::to_string(record_.drops.spans_dropped) + " spans dropped";
                    }

                    /** @brief Returns the period and row of the first span that states the given FLOPs. */
                    std::optional<std::pair<const SushiProfiler::PeriodRecord*, std::size_t>>
                    span_with_flops(std::int64_t flops) const
                    {
                        for (const SushiProfiler::PeriodRecord& period : record_.periods)
                        {
                            for (std::size_t row = 0; row < period.spans.size(); ++row)
                            {
                                if (attribute_of(period, row, "flops") == flops)
                                    return std::make_pair(&period, row);
                            }
                        }
                        return std::nullopt;
                    }

                    /** @brief Expects a roofline row to state a limit of compute or memory and a positive share. */
                    static void expect_row_has_a_bound(const SushiProfiler::Row& row,
                                                       std::size_t limit, std::size_t share)
                    {
                        ASSERT_NE(row[limit].as_text(), nullptr);
                        EXPECT_TRUE(*row[limit].as_text() == "compute" ||
                                    *row[limit].as_text() == "memory");
                        ASSERT_NE(row[share].as_real(), nullptr);
                        EXPECT_GT(*row[share].as_real(), 0.0);
                    }

                    /** @brief Returns how many roofline rows each period holds, in period order. */
                    std::vector<std::size_t> roofline_rows_per_period() const
                    {
                        const SushiProfiler::Table* table = roofline();
                        std::map<std::int64_t, std::size_t> counted;
                        if (table == nullptr)
                            return {};
                        const std::size_t period = table->column("period");
                        for (const SushiProfiler::Row& row : table->rows)
                        {
                            if (row[period].as_whole() != nullptr)
                                ++counted[*row[period].as_whole()];
                        }
                        std::vector<std::size_t> rows;
                        for (const auto& entry : counted)
                            rows.push_back(entry.second);
                        return rows;
                    }

                    /** @brief Returns the roofline table, or null when the session made none. */
                    const SushiProfiler::Table* roofline() const
                    {
                        for (const SushiProfiler::Table& table : tables_)
                        {
                            if (table.name == "roofline")
                                return &table;
                        }
                        return nullptr;
                    }

                    /// @brief The folder the ceilings file lives in.
                    std::filesystem::path folder_;

                    /// @brief What the session recorded.
                    SushiProfiler::RunRecord record_;

                    /// @brief What its analyses made of the record.
                    std::vector<SushiProfiler::Table> tables_;
            };
        } // namespace

        /** @brief Runs the end-to-end case on HALF operands, which only a device with fp16 computes. */
        class Medium_ProfilingEndToEndHalf : public Medium_ProfilingEndToEnd
        {
            protected:
                void SetUp() override
                {
                    if (!TestEnvironment::queue(0).get_device().has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device computes no half precision";
                    Medium_ProfilingEndToEnd::SetUp();
                }
        };

        TEST_F(Medium_ProfilingEndToEnd, AGemmSpanCarriesFlopsBytesAndAType)
        {
            record(2);

            const std::optional<std::pair<const SushiProfiler::PeriodRecord*, std::size_t>> gemm =
                span_with_flops(2 * 64 * 64 * 64);
            ASSERT_TRUE(gemm.has_value()) << "no span carries the FLOPs of the GEMM; the record holds "
                                          << record_summary();
            const SushiProfiler::PeriodRecord& period = *gemm->first;
            const std::size_t row = gemm->second;
            const SushiProfiler::AttributeColumn* dtype =
                period.spans.attribute(record_.find_name("dtype"));
            ASSERT_NE(dtype, nullptr);
            EXPECT_EQ(attribute_of(period, row, "bytes"), std::int64_t{3 * 64 * 64 * 4});
            EXPECT_EQ(record_.name_of(static_cast<SushiProfiler::NameID>(dtype->integers[row])),
                      "f32");
        }

        TEST_F(Medium_ProfilingEndToEnd, SomeRooflineRowIsJudgedAgainstACeiling)
        {
            record(2);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            ASSERT_FALSE(table->rows.empty());
            const std::size_t bound = table->column("bound_flops_per_s");
            ASSERT_LT(bound, table->columns.size());
            EXPECT_TRUE(std::any_of(table->rows.begin(), table->rows.end(),
                                    [bound](const SushiProfiler::Row& row)
                                    { return !row[bound].empty(); }))
                << "no row was judged against a ceiling";
        }

        TEST_F(Medium_ProfilingEndToEnd, EveryRooflineRowHasABound)
        {
            record(2);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            const std::size_t limit = table->column("limit");
            const std::size_t share = table->column("bound_share");
            ASSERT_LT(limit, table->columns.size());
            ASSERT_LT(share, table->columns.size());
            std::for_each(table->rows.begin(), table->rows.end(),
                          [limit, share](const SushiProfiler::Row& row)
                          { expect_row_has_a_bound(row, limit, share); });
        }

        TEST_F(Medium_ProfilingEndToEnd, EachActivePeriodHasOneRooflineRowPerCostedNode)
        {
            record(2);

            // The GEMM and the add are the two costed nodes of a period.
            EXPECT_EQ(roofline_rows_per_period(), (std::vector<std::size_t>{2, 2}));
        }

        TEST_F(Medium_ProfilingEndToEndHalf, AnF16SpanIsJudgedAgainstTheF16Ceiling)
        {
            record(2, Core::DataType::Half, false);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            const std::size_t bound = table->column("bound_flops_per_s");
            ASSERT_LT(bound, table->columns.size());
            ASSERT_FALSE(table->rows.empty());
            const double* judged = table->rows.front()[bound].as_real();
            ASSERT_NE(judged, nullptr);
            EXPECT_LE(*judged, 4.0e12) << "the bound of an f16 row never exceeds the f16 ceiling";
            EXPECT_GT(*judged, 1.0e12) << "an f16 row is judged against the f16 ceiling, not the f32 one";
        }
    } // namespace Test
} // namespace SushiBLAS
