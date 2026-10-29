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
 * @file test_profiler_parts.cpp
 * @brief Tests that register_profiler_parts adds SushiBLAS to a registry once and keeps it there.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <map>
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

            /** @brief Returns whether any fact of a record is named @p name. */
            bool has_fact(const std::vector<SushiProfiler::Fact>& facts, const std::string& name)
            {
                for (const SushiProfiler::Fact& fact : facts)
                {
                    if (fact.name == name)
                        return true;
                }
                return false;
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

            EXPECT_EQ(second_registry.contributors().size(), std::size_t{1});
        }

        TEST(Small_RegisterProfilerParts, TheContributorStatesItsFactsWhileThePartsLive)
        {
            SushiProfiler::Registry registry;
            const Profiling::SushiBLASProfilerParts parts =
                Profiling::register_profiler_parts(registry, no_devices(), "");

            const SushiProfiler::RecordFacts first = SushiProfiler::gather_facts(registry.contributors());
            const SushiProfiler::RecordFacts second = SushiProfiler::gather_facts(registry.contributors());

            EXPECT_TRUE(has_fact(first.build, "sushiblas.version"));
            EXPECT_EQ(first.build, second.build);
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
                    void record(std::uint32_t steps)
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
                        RecordingGraph graph(engine);
                        const Tensor a = engine.create_tensor({64, 64}, Core::DataType::Float32);
                        const Tensor b = engine.create_tensor({64, 64}, Core::DataType::Float32);
                        Tensor c = engine.create_tensor({64, 64}, Core::DataType::Float32);
                        Tensor d = engine.create_tensor({64, 64}, Core::DataType::Float32);
                        SushiProfiler::PeriodTimeline& timeline = started.session->timeline("blas.step");
                        for (std::uint32_t step = 0; step < steps; ++step)
                        {
                            timeline.mark();
                            BLASOps(graph.recorder()).gemm(a, b, c);
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

        TEST_F(Medium_ProfilingEndToEnd, AGemmSpanCarriesFlopsBytesAndAType)
        {
            record(2);

            bool found = false;
            for (const SushiProfiler::PeriodRecord& period : record_.periods)
            {
                for (std::size_t row = 0; row < period.spans.size(); ++row)
                {
                    if (attribute_of(period, row, "flops") == 2 * 64 * 64 * 64)
                    {
                        found = true;
                        EXPECT_EQ(attribute_of(period, row, "bytes"), std::int64_t{3 * 64 * 64 * 4});
                        const SushiProfiler::AttributeColumn* dtype =
                            period.spans.attribute(record_.find_name("dtype"));
                        ASSERT_NE(dtype, nullptr);
                        EXPECT_EQ(record_.name_of(static_cast<SushiProfiler::NameID>(dtype->integers[row])),
                                  "f32");
                    }
                }
            }
            EXPECT_TRUE(found) << "no span carries the FLOPs of the GEMM; the record holds "
                               << record_summary();
        }

        TEST_F(Medium_ProfilingEndToEnd, TheRooflineRowNamesTheCeiling)
        {
            record(2);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            ASSERT_FALSE(table->rows.empty());
            const std::size_t bound = table->column("bound_flops_per_s");
            ASSERT_LT(bound, table->columns.size());
            bool judged = false;
            for (const SushiProfiler::Row& row : table->rows)
                judged = judged || !row[bound].empty();
            EXPECT_TRUE(judged) << "no row was judged against a ceiling";
        }

        TEST_F(Medium_ProfilingEndToEnd, TheRooflineRowHasABound)
        {
            record(2);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            const std::size_t limit = table->column("limit");
            const std::size_t share = table->column("bound_share");
            ASSERT_LT(limit, table->columns.size());
            ASSERT_LT(share, table->columns.size());
            for (const SushiProfiler::Row& row : table->rows)
            {
                ASSERT_NE(row[limit].as_text(), nullptr);
                EXPECT_TRUE(*row[limit].as_text() == "compute" || *row[limit].as_text() == "memory");
                ASSERT_NE(row[share].as_real(), nullptr);
                EXPECT_GT(*row[share].as_real(), 0.0);
            }
        }

        TEST_F(Medium_ProfilingEndToEnd, AnUncostedOperationHasNoRow)
        {
            record(2);

            const SushiProfiler::Table* table = roofline();
            ASSERT_NE(table, nullptr);
            const std::size_t period = table->column("period");
            ASSERT_LT(period, table->columns.size());
            std::map<std::int64_t, std::size_t> rows_per_period;
            for (const SushiProfiler::Row& row : table->rows)
            {
                ASSERT_NE(row[period].as_whole(), nullptr);
                ++rows_per_period[*row[period].as_whole()];
            }
            EXPECT_EQ(rows_per_period.size(), std::size_t{2}) << "one group of rows per active period";
            for (const auto& [index, rows] : rows_per_period)
                EXPECT_EQ(rows, std::size_t{2})
                    << "period " << index << ": one row per costed node, the GEMM and the add";
        }

        TEST_F(Medium_ProfilingEndToEnd, AnF16SpanIsJudgedAgainstTheF16Ceiling)
        {
            SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
            if (!TestEnvironment::queue(0).get_device().has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device computes no half precision";
            const std::string device = runtime.advanced().devices().front().name;
            Profiling::DeviceCeilings ceilings;
            ceilings.device_name = device;
            ceilings.backend = "test";
            ceilings.driver = "1";
            ceilings.smoke_only = true;
            ceilings.method = {1, 3, 1.0, 1.0};
            ceilings.compute = {{"f32", 1.0e12, 0.0, 3}, {"f16", 4.0e12, 0.0, 3}};
            ceilings.bandwidth = {1.0e11, 0.0, 3};
            std::ofstream(folder_ / Profiling::ceilings_file_name(device), std::ios::binary)
                << Profiling::write_ceilings(ceilings);

            SushiProfiler::Registry registry;
            SushiRuntime::RuntimeProfilerParts runtime_parts(registry, runtime);
            const Profiling::SushiBLASProfilerParts blas_parts =
                Profiling::register_profiler_parts(registry, runtime, folder_.string());
            SushiProfiler::ProfilerConfig config;
            config.timeline = "blas.step";
            config.window.warmup = 0U;
            config.window.active = 2U;
            config.sources = {SushiProfiler::NamedSettings{"sushiruntime.device",
                                                          SushiProfiler::JsonValue::object()}};
            config.analyses = {"roofline"};
            config.buffers.linger_ms = 0U;
            SushiProfiler::SessionStart started = SushiProfiler::Session::start(config, registry);
            ASSERT_TRUE(started.ok()) << started.error;

            Engine engine(runtime);
            RecordingGraph graph(engine);
            const Tensor a = engine.create_tensor({64, 64}, Core::DataType::Half);
            const Tensor b = engine.create_tensor({64, 64}, Core::DataType::Half);
            Tensor c = engine.create_tensor({64, 64}, Core::DataType::Half);
            SushiProfiler::PeriodTimeline& timeline = started.session->timeline("blas.step");
            for (int step = 0; step < 2; ++step)
            {
                timeline.mark();
                BLASOps(graph.recorder()).gemm(a, b, c);
                graph.run().value();
            }
            timeline.mark();
            SushiProfiler::SessionStop stopped = started.session->stop();
            const std::vector<SushiProfiler::Table> tables = started.session->report(stopped.record).tables;

            const SushiProfiler::Table* table = nullptr;
            for (const SushiProfiler::Table& candidate : tables)
            {
                if (candidate.name == "roofline")
                    table = &candidate;
            }
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
