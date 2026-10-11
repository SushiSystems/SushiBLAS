/**************************************************************************/
/* test_task_dependencies.cpp                                             */
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
 * @file test_task_dependencies.cpp
 * @brief Tests that a recorded task body orders its device work behind the events it is given.
 * @author Mustafa Garip
 */

#include <future>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/spatial_geometry.hpp>
#include "../common/test_utils.hpp"
#include "../common/graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds a queued event that stays incomplete until the gate is opened. */
            class DependencyGate final
            {
                public:
                    /** @brief Submits a host task to queue that blocks until open() is called. */
                    explicit DependencyGate(sycl::queue& queue)
                        : event_(queue.submit(
                              [opened = opened_.get_future().share()](sycl::handler& h)
                              {
                                  h.host_task([opened]() { opened.wait(); });
                              }))
                    {

                    }

                    DependencyGate(const DependencyGate&) = delete;
                    DependencyGate& operator=(const DependencyGate&) = delete;
                    DependencyGate(DependencyGate&&) = delete;
                    DependencyGate& operator=(DependencyGate&&) = delete;

                    /** @brief Opens the gate, so that a test leaving early blocks no host task. */
                    ~DependencyGate()
                    {
                        open();
                    }

                    /** @brief Returns the event a task body is handed as its dependency. */
                    [[nodiscard]] const sycl::event& event() const noexcept { return event_; }

                    /** @brief Lets the host task finish; a second call does nothing. */
                    void open()
                    {
                        if (is_open_)
                            return;

                        is_open_ = true;
                        opened_.set_value();
                    }

                private:
                    /// @brief Fulfilled by open(); declared first, because event_ is built from it.
                    std::promise<void> opened_;

                    /// @brief The event of the blocked host task.
                    sycl::event event_;

                    /// @brief Whether open() has fulfilled the promise.
                    bool is_open_ = false;
            };
        } // namespace

        /** @brief Calls the body of one recorded task with a dependency that has not finished. */
        class Regression_TaskDependencies : public SushiBLASTest
        {
            protected:
                /** @brief Returns a FLOAT32 vector holding values. */
                [[nodiscard]] Tensor vector_of(const std::vector<float>& values)
                {
                    Tensor tensor = engine->create_tensor({static_cast<int64_t>(values.size())},
                                                          Core::DataType::Float32);
                    fill_tensor(tensor, values);
                    return tensor;
                }

                /**
                 * @brief Expects the one task record adds to stay incomplete behind a closed gate.
                 * @param record Callable `(TaskRecorder&) -> TaskHandle` that records one task.
                 * @param verify Callable `() -> void` comparing the task's output with the host
                 *        formula; it runs after the gate is open and the task has finished.
                 */
                template<typename Record, typename Verify>
                void expect_body_waits(const char* seam, Record&& record, Verify&& verify)
                {
                    SCOPED_TRACE(seam);

                    GraphProbe probe(ExecutionMode::ForceSYCL);
                    ASSERT_TRUE(record(probe.recorder()).is_valid());

                    ASSERT_EQ(1u, probe.recorded_count());
                    const auto tasks = probe.recorded_tasks();
                    if (!tasks)
                        GTEST_SKIP() << GraphProbe::K_NO_RECORDED_TASK_VIEW;
                    ASSERT_TRUE(static_cast<bool>((*tasks)[0].body));

                    sycl::queue& queue = TestEnvironment::queue();
                    DependencyGate gate(queue);

                    sycl::event done =
                        (*tasks)[0].body(queue, std::vector<sycl::event>{gate.event()});

                    const auto status =
                        done.get_info<sycl::info::event::command_execution_status>();
                    EXPECT_TRUE(status != sycl::info::event_command_status::complete)
                        << "The body's event completed while its dependency was still running, "
                           "so a command group it submitted does not depend on the events it "
                           "was given.";

                    gate.open();
                    done.wait_and_throw();
                    verify();
                }
        };

        /** @brief Tests one operation of each recording seam against a dependency held back. */
        TEST_F(Regression_TaskDependencies, ARecordedBodyWaitsForItsDependencies)
        {
            const std::vector<float> ramp = {1.0f, 2.0f, 3.0f, 4.0f};
            const std::vector<float> tens = {10.0f, 20.0f, 30.0f, 40.0f};

            {
                auto a = vector_of(ramp);
                auto b = vector_of(tens);
                auto c = vector_of({0.0f, 0.0f, 0.0f, 0.0f});
                expect_body_waits("elementwise add",
                    [&](TaskRecorder& r) { return SushiBLAS::ElementwiseOps(r).add(a, b, c); },
                    [&] { EXPECT_TENSOR_NEAR(c, (std::vector<float>{11, 22, 33, 44}), 1e-5); });
            }

            {
                auto t = vector_of({-1.0f, 2.0f, -3.0f, 4.0f});
                expect_body_waits("nonlinear relu",
                    [&](TaskRecorder& r) { return SushiBLAS::NonLinearOps(r).relu(t); },
                    [&] { EXPECT_TENSOR_NEAR(t, (std::vector<float>{0, 2, 0, 4}), 1e-5); });
            }

            {
                auto a = vector_of(ramp);
                auto b = vector_of({1.0f, 5.0f, 3.0f, 7.0f});
                auto same = vector_of({9.0f, 9.0f, 9.0f, 9.0f});
                expect_body_waits("logic equal",
                    [&](TaskRecorder& r) { return SushiBLAS::LogicOps(r).equal(a, b, same); },
                    [&] { EXPECT_TENSOR_NEAR(same, (std::vector<float>{1, 0, 1, 0}), 1e-5); });
            }

            {
                auto t = vector_of(ramp);
                auto total = vector_of({0.0f});
                expect_body_waits("reduction sum",
                    [&](TaskRecorder& r) { return SushiBLAS::ReductionOps(r).sum(t, total); },
                    [&] { EXPECT_TENSOR_NEAR(total, (std::vector<float>{10}), 1e-5); });
            }

            {
                namespace Random = Kernels::SYCL::Random;

                constexpr std::uint64_t K_SEED = 20261007u;

                auto t = vector_of(std::vector<float>(Random::K_DRAWS_PER_BLOCK<float>, 9.0f));
                engine->rng().reseed(K_SEED);
                const std::uint64_t offset = engine->rng().reserve(0).offset;
                expect_body_waits("random uniform",
                    [&](TaskRecorder& r)
                    {
                        return SushiBLAS::RandomOps(r, engine->rng()).uniform(t, 0.0, 1.0);
                    },
                    [&]
                    {
                        const auto draws =
                            Random::uniform_block<float>(K_SEED, offset, 0, 0.0f, 1.0f);
                        const std::vector<float> expected(draws.begin(), draws.end());
                        EXPECT_TENSOR_NEAR(t, expected, 1e-6);
                    });
            }

            {
                auto x = vector_of(ramp);
                auto y = vector_of(tens);
                expect_body_waits("blas axpy",
                    [&](TaskRecorder& r) { return SushiBLAS::BLASOps(r).axpy(2.0, x, y); },
                    [&] { EXPECT_TENSOR_NEAR(y, (std::vector<float>{12, 24, 36, 48}), 1e-5); });
            }

            {
                // A 1x1 window gathers each pixel into its own row, so the columns equal the image.
                auto image = engine->create_tensor({1, 2, 2, 1}, Core::DataType::Float32);
                auto columns = engine->create_tensor({4, 1}, Core::DataType::Float32);
                fill_tensor(image, ramp);
                fill_tensor(columns, {0.0f, 0.0f, 0.0f, 0.0f});
                const Spatial::ConvGeometry unit_window;
                expect_body_waits("spatial im2col",
                    [&](TaskRecorder& r)
                    {
                        return SushiBLAS::SpatialOps(r).im2col(image, columns, unit_window);
                    },
                    [&] { EXPECT_TENSOR_NEAR(columns, ramp, 1e-5); });
            }
        }
    } // namespace Test
} // namespace SushiBLAS
