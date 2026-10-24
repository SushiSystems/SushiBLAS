/**************************************************************************/
/* run_failure_demo.cpp                                                   */
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
 * @file run_failure_demo.cpp
 * @brief Runs a task whose body throws, reads the error run() returns and runs again.
 * @author Mustafa Garip
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <SushiBLAS/SushiBLAS.hpp>

namespace
{
    /// @brief Holds the sentence the failing task throws.
    constexpr const char* K_FAILURE = "the consumer's kernel refused to run";

    /** @brief Records a task of the caller's own that writes target and throws when it runs. */
    void record_failing_task(SushiBLAS::TaskRecorder& recorder, const SushiBLAS::Tensor& target)
    {
        using namespace SushiRuntime::Literals;

        SushiBLAS::TaskDescription description;
        description.name = "run_failure_demo.failing";
        description.operation = "run_failure_demo.failing"_op;

        SushiBLAS::OperandSet operands;
        operands.writes(target);

        recorder.record_pair(
            description, operands, "Example operation",
            [](const auto&, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event
            {
                throw std::runtime_error(K_FAILURE);
            });
    }
} // namespace

int main()
{
    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
    SushiBLAS::Engine engine(runtime);
    SushiBLAS::IO io;
    SushiRuntime::Graph graph = runtime.graph();
    SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), engine.mode());

    const std::vector<float> host_x{1.0f, 2.0f, 3.0f, 4.0f};
    SushiBLAS::Tensor x = engine.create_tensor({4});
    SushiBLAS::Tensor y = engine.create_tensor({4});
    io.from_vector(x, host_x);
    io.from_vector(y, std::vector<float>(4, 0.0f));

    // Recording succeeds: the body has not run, so nothing has thrown yet.
    record_failing_task(recorder, y);

    const SushiRuntime::Result<SushiRuntime::RunReport> failed = graph.run();
    if (failed)
    {
        std::cerr << "run: the run succeeded, and the task was meant to fail it\n";
        return 1;
    }

    const SushiRuntime::Error& error = failed.error();
    std::cout << "run: " << SushiRuntime::to_string(error.code) << ": " << error.message
              << "\n";
    const bool reported = error.code == SushiRuntime::ErrorCode::KernelFailed &&
                          error.message.find(K_FAILURE) != std::string::npos;

    // A failed run leaves the failing task on the graph, so the caller clears it.
    graph.clear();
    const bool nothing_left = graph.recorder().node_count() == 0;

    SushiBLAS::BLASOps(recorder).axpy(2.0, x, y);
    const SushiRuntime::Result<SushiRuntime::RunReport> ran = graph.run();
    if (!ran)
    {
        std::cerr << "run after the failure: " << SushiRuntime::to_string(ran.error().code)
                  << ": " << ran.error().message << "\n";
        return 1;
    }

    const bool survived = io.to_vector<float>(y) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f};
    std::cout << "run after the failure: ok, y = 2 * x\n";

    const bool correct = reported && nothing_left && survived;
    std::cout << "run_failure_demo: " << (correct ? "ok" : "MISMATCH") << "\n";
    return correct ? 0 : 1;
}
