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
 * @brief Runs a task whose body throws, reads the error execute() returns and runs again.
 * @author Mustafa Garip
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <SushiBLAS/SushiBLAS.h>

namespace
{
    /// @brief Holds the sentence the failing task throws.
    constexpr const char* K_FAILURE = "the consumer's kernel refused to run";

    /** @brief Records a task of the caller's own that writes target and throws when it runs. */
    void record_failing_task(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& target)
    {
        using namespace SushiRuntime::Literals;

        SushiBLAS::TaskDescription description;
        description.name = "run_failure_demo.failing";
        description.operation = "run_failure_demo.failing"_op;

        SushiBLAS::OperandSet operands;
        operands.writes(target);

        engine.recorder().record_pair(
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
    SushiBLAS::IO io(engine);

    const std::vector<float> host_x{1.0f, 2.0f, 3.0f, 4.0f};
    SushiBLAS::Tensor x = engine.create_tensor({4});
    SushiBLAS::Tensor y = engine.create_tensor({4});
    io.from_vector(x, host_x);
    io.from_vector(y, std::vector<float>(4, 0.0f));

    // Recording succeeds: the body has not run, so nothing has thrown yet.
    record_failing_task(engine, y);

    const SushiRuntime::Result<void> failed = engine.execute();
    if (failed)
    {
        std::cerr << "execute: the run succeeded, and the task was meant to fail it\n";
        return 1;
    }

    const SushiRuntime::Error& error = failed.error();
    std::cout << "execute: " << SushiRuntime::to_string(error.code) << ": " << error.message
              << "\n";
    const bool reported = error.code == SushiRuntime::ErrorCode::KernelFailed &&
                          error.message.find(K_FAILURE) != std::string::npos;

    // The failed run emptied the graph, so this execute has nothing to run and succeeds.
    const bool nothing_left = engine.execute().has_value();

    SushiBLAS::BLASOps(engine).axpy(2.0, x, y);
    const SushiRuntime::Result<void> ran = engine.execute();
    if (!ran)
    {
        std::cerr << "execute after the failure: " << SushiRuntime::to_string(ran.error().code)
                  << ": " << ran.error().message << "\n";
        return 1;
    }

    const bool survived = io.to_vector<float>(y) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f};
    std::cout << "execute after the failure: ok, y = 2 * x\n";

    const bool correct = reported && nothing_left && survived;
    std::cout << "run_failure_demo: " << (correct ? "ok" : "MISMATCH") << "\n";
    return correct ? 0 : 1;
}
