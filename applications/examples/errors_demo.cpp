/**************************************************************************/
/* errors_demo.cpp                                                        */
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
 * @file errors_demo.cpp
 * @brief Provokes each record-time exception and a file status, and tests the result of a run.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <SushiBLAS/SushiBLAS.hpp>

namespace
{
    /** @brief Names what a record call or a creation did. */
    enum class Outcome : std::uint8_t
    {
        Returned,        /**< The call returned. */
        Unsupported,     /**< The mode or the device cannot run the operation. */
        InvalidArgument, /**< An operand, a flag or a scalar is wrong. */
        CreationFailed,  /**< A tensor could not be made, or the runtime refused a recording. */
    };

    /**
     * @brief Runs call, prints what it threw under label and returns the outcome.
     * @tparam Call Callable with no argument.
     */
    template<typename Call>
    [[nodiscard]] Outcome outcome_of(const char* label, const Call& call)
    {
        try
        {
            call();
        }
        catch (const SushiBLAS::UnsupportedExecution& error)
        {
            std::cout << label << ": UnsupportedExecution: " << error.what() << "\n";
            return Outcome::Unsupported;
        }
        catch (const std::invalid_argument& error)
        {
            std::cout << label << ": std::invalid_argument: " << error.what() << "\n";
            return Outcome::InvalidArgument;
        }
        catch (const std::runtime_error& error)
        {
            std::cout << label << ": std::runtime_error: " << error.what() << "\n";
            return Outcome::CreationFailed;
        }

        return Outcome::Returned;
    }
} // namespace

int main()
{
    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
    SushiBLAS::Engine engine(runtime);
    SushiRuntime::Graph graph = runtime.graph();
    SushiBLAS::LateCountHold late_counts(runtime);
    SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), late_counts,
                                     engine.mode());
    SushiBLAS::BLASOps blas(recorder);
    SushiBLAS::IO io;

    SushiBLAS::Tensor x = engine.create_tensor({4});
    SushiBLAS::Tensor y = engine.create_tensor({4});
    SushiBLAS::Tensor shorter = engine.create_tensor({3});
    SushiBLAS::Tensor on_device = engine.create_tensor(
        {4}, SushiBLAS::Core::DataType::Float32, SushiRuntime::Residency::Device);
    io.from_vector(x, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    io.from_vector(y, std::vector<float>(4, 0.0f));

    // This call is recorded; the wrong operand of the next one throws before a task is added.
    blas.axpy(2.0, x, y);
    const Outcome wrong_operand = outcome_of("axpy on 4 and 3 elements", [&]()
    {
        blas.axpy(2.0, x, shorter);
    });

    // The axpy recorded above is still on the graph; clear drops it.
    graph.clear();
    const bool discarded = io.to_vector<float>(y) == std::vector<float>(4, 0.0f);

    recorder.set_mode(SushiBLAS::ExecutionMode::ForceReference);
    const Outcome wrong_mode = outcome_of("axpy on device memory under ForceReference", [&]()
    {
        blas.axpy(2.0, on_device, y);
    });
    recorder.set_mode(SushiBLAS::ExecutionMode::Auto);

    const Outcome wrong_shape = outcome_of("create_tensor({-1})", [&]()
    {
        const SushiBLAS::Tensor refused = engine.create_tensor({-1});
        static_cast<void>(refused);
    });

    blas.axpy(2.0, x, y);
    const SushiRuntime::Result<SushiRuntime::RunReport> ran = graph.run();
    if (!ran)
    {
        std::cerr << "run: " << SushiRuntime::to_string(ran.error().code) << ": "
                  << ran.error().message << "\n";
        return 1;
    }
    const bool ran_correctly =
        io.to_vector<float>(y) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f};
    std::cout << "run: ok, y = 2 * x\n";

    const SushiBLAS::IOStatus loaded = io.load_npy(y, "errors_demo_no_such_file.npy");
    std::cout << "load_npy of a missing file: " << SushiBLAS::to_string(loaded) << "\n";

    const bool correct = wrong_operand == Outcome::InvalidArgument && discarded &&
                         wrong_mode == Outcome::Unsupported &&
                         wrong_shape == Outcome::CreationFailed && ran_correctly &&
                         loaded == SushiBLAS::IOStatus::FileUnreadable;
    std::cout << (correct ? "errors_demo: ok\n" : "errors_demo: MISMATCH\n");
    return correct ? 0 : 1;
}
