/**************************************************************************/
/* recording_demo.cpp                                                     */
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
 * @file recording_demo.cpp
 * @brief Records three operations on tensors of two residencies and runs them with one execute.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <SushiBLAS/SushiBLAS.hpp>

int main()
{
    constexpr std::int64_t K_COUNT = 6;
    constexpr float K_UNWRITTEN = -1.0f;
    const std::vector<float> host_a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> host_b{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
    SushiBLAS::Engine engine(runtime);
    SushiBLAS::IO io(engine);

    // a and result are host-addressable; b lives in device memory and is reached through IO.
    SushiBLAS::Tensor a = engine.create_tensor({K_COUNT});
    SushiBLAS::Tensor b = engine.create_tensor({K_COUNT}, SushiBLAS::Core::DataType::Float32,
                                               SushiRuntime::Residency::Device);
    SushiBLAS::Tensor result = engine.create_tensor({K_COUNT});
    SushiBLAS::Tensor total = engine.create_tensor({1});
    io.from_vector(a, host_a);
    io.from_vector(b, host_b);
    io.from_vector(result, std::vector<float>(static_cast<std::size_t>(K_COUNT), K_UNWRITTEN));

    const bool residencies_hold =
        a.storage->residency() == SushiRuntime::Residency::Shared &&
        b.storage->residency() == SushiRuntime::Residency::Device;
    const std::size_t allocated = result.storage->size_bytes();
    const bool storage_is_aligned =
        result.is_aligned() && allocated % SushiBLAS::K_STORAGE_ALIGNMENT == 0;
    std::cout << "result holds " << result.num_elements << " elements in " << allocated
              << " bytes, aligned to "
              << SushiBLAS::K_STORAGE_ALIGNMENT << "\n";

    // Each call records one task and runs nothing: a = 2 * a, result = a + b, total = sum(result).
    SushiBLAS::BLASOps(engine).scal(2.0, a);
    SushiBLAS::ElementwiseOps(engine).add(a, b, result);
    SushiBLAS::ReductionOps(engine).sum(result, total);

    const float before = result.data_as<float>()[0];
    std::cout << "before execute: result[0] = " << before << "\n";

    engine.execute().value();

    const float after = result.data_as<float>()[0];
    std::cout << "after execute:  result[0] = " << after << "\n";

    // The graph is empty after a run, so a second execute has nothing to do and succeeds.
    const bool empty_run_succeeds = engine.execute().has_value();

    std::vector<float> expected(host_a.size());
    float expected_total = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = 2.0f * host_a[i] + host_b[i];
        expected_total += expected[i];
    }

    const std::vector<float> read_back = io.to_vector<float>(result);
    const float read_total = io.to_vector<float>(total)[0];
    const bool device_unchanged = io.to_vector<float>(b) == host_b;
    std::cout << "total = " << read_total << " (expected " << expected_total << ")\n";

    const bool correct = residencies_hold && storage_is_aligned && before == K_UNWRITTEN &&
                         empty_run_succeeds && read_back == expected &&
                         read_total == expected_total && device_unchanged;
    std::cout << (correct ? "recording_demo: ok\n" : "recording_demo: MISMATCH\n");
    return correct ? 0 : 1;
}
