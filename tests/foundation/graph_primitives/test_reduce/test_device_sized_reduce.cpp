/**************************************************************************/
/* test_device_sized_reduce.cpp                                           */
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
 * @file test_device_sized_reduce.cpp
 * @brief Tests that reduce() with a device count gives one result on the chain and the scheduler.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include "../device_sized_harness.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests reduce() with a device count on both layers as the count changes. */
        TEST(Medium_GraphPrimitivesDeviceSized, ReduceChainMatchesSchedulerAsTheCountChanges)
        {
            using namespace DeviceSized;

            constexpr std::size_t K_CAPACITY = 70000;
            expect_chain_matches_scheduler([](Runtime& runtime)
            {
                Harness harness(runtime, K_CAPACITY);
                auto values = runtime.buffer<float>(K_CAPACITY).value();
                auto result = runtime.buffer<float>(1).value();
                auto bits = runtime.buffer<std::uint32_t>(1).value();
                for (std::size_t i = 0; i < K_CAPACITY; ++i)
                    values[i] = static_cast<float>((i * 7919u) % 1013u) * 0.001f + 0.1f;

                auto graph = runtime.graph();
                harness.add_producer(graph);
                GraphPrimitives::reduce(graph.recorder(), values, result,
                                        harness.device_count(K_CAPACITY), Sum<float>{}).value();
                const float* r = result.data();
                std::uint32_t* b = bits.data();
                // The float result reaches the digest as its bit pattern.
                graph.add(std::size_t{1}, Reads(result), Writes(bits),
                          [r, b](std::size_t)
                          {
                              std::uint32_t word = 0;
                              std::memcpy(&word, r, sizeof(word));
                              b[0] = word;
                          });
                harness.add_digest(graph, bits, 1);
                return harness.record(graph);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
