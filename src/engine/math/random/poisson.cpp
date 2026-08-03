/**************************************************************************/
/* poisson.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::poisson(Tensor& t, double lambda)
    {
        return Dispatch::execute_random(engine_, t, "random.poisson", "random.poisson"_op, {lambda},
            [lambda](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for poisson operation.");
                    return sycl::event();
                }
                else
                {
                    // Knuth's algorithm: multiply uniform draws until the running product drops
                    // below exp(-lambda), counting draws. Iteration count is capped as a safety
                    // valve against pathologically large lambda (each draw needs one Philox call,
                    // keyed on a per-element attempt counter so it never repeats a uniform draw
                    // used elsewhere).
                    double L = sycl::exp(-lambda);
                    constexpr int MAX_ITERS = 10000;

                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx)
                        {
                            int32_t k = 0;
                            double running_p = 1.0;
                            for (int attempt = 0; attempt < MAX_ITERS; ++attempt)
                            {
                                uint64_t draw_index = idx[0] * static_cast<uint64_t>(MAX_ITERS) + static_cast<uint64_t>(attempt);
                                double u = Kernels::SYCL::Random::draw_uniform<double>(seed, offset, draw_index, 0.0, 1.0);
                                running_p *= u;
                                if (running_p <= L) break;
                                ++k;
                            }
                            pT[idx[0]] = static_cast<T>(k);
                        });
                    });
                }
            });
    }
} // namespace SushiBLAS
