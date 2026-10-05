/****************************************************************/
/* poisson.cpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        int32_t k = 0;
                        double running_p = 1.0;
                        for (int attempt = 0; attempt < MAX_ITERS; ++attempt)
                        {
                            uint64_t draw_index = i * static_cast<uint64_t>(MAX_ITERS) + static_cast<uint64_t>(attempt);
                            double u = Kernels::SYCL::Random::draw_uniform<double>(seed, offset, draw_index, 0.0, 1.0);
                            running_p *= u;
                            if (running_p <= L) break;
                            ++k;
                        }
                        return static_cast<T>(k);
                    });
                }
            });
    }
} // namespace SushiBLAS
