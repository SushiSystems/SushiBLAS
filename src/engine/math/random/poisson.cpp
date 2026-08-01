/**************************************************************************/
/* poisson.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include "random_internal.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::poisson(Tensor& t, double lambda)
    {
        return Internal::execute_random(engine_, t, "random.poisson", "random.poisson"_op, {lambda},
            [lambda](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Internal::is_complex_v<T>)
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
