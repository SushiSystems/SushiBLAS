/**************************************************************************/
/* normal.cpp                                                             */
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

    sycl::event RandomOps::normal(Tensor& t, double mean, double stddev)
    {
        return Dispatch::execute_random(engine_, t, "random.normal", "random.normal"_op, {mean, stddev},
            [mean, stddev](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                if constexpr (Core::is_complex_v<T>)
                {
                    using RealT = typename T::value_type;
                    RealT m = static_cast<RealT>(mean);
                    RealT sd = static_cast<RealT>(stddev) / static_cast<RealT>(std::sqrt(2.0));
                    RealT* pR = reinterpret_cast<RealT*>(pT);
                    int64_t n = size * 2;

                    auto ev = q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                        {
                            pR[idx[0]] = Kernels::SYCL::Random::draw_normal<RealT>(seed, offset, idx[0], m, sd);
                        });
                    });
                    return ev;
                }
                else
                {
                    T m = static_cast<T>(mean), sd = static_cast<T>(stddev);

                    auto ev = q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx)
                        {
                            pT[idx[0]] = Kernels::SYCL::Random::draw_normal<T>(seed, offset, idx[0], m, sd);
                        });
                    });
                    return ev;
                }
            });
    }
} // namespace SushiBLAS
