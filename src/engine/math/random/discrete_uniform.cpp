/****************************************************************/
/* discrete_uniform.cpp                                         */
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

    sycl::event RandomOps::discrete_uniform(Tensor& t, int32_t min, int32_t max)
    {
        return Dispatch::execute_random(engine_, t, "random.discrete_uniform", "random.discrete_uniform"_op, {static_cast<double>(min), static_cast<double>(max)},
            [min, max](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for discrete_uniform operation.");
                    return sycl::event();
                }
                else
                {
                    T lo = static_cast<T>(min), hi = static_cast<T>(max) + static_cast<T>(1.0);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        return sycl::floor(Kernels::SYCL::Random::draw_uniform<T>(seed, offset, i, lo, hi));
                    });
                }
            });
    }
} // namespace SushiBLAS
