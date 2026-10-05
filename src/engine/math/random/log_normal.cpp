/****************************************************************/
/* log_normal.cpp                                               */
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

    sycl::event RandomOps::log_normal(Tensor& t, double mean, double stddev)
    {
        return Dispatch::execute_random(engine_, t, "random.log_normal", "random.log_normal"_op, {mean, stddev},
            [mean, stddev](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for log_normal operation.");
                    return sycl::event();
                }
                else
                {
                    T m = static_cast<T>(mean), sd = static_cast<T>(stddev);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        return sycl::exp(Kernels::SYCL::Random::draw_normal<T>(seed, offset, i, m, sd));
                    });
                }
            });
    }
} // namespace SushiBLAS
