/****************************************************************/
/* bernoulli.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file bernoulli.cpp
 * @brief Defines RandomOps::bernoulli, which records the Bernoulli fill of a tensor.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <kernels/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::bernoulli(Tensor& t, double p)
    {
        return Dispatch::execute_random(engine_, t, "random.bernoulli", "random.bernoulli"_op, {p},
            [p](auto scalar_type, sycl::queue& q, uint64_t seed, uint64_t offset, int64_t size, auto* pT, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (Core::is_complex_v<T>)
                {
                    SB_THROW_IF(true, "Unsupported data type for bernoulli operation.");
                    return sycl::event();
                }
                else
                {
                    T pt = static_cast<T>(p);

                    return Dispatch::submit_random_fill(q, deps, size, pT, [=](uint64_t i) -> T
                    {
                        T u = Kernels::SYCL::Random::draw_uniform<T>(seed, offset, i, static_cast<T>(0.0), static_cast<T>(1.0));
                        return (u <= pt) ? static_cast<T>(1.0) : static_cast<T>(0.0);
                    });
                }
            });
    }
} // namespace SushiBLAS
