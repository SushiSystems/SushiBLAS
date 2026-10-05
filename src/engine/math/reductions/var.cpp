/****************************************************************/
/* var.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file var.cpp
 * @brief Defines ReductionOps::var, which records the variance reduction.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::var(const Tensor& t, Tensor& result) 
    {
        auto dispatch = [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                           [[maybe_unused]] auto scratch,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.single_task([=]() 
                {
                    if (size == 0) return;
                    T mean = 0;
                    T M2 = 0;
                    for (int64_t i = 0; i < size; ++i) 
                    {
                        T val = pT[i];
                        T delta = val - mean;
                        mean += delta / static_cast<T>(i + 1);
                        T delta2 = val - mean;
                        M2 += delta * delta2;
                    }
                    *pR = M2 / static_cast<T>(size); // Population variance
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.var", "math.var"_op, dispatch);
    }
} // namespace SushiBLAS
