/****************************************************************/
/* logsumexp.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file logsumexp.cpp
 * @brief Defines ReductionOps::logsumexp, which records the log-sum-exp reduction.
 * @author Mustafa Garip
 */

#include <limits>
#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::logsumexp(const Tensor& t, Tensor& result) 
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
                    T max_val = pT[0];
                    for (int64_t i = 1; i < size; ++i) 
                    {
                        if (pT[i] > max_val) max_val = pT[i];
                    }
                    T sum_exp = 0;
                    for (int64_t i = 0; i < size; ++i) 
                    {
                        sum_exp += sycl::exp(pT[i] - max_val);
                    }
                    *pR = max_val + sycl::log(sum_exp);
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.logsumexp", "math.logsumexp"_op, dispatch);
    }
} // namespace SushiBLAS
