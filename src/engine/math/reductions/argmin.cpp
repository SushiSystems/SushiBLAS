/****************************************************************/
/* argmin.cpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file argmin.cpp
 * @brief Defines ReductionOps::argmin, which records the argmin reduction.
 * @author Mustafa Garip
 */

#include <limits>
#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::argmin(const Tensor& t, Tensor& result)
    {
        // A sequential scan gives the lowest index on a tie, stored as INT64; see src/README.md.
        auto dispatch = [](sycl::queue& q, const auto* pT, int64_t* pR, int64_t size,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;

            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.single_task([=]()
                {
                    if (size == 0) return;
                    T min_val = pT[0];
                    int64_t min_idx = 0;
                    for (int64_t i = 1; i < size; ++i)
                    {
                        if (pT[i] < min_val)
                        {
                            min_val = pT[i];
                            min_idx = i;
                        }
                    }
                    *pR = min_idx;
                });
            });
        };
        return Dispatch::execute_index_reduction(engine_, t, result, "math.argmin", "math.argmin"_op, dispatch);
    }
} // namespace SushiBLAS
