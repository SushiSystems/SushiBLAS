/****************************************************************/
/* std.cpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <type_traits>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::std(const Tensor& t, Tensor& result) 
    {
        // TODO: Implement a parallel reduction (e.g. Chan et al. algorithm) for standard deviation to replace single_task.
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
                    *pR = sycl::sqrt(M2 / static_cast<T>(size)); // Population std
                });
            });
        };
        return Dispatch::execute_reduction(engine_, t, result, "math.std", "math.std"_op, dispatch);
    }
} // namespace SushiBLAS
