/****************************************************************/
/* argmax.cpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file argmax.cpp
 * @brief Defines ReductionOps::argmax, which records the argmax reduction.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include <limits>
#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        struct ArgMaxPair
        {
            T val;
            int64_t idx;
        };

        template<typename T>
        struct ArgMaxCombine
        {
            ArgMaxPair<T> operator()(const ArgMaxPair<T>& a, const ArgMaxPair<T>& b) const
            {
                if (a.val > b.val) return a;
                if (a.val < b.val) return b;
                return (a.idx < b.idx) ? a : b;
            }
        };
    }

    sycl::event ReductionOps::argmax(const Tensor& t, Tensor& result)
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
                    T max_val = pT[0];
                    int64_t max_idx = 0;
                    for (int64_t i = 1; i < size; ++i)
                    {
                        if (pT[i] > max_val)
                        {
                            max_val = pT[i];
                            max_idx = i;
                        }
                    }
                    *pR = max_idx;
                });
            });
        };
        return Dispatch::execute_index_reduction(engine_, t, result, "math.argmax", "math.argmax"_op, dispatch);
    }
} // namespace SushiBLAS
