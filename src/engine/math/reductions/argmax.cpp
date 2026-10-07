/**************************************************************************/
/* argmax.cpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file argmax.cpp
 * @brief Defines ReductionOps::argmax, which records the argmax reduction.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event ReductionOps::argmax(const Tensor& t, Tensor& result)
    {
        Dispatch::require_min_elements(t, 1, "math.argmax", "t");

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
