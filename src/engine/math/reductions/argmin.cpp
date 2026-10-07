/**************************************************************************/
/* argmin.cpp                                                             */
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
        Dispatch::require_min_elements(t, 1, "math.argmin", "t");

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
