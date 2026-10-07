/**************************************************************************/
/* var.cpp                                                                */
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

    TaskHandle ReductionOps::var(const Tensor& t, Tensor& result) 
    {
        Dispatch::require_min_elements(t, 1, "math.var", "t");

        auto dispatch = [](sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                           [[maybe_unused]] auto scratch,
                           const std::vector<sycl::event>& deps)
        {
            using T = std::decay_t<decltype(*pT)>;
            using W = Reduce::AccumulationOf<T>;

            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.single_task([=]() 
                {
                    W mean = 0;
                    W M2 = 0;
                    for (int64_t i = 0; i < size; ++i) 
                    {
                        W val = static_cast<W>(pT[i]);
                        W delta = val - mean;
                        mean += delta / static_cast<W>(i + 1);
                        W delta2 = val - mean;
                        M2 += delta * delta2;
                    }
                    *pR = static_cast<T>(M2 / static_cast<W>(size)); // Population variance
                });
            });
        };
        return Dispatch::execute_reduction(recorder_, t, result, "math.var", "math.var"_op, dispatch);
    }
} // namespace SushiBLAS
