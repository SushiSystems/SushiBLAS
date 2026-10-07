/**************************************************************************/
/* logsumexp.cpp                                                          */
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

    TaskHandle ReductionOps::logsumexp(const Tensor& t, Tensor& result) 
    {
        Dispatch::require_min_elements(t, 1, "math.logsumexp", "t");

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
                    W max_val = static_cast<W>(pT[0]);
                    for (int64_t i = 1; i < size; ++i) 
                    {
                        if (static_cast<W>(pT[i]) > max_val) max_val = static_cast<W>(pT[i]);
                    }
                    W sum_exp = 0;
                    for (int64_t i = 0; i < size; ++i) 
                    {
                        sum_exp += sycl::exp(static_cast<W>(pT[i]) - max_val);
                    }
                    *pR = static_cast<T>(max_val + sycl::log(sum_exp));
                });
            });
        };
        return Dispatch::execute_reduction(recorder_, t, result, "math.logsumexp", "math.logsumexp"_op, dispatch);
    }
} // namespace SushiBLAS
