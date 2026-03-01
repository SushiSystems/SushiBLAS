/**************************************************************************/
/* logsumexp.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <limits>
#include <SushiBLAS/engine/math/reductions.hpp>
#include "red_internal.hpp"

namespace SushiBLAS 
{
    sycl::event ReductionOps::logsumexp(const Tensor& t, Tensor& result) 
    {
        // TODO: Implement a numerically stable parallel tree reduction for logsumexp to replace single_task.
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
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
        return Internal::execute_reduction(engine_, t, result, "math.logsumexp", "math.logsumexp"_op, dispatch);
    }
} // namespace SushiBLAS
