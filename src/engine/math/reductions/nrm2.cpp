/**************************************************************************/
/* nrm2.cpp                                                               */
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

#include <SushiBLAS/engine/math/reductions.hpp>
#include "red_internal.hpp"

namespace SushiBLAS 
{
    sycl::event ReductionOps::nrm2(const Tensor& t, Tensor& result) 
    {
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
            auto e1 = q.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), sycl::reduction(pR, T(0), sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}), [=](sycl::id<1> idx, auto& reducer) 
                {
                    T val = pT[idx[0]];
                    reducer.combine(val * val);
                });
            });
            return q.submit([&](sycl::handler& h) 
            {
                h.depends_on(e1);
                h.single_task([=]() 
                {
                    *pR = sycl::sqrt(*pR);
                });
            });
        };
        return Internal::execute_reduction(engine_, t, result, "math.nrm2", "math.nrm2"_op, dispatch);
    }
} // namespace SushiBLAS
