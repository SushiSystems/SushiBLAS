/**************************************************************************/
/* argmax.cpp                                                             */
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

#include "red_internal.hpp"
#include <limits>
#include <SushiBLAS/engine/math/reductions.hpp>

namespace SushiBLAS 
{
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
        // TODO: Implement a parallel tree reduction for argmax to replace single_task.
        // For argmax, result tensor actually stores the index. Since red_internal assumes 
        // type T for pR, we will use a temporary array or variable, but we can also just 
        // fall back to a simpler approach or allocate a small struct.
        // For now, to keep red_internal clean and avoid type mismatches, we'll implement 
        // a simple single-threaded loop on the device, as Custom Reductions with sycl::reduction 
        // can be tricky in OpenCL vs Level-Zero backends.
        auto dispatch = []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps) 
        {
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
                    *pR = static_cast<T>(max_idx);
                });
            });
        };
        return Internal::execute_reduction(engine_, t, result, "math.argmax", "math.argmax"_op, dispatch);
    }
} // namespace SushiBLAS
