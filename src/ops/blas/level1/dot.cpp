/**************************************************************************/
/* dot.cpp                                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
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

#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/ops/blas/utils.hpp>
#include <SushiBLAS/ops/blas/level1.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include "level1_internal.hpp"

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event Level1::dot(const Tensor& x, const Tensor& y, Tensor& result) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in DOT.");
        // Note: For complex numbers dotc returns complex type. Result tensor should be complex.
        
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; 
        Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "DOT requires tensors of the same number of elements.");

        // TODO: Implement multi-dimensional batch support for Level 1 DOT

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* read_y = y.storage ? y.storage->data_ptr : nullptr;
        void* write_r = result.storage ? result.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        if (read_y) reads.push_back(read_y);
        std::vector<void*> writes = {};
        if (write_r) writes.push_back(write_r);

        return Internal::execute_level1(engine_, "blas.lvl1.dot", "blas.lvl1.dot"_op, x.dtype, reads, writes, {},
            [n, incx, incy, pX=read_x, pY=read_y, pR=write_r](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                if constexpr (std::is_same_v<T, sycl::half>) 
                {
                    SB_THROW_IF(true, "MKL Level 1 BLAS does not support HALF precision natively.");
                    return sycl::event();
                } 
                else 
                {
                    SB_LOG_INFO("MKL DOT: {} elements", n);
                    if constexpr (Internal::is_complex_v<T>) 
                        return oneapi::mkl::blas::column_major::dotc(q, n, static_cast<const T*>(pX), incx, static_cast<const T*>(pY), incy, static_cast<T*>(pR), deps);
                    else 
                        return oneapi::mkl::blas::column_major::dot(q, n, static_cast<const T*>(pX), incx, static_cast<const T*>(pY), incy, static_cast<T*>(pR), deps);
                }
            });
    }
} // namespace SushiBLAS