/**************************************************************************/
/* scal.cpp                                                               */
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

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event scal_dispatch(sycl::queue& queue, int64_t n, T alpha, T* x, int64_t incx, const std::vector<sycl::event>& deps) 
        {
            SB_LOG_INFO("MKL SCAL: {} elements", n);
            return oneapi::mkl::blas::column_major::scal(queue, n, alpha, x, incx, deps);
        }
    }

    sycl::event Level1::scal(float alpha, Tensor& x) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        // TODO: Implement multi-dimensional batch support for Level 1 SCAL

        void* write_x = x.storage ? x.storage->data_ptr : nullptr;
        std::vector<void*> reads = {};
        std::vector<void*> writes = {};
        if (write_x) writes.push_back(write_x);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_scal";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.scal"_op;
        meta.set_param(0, alpha);

        switch (x.dtype) 
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha, incx, px=x.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return scal_dispatch<float>(q, n, alpha, px, incx, deps);
                    });
                break;
            case Core::DataType::FLOAT64: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_d=static_cast<double>(alpha), incx, px=x.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return scal_dispatch<double>(q, n, alpha_d, px, incx, deps);
                    });
                break;
            case Core::DataType::COMPLEX32: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_c=std::complex<float>(alpha, 0.0f), incx, px=x.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return scal_dispatch<std::complex<float>>(q, n, alpha_c, px, incx, deps);
                    });
                break;
            case Core::DataType::COMPLEX64: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_c=std::complex<double>(alpha, 0.0), incx, px=x.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return scal_dispatch<std::complex<double>>(q, n, alpha_c, px, incx, deps);
                    });
                break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for SCAL.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS

