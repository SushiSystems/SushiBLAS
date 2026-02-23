/**************************************************************************/
/* axpy.cpp                                                               */
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
        sycl::event axpy_dispatch(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps) 
        {
            SB_LOG_INFO("MKL AXPY: {} elements", n);
            return oneapi::mkl::blas::column_major::axpy(queue, n, alpha, x, incx, y, incy, deps);
        }
    }

    sycl::event Level1::axpy(float alpha, const Tensor& x, Tensor& y) 
    {
        SB_THROW_IF(x.dtype != y.dtype, "Data type mismatch in AXPY.");
        int64_t n, incx, incy;
        Internal::get_vec_params(x, n, incx);
        int64_t ny; Internal::get_vec_params(y, ny, incy);
        SB_THROW_IF(n != ny, "AXPY requires tensors of the same number of elements.");

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_y = y.storage ? y.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        std::vector<void*> writes = {};
        if (write_y) writes.push_back(write_y);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_axpy";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.axpy"_op;
        meta.set_param(0, alpha);

        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha, incx, incy, px=x.data_as<float>(), py=y.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return axpy_dispatch<float>(q, n, alpha, px, incx, py, incy, deps);
                    });
                break;
            }
            case Core::DataType::FLOAT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_d=static_cast<double>(alpha), incx, incy, px=x.data_as<double>(), py=y.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return axpy_dispatch<double>(q, n, alpha_d, px, incx, py, incy, deps);
                    });
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_c=std::complex<float>(alpha, 0.0f), incx, incy, px=x.data_as<std::complex<float>>(), py=y.data_as<std::complex<float>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return axpy_dispatch<std::complex<float>>(q, n, alpha_c, px, incx, py, incy, deps);
                    });
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, alpha_c=std::complex<double>(alpha, 0.0), incx, incy, px=x.data_as<std::complex<double>>(), py=y.data_as<std::complex<double>>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        return axpy_dispatch<std::complex<double>>(q, n, alpha_c, px, incx, py, incy, deps);
                    });
                break;
            }
            default: 
                SB_THROW_IF(true, "Unsupported data type for AXPY.");
        }
        return sycl::event();
    }
}
