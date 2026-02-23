/**************************************************************************/
/* nrm2.cpp                                                               */
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
        template<typename T, typename ResT = T>
        sycl::event nrm2_dispatch(sycl::queue& queue, int64_t n, const T* x, int64_t incx, ResT* res, const std::vector<sycl::event>& deps) 
        {
            SB_LOG_INFO("MKL NRM2: {} elements", n);
            return oneapi::mkl::blas::column_major::nrm2(queue, n, x, incx, res, deps);
        }
    }

    sycl::event Level1::nrm2(const Tensor& x, Tensor& result) 
    {
        int64_t n, incx;
        Internal::get_vec_params(x, n, incx);

        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_r = result.storage ? result.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_x) reads.push_back(read_x);
        std::vector<void*> writes = {};
        if (write_r) writes.push_back(write_r);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "mkl_nrm2";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "blas.nrm2"_op;

        switch (x.dtype) 
        {
            case Core::DataType::FLOAT32: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, incx, px=x.data_as<float>(), pr=result.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return nrm2_dispatch<float>(q, n, px, incx, pr, deps);
                    });
                break;
            case Core::DataType::FLOAT64: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, incx, px=x.data_as<double>(), pr=result.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return nrm2_dispatch<double, double>(q, n, px, incx, pr, deps);
                    });
                break;
            case Core::DataType::COMPLEX32: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, incx, px=x.data_as<std::complex<float>>(), pr=result.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return nrm2_dispatch<std::complex<float>, float>(q, n, px, incx, pr, deps);
                    });
                break;
            case Core::DataType::COMPLEX64: 
                engine_.get_graph().add_task(meta, reads, writes,
                    [n, incx, px=x.data_as<std::complex<double>>(), pr=result.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event {
                        return nrm2_dispatch<std::complex<double>, double>(q, n, px, incx, pr, deps);
                    });
                break;
            default: 
                SB_THROW_IF(true, "Unsupported data type for NRM2.");
        }
        return sycl::event();
    }
}
