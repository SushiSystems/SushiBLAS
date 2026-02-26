/**************************************************************************/
/* poisson.cpp                                                            */
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

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include "random_internal.hpp"

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::poisson(Tensor& t, double lambda) 
    {
        const int64_t size = t.num_elements;
        void* write_ptr = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> writes = {};
        if (write_ptr) writes.push_back(write_ptr);
        
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random_poisson";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.poisson"_op;

        const uint64_t seed = engine_.get_seed();
        const uint64_t offset = engine_.get_and_increment_rng_offset();

        // MKL Poisson only supports integer types. We generate into a temporary 
        // integer buffer and then cast to floating point.
        // TODO: Move sycl::malloc_device and sycl::free out of the task lambda to avoid pipeline stalls.
        // TODO: Refactor to use a specialized cast kernel that can be reused across distributions.
        if (t.dtype == Core::DataType::FLOAT32)
        {
            engine_.get_graph().add_task(meta, {}, writes,
                [size, seed, offset, lambda, pT = t.data_as<float>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    std::int32_t* tmp = sycl::malloc_device<std::int32_t>(size, q);
                    oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size);
                    
                    auto ev = oneapi::mkl::rng::generate(oneapi::mkl::rng::poisson<std::int32_t>(lambda), engine_obj, size, tmp, deps);
                    
                    auto cast_ev = q.submit([&](sycl::handler& h) {
                        h.depends_on(ev);
                        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                            pT[idx] = static_cast<float>(tmp[idx]);
                        });
                    });

                    q.submit([&](sycl::handler& h) {
                        h.depends_on(cast_ev);
                        h.host_task([=]() {
                            sycl::free(tmp, q);
                        });
                    });
                    
                    return cast_ev;
                }
            );
        }
        else if (t.dtype == Core::DataType::FLOAT64)
        {
             engine_.get_graph().add_task(meta, {}, writes,
                [size, seed, offset, lambda, pT = t.data_as<double>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    std::int32_t* tmp = sycl::malloc_device<std::int32_t>(size, q);
                    oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size);
                    
                    auto ev = oneapi::mkl::rng::generate(oneapi::mkl::rng::poisson<std::int32_t>(lambda), engine_obj, size, tmp, deps);
                    
                    auto cast_ev = q.submit([&](sycl::handler& h) {
                        h.depends_on(ev);
                        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                            pT[idx] = static_cast<double>(tmp[idx]);
                        });
                    });

                    q.submit([&](sycl::handler& h) {
                        h.depends_on(cast_ev);
                        h.host_task([=]() {
                            sycl::free(tmp, q);
                        });
                    });
                    
                    return cast_ev;
                }
            );
        }
        else
        {
            SB_THROW_IF(true, "Unsupported data type for poisson operation.");
        }
        return sycl::event();
    }
}
