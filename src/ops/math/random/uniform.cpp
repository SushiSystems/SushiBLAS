/**************************************************************************/
/* uniform.cpp                                                            */
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

    sycl::event RandomOps::uniform(Tensor& t, double min, double max) 
    {
        const int64_t size = t.num_elements;
        void* write_ptr = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> writes = {};
        if (write_ptr) writes.push_back(write_ptr);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random_uniform";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.uniform"_op;

        const uint64_t seed = engine_.get_seed();
        const uint64_t offset = engine_.get_and_increment_rng_offset();

        if (t.dtype == Core::DataType::FLOAT32)
            return Internal::add_rng_task<float>(engine_, t, "random_uniform", "random.uniform"_op, oneapi::mkl::rng::uniform<float>(static_cast<float>(min), static_cast<float>(max)));
        else if (t.dtype == Core::DataType::FLOAT64)
            return Internal::add_rng_task<double>(engine_, t, "random_uniform", "random.uniform"_op, oneapi::mkl::rng::uniform<double>(min, max));
        else if (t.dtype == Core::DataType::COMPLEX32) 
        {
            engine_.get_graph().add_task(meta, {}, writes,
                [size, seed, offset, min, max, pT = reinterpret_cast<float*>(t.data_as<std::complex<float>>())](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size * 2);
                    return oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<float>(static_cast<float>(min), static_cast<float>(max)), engine_obj, size * 2, pT, deps);
                }
            );
        } 
        else if (t.dtype == Core::DataType::COMPLEX64) 
        {
            engine_.get_graph().add_task(meta, {}, writes,
                [size, seed, offset, min, max, pT = reinterpret_cast<double*>(t.data_as<std::complex<double>>())](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size * 2);
                    return oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<double>(min, max), engine_obj, size * 2, pT, deps);
                }
            );
        } 
        else
            SB_THROW_IF(true, "Unsupported data type for uniform operation.");

        return sycl::event();
    }
}
