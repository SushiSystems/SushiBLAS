/**************************************************************************/
/* shuffle.cpp                                                            */
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

    sycl::event RandomOps::shuffle(Tensor& t) 
    {
        auto op_id = "random.shuffle"_op;
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random.shuffle";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = op_id;

        const int64_t size = t.num_elements;
        void* ptr = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> reads = {};
        std::vector<void*> writes = {};
        if (ptr) writes.push_back(ptr);

        const uint64_t seed = engine_.get_seed();
        const uint64_t offset = engine_.get_and_increment_rng_offset();
        meta.set_param(10, seed);
        meta.set_param(11, offset);

        engine_.get_graph().add_task(meta, reads, writes,
            [size](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
            {
                (void)q;
                (void)deps;
                SB_LOG_INFO("RandomOps: shuffle ({} elements) - TODO: Implement parallel shuffle", size);
                // TODO: Implement parallel shuffle algorithm (e.g., using random keys and sorting).
                return sycl::event();
            }
        );

        return sycl::event();
    }
} // namespace SushiBLAS
