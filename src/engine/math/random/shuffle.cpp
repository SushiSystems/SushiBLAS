/**************************************************************************/
/* shuffle.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
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
