/**************************************************************************/
/* random_internal.hpp                                                    */
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

#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    namespace Internal 
    {
        /**
         * @brief Internal helper to add an RNG task to the engine's task graph.
         * Optimized to eliminate std::string allocations and use compile-time constants.
         * 
         * // TODO: Implement transform_rng_task helper to handle engine setup and 
         * // subsequent transformations (threshold, cast, floor) in a single conceptual block.
         * // TODO: Move common transformation kernels here to leverage pre-compilation/AOT.
         */
        template<typename T, typename DistType>
        sycl::event add_rng_task(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, DistType dist) 
        {
            const int64_t size = t.num_elements;
            void* write_ptr = t.storage ? t.storage->data_ptr : nullptr;
            std::vector<void*> writes = {};
            if (write_ptr) writes.push_back(write_ptr);
            
            // Empty reads for RNG
            std::vector<void*> reads = {};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            const uint64_t seed = engine.get_seed();
            const uint64_t offset = engine.get_and_increment_rng_offset();

            SB_LOG_DEBUG("Dispatching RNG Task [{}] (Op ID: {}), Size: {}, Seed: {}, Offset: {}", name, op_id.value, size, seed, offset);

            engine.get_graph().add_task(meta, reads, writes,
                [size, seed, offset, dist, pT = t.data_as<T>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    oneapi::mkl::rng::philox4x32x10 engine_obj(q, seed);
                    oneapi::mkl::rng::skip_ahead(engine_obj, offset * size);
                    return oneapi::mkl::rng::generate(dist, engine_obj, size, pT, deps);
                }
            );
            return sycl::event();
        }
    }
}
