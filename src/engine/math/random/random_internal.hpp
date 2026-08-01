/**************************************************************************/
/* random_internal.hpp                                                    */
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

#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <type_traits>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    namespace Internal 
    {
        template <typename T>
        struct is_complex : std::false_type {};
        
        template <typename T>
        struct is_complex<std::complex<T>> : std::true_type {};

        template <typename T>
        inline constexpr bool is_complex_v = is_complex<T>::value;
        
        template<typename Func>
        sycl::event execute_random(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, const std::vector<double>& params, Func&& task_func)
        {
            const int64_t size = t.num_elements;
            void* ptr = t.storage ? t.storage->data_ptr : nullptr;
            
            std::vector<void*> reads = {};
            std::vector<void*> writes = {};
            if (ptr) writes.push_back(ptr);
            
            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            for (size_t i = 0; i < params.size(); ++i) {
                meta.set_param(i, params[i]);
            }

            const uint64_t seed = engine.get_seed();
            const uint64_t offset = engine.get_and_increment_rng_offset();
            
            meta.set_param(10, static_cast<double>(seed));
            meta.set_param(11, static_cast<double>(offset));

            engine.get_graph().add_task(meta, reads, writes,
                [dtype=t.dtype, size, seed, offset, task_func, pT = ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);
                    
                    switch (dtype) {
                        case Core::DataType::FLOAT32:
                            return task_func(float{}, q, seed, offset, size, static_cast<float*>(pT), deps);
                        case Core::DataType::FLOAT64:
                            return task_func(double{}, q, seed, offset, size, static_cast<double*>(pT), deps);
                        case Core::DataType::COMPLEX32:
                            return task_func(std::complex<float>{}, q, seed, offset, size, static_cast<std::complex<float>*>(pT), deps);
                        case Core::DataType::COMPLEX64:
                            return task_func(std::complex<double>{}, q, seed, offset, size, static_cast<std::complex<double>*>(pT), deps);
                        case Core::DataType::HALF:
                            SB_THROW_IF(true, "HALF precision is not supported for RNG operations.");
                            return sycl::event();
                        default:
                            SB_THROW_IF(true, "Unsupported data type for RNG operation.");
                            return sycl::event();
                    }
                }
            );
            return sycl::event();
        }
    }
}
