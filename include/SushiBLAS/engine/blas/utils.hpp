/**************************************************************************/
/* utils.hpp                                                              */
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

#include <vector>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        template<typename T> struct is_complex : std::false_type {};
        template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

        /**
         * @brief Helper to extract 1D vector parameters (size and increment) from a Tensor.
         *
         * @param t The input tensor.
         * @param n Output: Number of elements.
         * @param inc Output: Increment (stride).
         */
        inline void get_vec_params(const Tensor& t, int64_t& n, int64_t& inc)
        {
            n = t.num_elements;
            if (t.is_contiguous())
            {
                inc = 1;
            }
            else
            {
                // TODO: Support rank > 1 tensors by allowing 2D matrix row/column slices (extracting correct strides).
                SB_THROW_IF(t.rank > 1, "Level 1 BLAS expects contiguous memory for rank > 1 tensors.");
                inc = t.strides[0];
            }
        }

        /**
         * @brief Registers a single generically-dispatched BLAS task (Level 2/3) on the engine's
         * task graph. This is the level1_internal.hpp::execute_level1 shape (one add_task, one
         * switch(dtype) dispatching a generically-templated closure) generalized for ops with
         * matrix operands. Unsupported dtypes fail loudly (P1 corollary), never silently no-op.
         *
         * @param op_func A callable `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`,
         * generic over the runtime dtype via `decltype(scalar_type)`.
         */
        template<typename Func>
        sycl::event execute_blas_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, Func&& op_func)
        {
            engine.get_graph().add_task(meta, reads, writes,
                [dtype, op_func, name = meta.name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    switch (dtype)
                    {
                        case Core::DataType::HALF:
                            return op_func(sycl::half{}, q, deps);
                        case Core::DataType::FLOAT32:
                            return op_func(float{}, q, deps);
                        case Core::DataType::FLOAT64:
                            return op_func(double{}, q, deps);
                        case Core::DataType::COMPLEX32:
                            return op_func(std::complex<float>{}, q, deps);
                        case Core::DataType::COMPLEX64:
                            return op_func(std::complex<double>{}, q, deps);
                        default:
                            SB_THROW_IF(true, "Unsupported data type for {}", name);
                    }
                    return sycl::event();
                });
            return sycl::event();
        }

        /**
         * @brief Collapses the FORCE_REFERENCE / normal / VERIFY execution-mode triad that BLAS
         * ops used to hand-write at every call site (5x in gemm.cpp, 5x in syrk.cpp).
         *
         * @param reference_fn A callable `() -> void` running the pure-C++ reference kernel.
         * @param sycl_fn A callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`
         * running the SYCL/MKL kernel.
         */
        template<typename ReferenceFn, typename SyclFn>
        sycl::event dispatch_with_mode(ExecutionMode mode, sycl::queue& q, const std::vector<sycl::event>& deps,
                                        ReferenceFn&& reference_fn, SyclFn&& sycl_fn)
        {
            if (mode == ExecutionMode::FORCE_REFERENCE)
            {
                sycl::event::wait(deps);
                reference_fn();
                return sycl::event();
            }

            auto ev = sycl_fn(q, deps);

            if (mode == ExecutionMode::VERIFY)
            {
                ev.wait();
                reference_fn();
            }
            return ev;
        }
    }
}
