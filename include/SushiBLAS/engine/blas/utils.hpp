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
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        // `is_complex` / `is_complex_v` were declared here too, identically to two copies
        // under src/. They now have one home, <SushiBLAS/core/traits.hpp>, which this header
        // includes so anything that reached the trait through utils.hpp still compiles -
        // spell it `Core::is_complex_v<T>`.

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
         * The body is @ref Internal::execute_typed_task, which this forwards to with the
         * family string "BLAS operation" - the one thing about the recorder that was ever
         * BLAS-specific. This name is kept because ~40 call sites read better for it and
         * because a BLAS op should not have to name its own family at every site.
         *
         * @tparam Dom Which element types to instantiate @p op_func for, in exactly the sense
         *         `Dispatch::Domain` defines for the math and logic families. It defaults to
         *         REAL_AND_COMPLEX because that is what every built-in BLAS *arithmetic* op
         *         wants and has always had; the parameter exists for `Dispatch::execute_gemm`,
         *         whose functor is written by the *caller* and is therefore no more likely to
         *         be well-formed for `std::complex` than an elementwise lambda is, and for
         *         `copy`, which moves elements without doing arithmetic to them and so passes
         *         Domain::ALL. Without it, a consumer's real-only GEMM epilogue is a compile
         *         error in the two arms that will never run.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param meta The task metadata to record.
         * @param reads Regions the task reads.
         * @param writes Regions the task writes.
         * @param dtype The operands' runtime element type.
         * @param op_func A callable `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`,
         * generic over the runtime dtype via `decltype(scalar_type)`.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         * @throws std::runtime_error at record time if @p dtype falls outside @p Dom.
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_blas_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, Func&& op_func)
        {
            return execute_typed_task<Dom>(engine, meta, reads, writes, dtype, "BLAS operation",
                                           std::forward<Func>(op_func));
        }

        // `Internal::dispatch_with_mode` used to live here and now lives in
        // <SushiBLAS/engine/task.hpp>, which this header includes - so every call
        // site that reached it through utils.hpp still does, unqualified and
        // unchanged. It moved for the same reason the recorder did: the spatial
        // domain needs the FORCE_REFERENCE / normal / VERIFY triad verbatim, and
        // "BLAS utilities" is the wrong home for something two domains depend on.
    }
}
