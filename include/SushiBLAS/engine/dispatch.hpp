/**************************************************************************/
/* dispatch.hpp                                                           */
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

#include <cstddef>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiRuntime/graph/task_types.hpp>

// The shared vocabulary of SushiBLAS's generic op-dispatch seam.
//
// `SushiBLAS::Dispatch` is the low-level, functor-templated layer every op in
// this library is built on: one `execute_*` helper per op family that records a
// task on the engine's graph, switches on the tensor's runtime dtype, and
// inlines a caller-supplied functor into a single SYCL kernel. The five family
// headers are:
//
//   <SushiBLAS/engine/math/elementwise/dispatch.hpp>
//   <SushiBLAS/engine/math/nonlinear/dispatch.hpp>
//   <SushiBLAS/engine/math/reductions/dispatch.hpp>
//   <SushiBLAS/engine/math/random/dispatch.hpp>
//   <SushiBLAS/engine/logic/dispatch.hpp>
//
// These used to be private `*_internal.hpp` headers under src/, uninstalled and
// therefore unreachable from outside this tree. They are public now because
// they are the seam a layer above (an autodiff/NN framework) needs in order to
// fuse: composing several math ops into one functor and handing it to
// `execute_binary` produces ONE kernel launch instead of one per op, and it does
// so without the consumer writing a line of SYCL - the functor is a plain C++
// lambda. Keeping that capability private would have forced every consumer to
// either eat a kernel launch per elementwise op or reimplement this dispatch
// layer itself.
//
// This header carries only what all five families share; include the family
// header for the executors themselves.

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief The set of element types a dispatch helper is allowed to instantiate its
         * kernel for.
         *
         * Not a runtime filter: it selects, at compile time, which `case` arms of the
         * dtype switch actually instantiate the caller's functor. That distinction matters
         * because the functor is a generic lambda whose body is only well-formed for some
         * element types - `sycl::max(x, y)` and `sycl::floor(x)`, for instance, reject
         * `std::complex`, and a generic lambda gives no SFINAE-visible signal of that
         * (return-type deduction instantiates the body, which is not the immediate
         * context). So the caller states the domain, the dispatch instantiates only the
         * arms inside it, and every arm outside it throws naming the op and the dtype.
         *
         * @warning REAL is the default deliberately: a functor written for real scalars
         *          always compiles under it. Opt in to REAL_AND_COMPLEX only when the
         *          functor is genuinely well-formed for `std::complex` - see the `safe_*`
         *          helpers in <SushiBLAS/engine/math/safe_math.hpp>.
         */
        enum class Domain
        {
            REAL,             /**< HALF, FLOAT32, FLOAT64. Complex dtypes throw. */
            REAL_AND_COMPLEX  /**< The above, plus COMPLEX32 and COMPLEX64. */
        };

        /**
         * @brief Copies a caller-supplied parameter list into a task's inline metadata slots.
         *
         * `TaskMetadata::set_param` silently ignores an out-of-range index, so a caller that
         * writes past `K_MAX_PARAMS` loses the value with no diagnostic at all. This wrapper
         * turns that into a loud failure at record time (P1: fail loudly, never silently).
         *
         * @tparam T The parameter element type; must fit in 64 bits.
         * @param meta The metadata block to populate, starting at slot 0.
         * @param params The values to store.
         * @param name The op name, used only in the failure message.
         * @param reserved_slots Number of trailing slots the caller keeps for its own use
         *        (for example the RNG seed/offset pair) and which `params` must not reach.
         */
        template<typename T>
        inline void set_task_params(SushiRuntime::Graph::TaskMetadata& meta, const std::vector<T>& params,
                                     const char* name, size_t reserved_slots = 0)
        {
            constexpr size_t k_capacity = SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS;
            SB_THROW_IF(params.size() + reserved_slots > k_capacity,
                        "Operation '{}' passes {} task parameters (+{} reserved) but TaskMetadata holds only {}; "
                        "the excess would be silently dropped.",
                        name, params.size(), reserved_slots, k_capacity);

            for (size_t i = 0; i < params.size(); ++i)
            {
                meta.set_param(static_cast<int>(i), params[i]);
            }
        }

    } // namespace Dispatch
} // namespace SushiBLAS
