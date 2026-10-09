/**************************************************************************/
/* kernel_pair_dispatcher.hpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file kernel_pair_dispatcher.hpp
 * @brief Declares KernelPairDispatcher, which runs one kernel of a reference and SYCL pair.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Runs an operation's reference or SYCL kernel under the mode fixed at record time.
         * @see modules/engine/engine/README.md
         */
        class KernelPairDispatcher final
        {
            public:
                /** @brief Fixes the execution mode every run() routes by. */
                explicit KernelPairDispatcher(ExecutionMode mode) noexcept : mode_(mode)
                {

                }

                /**
                 * @brief Runs reference_fn on the host under ForceReference, else sycl_fn.
                 * @param reference_fn Callable `() -> void` running the reference kernel.
                 * @param sycl_fn Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
                 * @return The submitted event, or an empty one under ForceReference.
                 */
                template<typename ReferenceFn, typename SyclFn>
                sycl::event run(sycl::queue& q, const std::vector<sycl::event>& deps,
                                ReferenceFn&& reference_fn, SyclFn&& sycl_fn) const
                {
                    if (mode_ == ExecutionMode::ForceReference)
                    {
                        sycl::event::wait(deps);
                        std::forward<ReferenceFn>(reference_fn)();
                        return sycl::event();
                    }

                    return std::forward<SyclFn>(sycl_fn)(q, deps);
                }

            private:
                /// @brief The recorder's execution mode when the task was recorded.
                ExecutionMode mode_;
        };
    } // namespace Internal
} // namespace SushiBLAS
