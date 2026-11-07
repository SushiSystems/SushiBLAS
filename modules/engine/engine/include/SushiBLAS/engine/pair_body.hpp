/**************************************************************************/
/* pair_body.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file pair_body.hpp
 * @brief Declares the calls that run a recorded body in its current shape or in the shape it had before the queue moved into KernelPairDispatcher.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /// @brief Whether Body takes `(const KernelPairDispatcher&, const std::vector<sycl::event>&)`.
            template<typename Body>
            inline constexpr bool K_TAKES_HELD_QUEUE =
                std::is_invocable_v<const Body&, const KernelPairDispatcher&,
                                    const std::vector<sycl::event>&>;

            /**
             * @brief Names what a recorded body returns for a dispatcher and dependencies, in
             *        whichever shape it has.
             *
             * It is a trait, not a deduced return type, so a call inside a switch does not
             * instantiate a generic body while the switch is parsed (KI-20).
             */
            template<typename Body, bool Current = K_TAKES_HELD_QUEUE<Body>>
            struct PairBodyResult
            {
                /// @brief The result of the old shape, which takes the queue.
                using type = std::invoke_result_t<const Body&, const KernelPairDispatcher&,
                                                  sycl::queue&, const std::vector<sycl::event>&>;
            };

            /// @brief The result of the current shape.
            template<typename Body>
            struct PairBodyResult<Body, true>
            {
                /// @brief The result of the current shape, which takes no queue.
                using type = std::invoke_result_t<const Body&, const KernelPairDispatcher&,
                                                  const std::vector<sycl::event>&>;
            };

            /**
             * @brief Runs a recorded body with its dispatcher and dependencies.
             *
             * A body of the current shape gets `(dispatcher, deps)`; one of the old shape
             * `(dispatcher, sycl::queue&, deps)` gets the dispatcher's queue.
             * @deprecated The old body shape is removed in H5 of the test review plan.
             * @throws UnsupportedExecution for an old-shape body when the dispatcher holds no queue.
             */
            template<typename Body>
            [[nodiscard]] typename PairBodyResult<Body>::type call_pair_body(
                const Body& body, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps)
            {
                if constexpr (K_TAKES_HELD_QUEUE<Body>)
                    return body(dispatcher, deps);
                else
                    return body(dispatcher, dispatcher.queue(), deps);
            }

            /**
             * @brief Names what a dtype-dispatched body returns for a value of its element type.
             *
             * A trait for the reason PairBodyResult is one.
             */
            template<typename Func, typename T,
                     bool Current = std::is_invocable_v<const Func&, T, const KernelPairDispatcher&,
                                                        const std::vector<sycl::event>&>>
            struct ScalarBodyResult
            {
                /// @brief The result of the old shape, which takes the queue.
                using type = std::invoke_result_t<const Func&, T, const KernelPairDispatcher&,
                                                  sycl::queue&, const std::vector<sycl::event>&>;
            };

            /// @brief The result of the current shape.
            template<typename Func, typename T>
            struct ScalarBodyResult<Func, T, true>
            {
                /// @brief The result of the current shape, which takes no queue.
                using type = std::invoke_result_t<const Func&, T, const KernelPairDispatcher&,
                                                  const std::vector<sycl::event>&>;
            };

            /**
             * @brief Runs a dtype-dispatched body with a value of its element type.
             *
             * A body of the current shape gets `(T, dispatcher, deps)`; one of the old shape
             * `(T, dispatcher, sycl::queue&, deps)` gets the queue the dispatcher holds.
             * @deprecated The old body shape is removed in H5, as for call_pair_body().
             * @throws UnsupportedExecution for an old-shape body when the dispatcher holds no queue.
             */
            template<typename Func, typename T>
            [[nodiscard]] typename ScalarBodyResult<Func, T>::type call_scalar_body(
                const Func& body, T zero, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps)
            {
                if constexpr (std::is_invocable_v<const Func&, T, const KernelPairDispatcher&,
                                                  const std::vector<sycl::event>&>)
                    return body(zero, dispatcher, deps);
                else
                    return body(zero, dispatcher, dispatcher.queue(), deps);
            }
        } // namespace Detail
    } // namespace Internal
} // namespace SushiBLAS
