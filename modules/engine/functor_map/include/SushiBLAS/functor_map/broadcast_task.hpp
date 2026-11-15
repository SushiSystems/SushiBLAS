/**************************************************************************/
/* broadcast_task.hpp                                                     */
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
 * @file broadcast_task.hpp
 * @brief Declares the task body of the broadcast seam, with its nested host loops and its SYCL path.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /** @brief Names which index of the matrix selects the element of a broadcast vector. */
        enum class BroadcastIndex : std::uint8_t
        {
            Outer, /**< The outer index o selects: `v[o]` meets every element of row o. */
            Inner  /**< The inner index n selects: `v[n]` meets element n of every row. */
        };

        namespace Detail
        {
            /**
             * @brief Holds the geometry and the addresses of one broadcast call, as the task keeps them.
             * @tparam N How many vectors the call broadcasts, one to four.
             */
            template<std::size_t N>
            struct BroadcastOperands
            {
                /** @brief Holds the extent of the outer index. */
                std::size_t outer;
                /** @brief Holds the extent of the inner index. */
                std::size_t inner;
                /** @brief Holds the output's base address; it may equal a. */
                void* out;
                /** @brief Holds the matrix operand's base address. */
                const void* a;
                /** @brief Holds the vectors' base addresses, in functor order. */
                std::array<const void*, N> vectors;
                /** @brief Holds the index that selects from each vector. */
                std::array<BroadcastIndex, N> selected_by;
            };

            /**
             * @brief Computes one element, `out[o, n] = op(a[o, n], v0, ..., vN-1)`, for the element type T.
             * @tparam Fn What kernel_op returned for the seam's functor, decayed.
             * @pre o is below the outer extent and n below the inner one.
             */
            template<typename Fn, typename T, std::size_t N>
            class BroadcastElement final
            {
                public:
                    /** @brief Stores the functor and the typed addresses. */
                    BroadcastElement(Fn fn, const BroadcastOperands<N>& operands)
                        : fn_(std::move(fn)), inner_(operands.inner),
                          out_(static_cast<T*>(operands.out)),
                          a_(static_cast<const T*>(operands.a)),
                          vectors_(typed(operands.vectors, std::make_index_sequence<N>{})),
                          selected_by_(operands.selected_by)
                    {

                    }

                    /** @brief Reads the element of a and the selected element of each vector, then writes out. */
                    void operator()(std::size_t o, std::size_t n) const
                    {
                        store(o, n, std::make_index_sequence<N>{});
                    }

                private:
                    /** @brief Reinterprets the vectors' addresses as typed ones. */
                    template<std::size_t... K>
                    static std::array<const T*, N> typed(const std::array<const void*, N>& raw,
                                                         std::index_sequence<K...>)
                    {
                        return std::array<const T*, N>{static_cast<const T*>(raw[K])...};
                    }

                    /** @brief Expands the vectors; the element of a is read before out is written. */
                    template<std::size_t... K>
                    void store(std::size_t o, std::size_t n, std::index_sequence<K...>) const
                    {
                        const std::size_t i = o * inner_ + n;
                        out_[i] = fn_(a_[i], vectors_[K][selected_by_[K] == BroadcastIndex::Outer
                                                             ? o
                                                             : n]...);
                    }

                    /// @brief The functor, with its scalars narrowed and its reads typed.
                    Fn fn_;

                    /// @brief The extent of the inner index.
                    std::size_t inner_;

                    /// @brief The output.
                    T* out_;

                    /// @brief The matrix operand.
                    const T* a_;

                    /// @brief The vectors.
                    std::array<const T*, N> vectors_;

                    /// @brief The index that selects from each vector.
                    std::array<BroadcastIndex, N> selected_by_;
            };

            /**
             * @brief Builds the element computation of a broadcast for the element type of an arm.
             * @param fn What kernel_op returned for the seam's functor.
             */
            template<typename T, std::size_t N, typename Fn>
            [[nodiscard]] inline BroadcastElement<std::decay_t<Fn>, T, N> make_broadcast_element(
                Fn&& fn, const BroadcastOperands<N>& operands)
            {
                return BroadcastElement<std::decay_t<Fn>, T, N>(std::forward<Fn>(fn), operands);
            }

            /**
             * @brief Runs a broadcast on the host for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t N>
            struct SerialBroadcastArm
            {
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the geometry and the addresses. */
                const BroadcastOperands<N>& operands;

                /** @brief Runs the two nested loops for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    using T = typename Tag::type;

                    const auto element = make_broadcast_element<T>(kernel_op(tag, op), operands);
                    for (std::size_t o = 0; o < operands.outer; ++o)
                    {
                        for (std::size_t n = 0; n < operands.inner; ++n)
                            element(o, n);
                    }
                }
            };

            /**
             * @brief Launches a broadcast on the device for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t N>
            struct LaunchBroadcastArm
            {
                /** @brief Holds the command group the kernel is launched in. */
                sycl::handler& h;
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the geometry and the addresses. */
                const BroadcastOperands<N>& operands;

                /** @brief Launches one work-item per element for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    using T = typename Tag::type;

                    const auto element = make_broadcast_element<T>(kernel_op(tag, op), operands);
                    h.parallel_for(sycl::range<2>(operands.outer, operands.inner),
                                   [=](sycl::id<2> idx)
                                   {
                                       element(static_cast<std::size_t>(idx[0]),
                                               static_cast<std::size_t>(idx[1]));
                                   });
                }
            };

            /**
             * @brief Runs `out[o, n] = op(a[o, n], v0, ..., vN-1)` on the host or the device, as its mode says.
             * @tparam Dom The element families the functor is well-formed for.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             * @see modules/engine/functor_map/README.md
             */
            template<Domain Dom, typename Op, std::size_t N>
            class BroadcastTask final
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param family Stored, not copied; must outlive the task, as name must.
                     */
                    BroadcastTask(Core::DataType dtype, const char* family, const char* name,
                                  const BroadcastOperands<N>& operands, Op op)
                        : dtype_(dtype), family_(family), name_(name), operands_(operands),
                          op_(std::move(op))
                    {

                    }

                    /** @brief Runs the nested host loops under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps) const
                    {
                        return pair.run(deps,
                            [this]
                            {
                                with_element_type<Dom>(dtype_, family_, name_,
                                    SerialBroadcastArm<Op, N>{op_, operands_});
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_,
                                        LaunchBroadcastArm<Op, N>{h, op_, operands_});
                                });
                            });
                    }

                private:
                    /// @brief The dtype the task was recorded for.
                    Core::DataType dtype_;

                    /// @brief The operation family, for diagnostics.
                    const char* family_;

                    /// @brief The operation name, for diagnostics.
                    const char* name_;

                    /// @brief The geometry and the addresses of every operand.
                    BroadcastOperands<N> operands_;

                    /// @brief The functor, with the addresses of its declared reads if it has any.
                    Op op_;
            };

            /**
             * @brief Builds the task body of a broadcast-shaped seam.
             * @param op The seam's functor after Detail::task_functor.
             */
            template<Domain Dom, typename Op, std::size_t N>
            [[nodiscard]] inline BroadcastTask<Dom, Op, N> make_broadcast_task(
                Core::DataType dtype, const char* family, const char* name,
                const BroadcastOperands<N>& operands, const Op& op)
            {
                return BroadcastTask<Dom, Op, N>(dtype, family, name, operands, op);
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
