/**************************************************************************/
/* map_task.hpp                                                           */
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
 * @file map_task.hpp
 * @brief Declares the task bodies of the per-element seams, each with its serial host path and its SYCL path.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/functor_map/functor_map_log_category.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/handle/bound_operands.hpp>
#include <SushiBLAS/functor_map/map_operand.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Runs a map on the host for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t Arity>
            struct SerialMapArm
            {
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the element count. */
                std::size_t n;
                /** @brief Holds the output's base address. */
                void* out;
                /** @brief Holds the inputs' base addresses, in functor order. */
                const std::array<const void*, Arity>& in;

                /** @brief Runs the loop for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    run(tag, std::make_index_sequence<Arity>{});
                }

                /** @brief Expands the inputs as typed pointers. */
                template<typename Tag, std::size_t... K>
                void run(Tag tag, std::index_sequence<K...>) const
                {
                    using T = typename Tag::type;

                    Serial::map(n, kernel_op(tag, op), static_cast<T*>(out),
                                static_cast<const T*>(in[K])...);
                }
            };

            /**
             * @brief Launches a map on the device for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t Arity>
            struct LaunchMapArm
            {
                /** @brief Holds the command group the kernel is launched in. */
                sycl::handler& h;
                /** @brief Holds the queue whose device plans the launch shape. */
                const sycl::queue& q;
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the element count. */
                std::size_t n;
                /** @brief Holds whether every operand can base a 128-bit access. */
                bool vec_ok;
                /** @brief Holds the output's base address. */
                void* out;
                /** @brief Holds the inputs' base addresses, in functor order. */
                const std::array<const void*, Arity>& in;

                /** @brief Launches the kernel for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    run(tag, std::make_index_sequence<Arity>{});
                }

                /** @brief Expands the inputs as typed pointers. */
                template<typename Tag, std::size_t... K>
                void run(Tag tag, std::index_sequence<K...>) const
                {
                    using T = typename Tag::type;

                    Launch::map(h, q, n, vec_ok, kernel_op(tag, op), static_cast<T*>(out),
                                static_cast<const T*>(in[K])...);
                }
            };

            /**
             * @brief Runs `out[i] = op(in0[i], ..., inM[i])` on the host or the device, as its mode says.
             * @tparam Dom The element families the functor is well-formed for.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             * @see modules/operations/MATH_README.md
             */
            template<Domain Dom, typename Op, std::size_t Arity>
            class MapTask final
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param out Where the output is found at each run.
                     * @param in Where each input is found, in functor order.
                     * @param family Stored, not copied; must outlive the task, as name must.
                     */
                    MapTask(Core::DataType dtype, const char* family, const char* name,
                            std::size_t n, const MapOperand& out,
                            const std::array<MapOperand, Arity>& in, Op op)
                        : dtype_(dtype), family_(family), name_(name), n_(n), out_(out), in_(in),
                          op_(std::move(op))
                    {

                    }

                    /**
                     * @brief Runs the serial host loop under ForceReference, else submits the kernel.
                     * @param pointers The base addresses the task bound; empty for a task that
                     *        recorded its addresses.
                     */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps,
                                           const SushiRuntime::BoundOperands& pointers) const
                    {
                        // The runtime may bind another instance at each run, so read them now.
                        void* const out = const_cast<void*>(out_.resolve(pointers));
                        std::array<const void*, Arity> in{};
                        bool vec_ok = Launch::is_vector_aligned(out);
                        for (std::size_t k = 0; k < Arity; ++k)
                        {
                            in[k] = in_[k].resolve(pointers);
                            vec_ok = vec_ok && Launch::is_vector_aligned(in[k]);
                        }

                        return pair.run(deps,
                            [&]
                            {
                                with_element_type<Dom>(dtype_, family_, name_,
                                    SerialMapArm<Op, Arity>{op_, n_, out, in});
                            },
                            [&](sycl::queue& q2,
                                const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_,
                                        LaunchMapArm<Op, Arity>{h, q2, op_, n_, vec_ok, out, in});
                                });
                            });
                    }

                    /** @brief Runs the task as above, for a caller that holds no bound table. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps) const
                    {
                        return (*this)(pair, deps, SushiRuntime::BoundOperands());
                    }

                private:
                    /// @brief The dtype the task was recorded for.
                    Core::DataType dtype_;

                    /// @brief The operation family, for diagnostics.
                    const char* family_;

                    /// @brief The operation name, for diagnostics.
                    const char* name_;

                    /// @brief The element count of every operand.
                    std::size_t n_;

                    /// @brief Where the output is found.
                    MapOperand out_;

                    /// @brief Where each input is found, in functor order.
                    std::array<MapOperand, Arity> in_;

                    /// @brief The functor, with the addresses of its declared reads if it has any.
                    Op op_;
            };

            /**
             * @brief Builds the task body of a map-shaped seam over operands found by address.
             * @param op The seam's functor after Detail::task_functor.
             * @param ins The inputs' base addresses, in functor order.
             */
            template<Domain Dom, typename Op, typename... Ins>
            [[nodiscard]] inline MapTask<Dom, Op, sizeof...(Ins)> make_map_task(
                Core::DataType dtype, const char* family, const char* name, std::size_t n,
                void* out, const Op& op, const Ins*... ins)
            {
                return MapTask<Dom, Op, sizeof...(Ins)>(
                    dtype, family, name, n, MapOperand(static_cast<const void*>(out)),
                    std::array<MapOperand, sizeof...(Ins)>{
                        MapOperand(static_cast<const void*>(ins))...},
                    op);
            }

            /**
             * @brief Builds the task body of a map-shaped seam over operands found through a bound
             *        table or by address, as each MapOperand says.
             * @param op The seam's functor after Detail::task_functor.
             */
            template<Domain Dom, typename Op, typename... Ins>
            [[nodiscard]] inline MapTask<Dom, Op, sizeof...(Ins)> make_operand_map_task(
                Core::DataType dtype, const char* family, const char* name, std::size_t n,
                const MapOperand& out, const Op& op, const Ins&... ins)
            {
                return MapTask<Dom, Op, sizeof...(Ins)>(
                    dtype, family, name, n, out, std::array<MapOperand, sizeof...(Ins)>{ins...},
                    op);
            }

            /**
             * @brief Runs a fused update on the host for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t N>
            struct SerialZipArm
            {
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the element count. */
                std::size_t n;
                /** @brief Holds the operands' base addresses, in functor order. */
                const std::array<void*, N>& ptrs;

                /** @brief Runs the loop for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    using T = typename Tag::type;

                    Serial::zip_inplace(n, Launch::typed_ptrs<T>(ptrs), kernel_op(tag, op));
                }
            };

            /**
             * @brief Launches a fused update on the device for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op, std::size_t N>
            struct LaunchZipArm
            {
                /** @brief Holds the command group the kernel is launched in. */
                sycl::handler& h;
                /** @brief Holds the queue whose device plans the launch shape. */
                const sycl::queue& q;
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the element count. */
                std::size_t n;
                /** @brief Holds whether every operand can base a 128-bit access. */
                bool vec_ok;
                /** @brief Holds the operands' base addresses, in functor order. */
                const std::array<void*, N>& ptrs;

                /** @brief Launches the kernel for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    using T = typename Tag::type;

                    Launch::zip_inplace(h, q, n, vec_ok, Launch::typed_ptrs<T>(ptrs),
                                        kernel_op(tag, op));
                }
            };

            /**
             * @brief Runs `op(t0[i], ..., tN-1[i])` on the host or the device, as its mode says.
             * @tparam Dom The element families the functor is well-formed for.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             * @see modules/operations/MATH_README.md
             */
            template<Domain Dom, typename Op, std::size_t N>
            class ZipTask final
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param operands Where each operand is found, in functor order.
                     * @param family Stored, not copied; must outlive the task, as name must.
                     */
                    ZipTask(Core::DataType dtype, const char* family, const char* name,
                            std::size_t n, const std::array<MapOperand, N>& operands, Op op)
                        : dtype_(dtype), family_(family), name_(name), n_(n), operands_(operands),
                          op_(std::move(op))
                    {

                    }

                    /** @brief Holds the operands by their recorded addresses. */
                    ZipTask(Core::DataType dtype, const char* family, const char* name,
                            std::size_t n, const std::array<void*, N>& ptrs, Op op)
                        : ZipTask(dtype, family, name, n, recorded_operands(ptrs), std::move(op))
                    {

                    }

                    /**
                     * @brief Runs the serial host loop under ForceReference, else submits the kernel.
                     * @param pointers The base addresses the task bound; empty for a task that
                     *        recorded its addresses.
                     */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps,
                                           const SushiRuntime::BoundOperands& pointers) const
                    {
                        std::array<void*, N> ptrs{};
                        bool vec_ok = true;
                        for (std::size_t k = 0; k < N; ++k)
                        {
                            ptrs[k] = const_cast<void*>(operands_[k].resolve(pointers));
                            vec_ok = vec_ok && Launch::is_vector_aligned(ptrs[k]);
                        }

                        return pair.run(deps,
                            [&]
                            {
                                with_element_type<Dom>(dtype_, family_, name_,
                                    SerialZipArm<Op, N>{op_, n_, ptrs});
                            },
                            [&](sycl::queue& q2,
                                const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_,
                                        LaunchZipArm<Op, N>{h, q2, op_, n_, vec_ok, ptrs});
                                });
                            });
                    }

                    /** @brief Runs the task as above, for a caller that holds no bound table. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps) const
                    {
                        return (*this)(pair, deps, SushiRuntime::BoundOperands());
                    }

                private:
                    /** @brief Returns the operands that stand for the recorded addresses ptrs. */
                    [[nodiscard]] static std::array<MapOperand, N>
                    recorded_operands(const std::array<void*, N>& ptrs)
                    {
                        std::array<MapOperand, N> operands{};
                        for (std::size_t k = 0; k < N; ++k)
                            operands[k] = MapOperand(static_cast<const void*>(ptrs[k]));
                        return operands;
                    }

                    /// @brief The dtype the task was recorded for.
                    Core::DataType dtype_;

                    /// @brief The operation family, for diagnostics.
                    const char* family_;

                    /// @brief The operation name, for diagnostics.
                    const char* name_;

                    /// @brief The element count of every operand.
                    std::size_t n_;

                    /// @brief Where each operand is found, in functor order.
                    std::array<MapOperand, N> operands_;

                    /// @brief The functor, with the addresses of its declared reads if it has any.
                    Op op_;
            };
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
