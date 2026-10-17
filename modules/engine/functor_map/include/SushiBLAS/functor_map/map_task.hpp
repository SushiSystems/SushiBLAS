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
                     * @param family Stored, not copied; must outlive the task, as name must.
                     */
                    MapTask(Core::DataType dtype, const char* family, const char* name,
                            std::size_t n, void* out, const std::array<const void*, Arity>& in,
                            Op op)
                        : dtype_(dtype), family_(family), name_(name), n_(n), out_(out), in_(in),
                          vec_ok_(Launch::is_vector_aligned(out)), op_(std::move(op))
                    {
                        for (const void* input : in_)
                            vec_ok_ = vec_ok_ && Launch::is_vector_aligned(input);
                    }

                    /** @brief Runs the serial host loop under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           sycl::queue& q,
                                           const std::vector<sycl::event>& deps) const
                    {
                        SB_LOG_TRACE(K_FUNCTOR_MAP_LOG_CATEGORY, "{} '{}': {} elements", family_, name_, n_);

                        return pair.run(q, deps,
                            [this]
                            {
                                with_element_type<Dom>(dtype_, family_, name_,
                                    SerialMapArm<Op, Arity>{op_, n_, out_, in_});
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_,
                                        LaunchMapArm<Op, Arity>{h, q2, op_, n_, vec_ok_, out_, in_});
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

                    /// @brief The element count of every operand.
                    std::size_t n_;

                    /// @brief The output's base address.
                    void* out_;

                    /// @brief The inputs' base addresses, in functor order.
                    std::array<const void*, Arity> in_;

                    /// @brief Whether every operand can base a 128-bit access.
                    bool vec_ok_;

                    /// @brief The functor, with the addresses of its declared reads if it has any.
                    Op op_;
            };

            /**
             * @brief Builds the task body of a map-shaped seam.
             * @param op The seam's functor after Detail::task_functor.
             * @param ins The inputs' base addresses, in functor order.
             */
            template<Domain Dom, typename Op, typename... Ins>
            [[nodiscard]] inline MapTask<Dom, Op, sizeof...(Ins)> make_map_task(
                Core::DataType dtype, const char* family, const char* name, std::size_t n,
                void* out, const Op& op, const Ins*... ins)
            {
                return MapTask<Dom, Op, sizeof...(Ins)>(
                    dtype, family, name, n, out,
                    std::array<const void*, sizeof...(Ins)>{ins...}, op);
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
                     * @param family Stored, not copied; must outlive the task, as name must.
                     */
                    ZipTask(Core::DataType dtype, const char* family, const char* name,
                            std::size_t n, const std::array<void*, N>& ptrs, Op op)
                        : dtype_(dtype), family_(family), name_(name), n_(n), ptrs_(ptrs),
                          vec_ok_(true), op_(std::move(op))
                    {
                        for (const void* operand : ptrs_)
                            vec_ok_ = vec_ok_ && Launch::is_vector_aligned(operand);
                    }

                    /** @brief Runs the serial host loop under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           sycl::queue& q,
                                           const std::vector<sycl::event>& deps) const
                    {
                        SB_LOG_TRACE(K_FUNCTOR_MAP_LOG_CATEGORY, "{} '{}': {} elements over {} operands", family_, name_, n_, N);

                        return pair.run(q, deps,
                            [this]
                            {
                                with_element_type<Dom>(dtype_, family_, name_,
                                    SerialZipArm<Op, N>{op_, n_, ptrs_});
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_,
                                        LaunchZipArm<Op, N>{h, q2, op_, n_, vec_ok_, ptrs_});
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

                    /// @brief The element count of every operand.
                    std::size_t n_;

                    /// @brief The operands' base addresses, in functor order.
                    std::array<void*, N> ptrs_;

                    /// @brief Whether every operand can base a 128-bit access.
                    bool vec_ok_;

                    /// @brief The functor, with the addresses of its declared reads if it has any.
                    Op op_;
            };
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
