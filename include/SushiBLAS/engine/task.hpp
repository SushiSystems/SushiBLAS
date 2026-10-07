/**************************************************************************/
/* task.hpp                                                               */
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
 * @file task.hpp
 * @brief Declares the recorder of one dtype-dispatched task and the execution-mode dispatcher.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {

        /**
         * @brief Runs the reference kernel on the host under FORCE_REFERENCE, else the SYCL kernel.
         * @param reference_fn Callable `() -> void` running the pure-C++ reference kernel.
         * @param sycl_fn Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
         * @return The submitted event, or an empty one under FORCE_REFERENCE.
         * @throws std::logic_error if mode is none of AUTO, FORCE_SYCL and FORCE_REFERENCE.
         * @see include/SushiBLAS/engine/README.md
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
            else if (mode == ExecutionMode::AUTO || mode == ExecutionMode::FORCE_SYCL)
            {
                return sycl_fn(q, deps);
            }
            else
            {
                throw std::logic_error("dispatch_with_mode received an execution mode it does not handle.");
            }
        }

        /**
         * @brief Holds the pointers an operation's reference kernel reads or writes on the host.
         * @see include/SushiBLAS/engine/README.md
         */
        class ReferenceOperands
        {
            public:
                /** @brief Lists pointers; a null entry stands for an operand without storage. */
                ReferenceOperands(std::initializer_list<const void*> pointers) : pointers_(pointers) {}

                /** @brief Returns the listed pointers, null entries included. */
                [[nodiscard]] const std::vector<const void*>& pointers() const noexcept { return pointers_; }

            private:
                /// @brief The pointers as listed.
                std::vector<const void*> pointers_;
        };

        /**
         * @brief Throws if FORCE_REFERENCE would make the host dereference device-only USM.
         * @param context The runtime context whose device contexts the pointers are looked up in.
         * @throws std::invalid_argument naming the operation if mode is FORCE_REFERENCE and a
         *         pointer is usm::alloc::device in one of the context's device contexts.
         * @see include/SushiBLAS/engine/README.md
         */
        inline void require_host_reachable(ExecutionMode mode,
                                           SushiRuntime::Execution::RuntimeContext& context,
                                           const ReferenceOperands& operands, const char* family,
                                           const std::string& name)
        {
            if (mode != ExecutionMode::FORCE_REFERENCE)
                return;

            const std::size_t device_count = context.device_count();
            for (const void* pointer : operands.pointers())
            {
                if (pointer == nullptr)
                    continue;

                for (std::size_t index = 0; index < device_count; ++index)
                {
                    const sycl::context device_context = context.get_queue(index).get_context();
                    if (sycl::get_pointer_type(pointer, device_context) == sycl::usm::alloc::device)
                        throw std::invalid_argument(std::string(family) + " '" + name +
                                                    "' cannot run under FORCE_REFERENCE: an operand "
                                                    "is device-only USM, which the host reference "
                                                    "kernel cannot dereference.");
                }
            }
        }

        /**
         * @brief Runs an operation's reference and SYCL kernel pair under the mode fixed at record time.
         * @see include/SushiBLAS/engine/README.md
         */
        class KernelPairDispatcher
        {
            public:
                /** @brief Fixes the execution mode every run() routes by. */
                explicit KernelPairDispatcher(ExecutionMode mode) noexcept : mode_(mode) {}

                /**
                 * @brief Runs reference_fn or sycl_fn as dispatch_with_mode routes them.
                 * @return The submitted event, or an empty one under FORCE_REFERENCE.
                 */
                template<typename ReferenceFn, typename SyclFn>
                sycl::event run(sycl::queue& q, const std::vector<sycl::event>& deps,
                                ReferenceFn&& reference_fn, SyclFn&& sycl_fn) const
                {
                    return dispatch_with_mode(mode_, q, deps, std::forward<ReferenceFn>(reference_fn),
                                              std::forward<SyclFn>(sycl_fn));
                }

            private:
                /// @brief The engine's execution mode when the task was recorded.
                ExecutionMode mode_;
        };

        namespace Detail
        {
            /**
             * @brief Calls a task body with a value of the element type its dtype names.
             * @tparam WithDispatcher Whether the body takes a KernelPairDispatcher after the scalar.
             * @see include/SushiBLAS/engine/README.md
             */
            template<Dispatch::Domain Dom, typename Func, bool WithDispatcher>
            class TypedTaskBody
            {
                public:
                    /** @brief Holds everything the body needs at execute time. */
                    TypedTaskBody(Core::DataType dtype, const char* family, std::string name,
                                  Func op_func, KernelPairDispatcher dispatcher)
                        : dtype_(dtype), family_(family), name_(std::move(name)),
                          op_func_(std::move(op_func)), dispatcher_(dispatcher)
                    {

                    }

                    /**
                     * @brief Runs the body for the recorded dtype.
                     * @throws std::runtime_error if dtype falls outside Dom.
                     */
                    sycl::event operator()(sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        // The seven arms mirror Dispatch::with_element_type; see include/SushiBLAS/engine/README.md.
                        switch (dtype_)
                        {
                            case Core::DataType::HALF:
                                return call(sycl::half{}, q, deps);
                            case Core::DataType::FLOAT32:
                                return call(float{}, q, deps);
                            case Core::DataType::FLOAT64:
                                return call(double{}, q, deps);
                            case Core::DataType::COMPLEX32:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::COMPLEX))
                                    return call(std::complex<float>{}, q, deps);
                                break;
                            case Core::DataType::COMPLEX64:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::COMPLEX))
                                    return call(std::complex<double>{}, q, deps);
                                break;
                            case Core::DataType::INT32:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::INTEGRAL))
                                    return call(int32_t{}, q, deps);
                                break;
                            case Core::DataType::INT64:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::INTEGRAL))
                                    return call(int64_t{}, q, deps);
                                break;
                        }

                        // The switch ends in a throw, so a dtype that slips past require_domain cannot no-op; see include/SushiBLAS/engine/README.md.
                        SB_THROW_IF(true, "{} '{}' does not support dtype {}.",
                                    family_, name_, Core::to_string(dtype_));
                        return sycl::event();
                    }

                private:
                    /** @brief Calls the body with scalar, adding the dispatcher when it takes one. */
                    template<typename T>
                    sycl::event call(T scalar, sycl::queue& q, const std::vector<sycl::event>& deps) const
                    {
                        if constexpr (WithDispatcher)
                            return op_func_(scalar, dispatcher_, q, deps);
                        else
                            return op_func_(scalar, q, deps);
                    }

                    /// @brief The dtype the task was recorded for.
                    Core::DataType dtype_;
                    /// @brief The operation family, for diagnostics.
                    const char* family_;
                    /// @brief The operation name, for diagnostics.
                    std::string name_;
                    /// @brief The body the operation recorded.
                    Func op_func_;
                    /// @brief The mode router handed to a body that takes one.
                    KernelPairDispatcher dispatcher_;
            };
        } // namespace Detail

        /**
         * @brief Records one dtype-dispatched task on the engine's graph.
         * @param family What kind of operation this is, for diagnostics.
         * @param op_func Callable taking a scalar of the element type, the queue and the dependencies.
         * @return A default-constructed event that is already complete; the task finishes inside
         *         Engine::execute, which blocks.
         * @throws std::runtime_error at record time if dtype falls outside Dom.
         * @see include/SushiBLAS/engine/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_typed_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, const char* family, Func&& op_func)
        {
            Dispatch::require_domain<Dom>(dtype, family, meta.name);

            engine.get_graph().add_task(meta, reads, writes,
                Detail::TypedTaskBody<Dom, std::decay_t<Func>, false>(
                    dtype, family, meta.name, std::forward<Func>(op_func),
                    KernelPairDispatcher(ExecutionMode::AUTO)));
            return sycl::event();
        }

        /**
         * @brief Records one dtype-dispatched task whose body runs a reference and SYCL kernel pair.
         * @param host_operands Every pointer the reference kernel reads or writes.
         * @param op_func Callable taking a scalar of the element type, a const KernelPairDispatcher&,
         *        the queue and the dependencies.
         * @throws std::runtime_error at record time if dtype falls outside Dom.
         * @throws std::invalid_argument at record time, as require_host_reachable states.
         * @see include/SushiBLAS/engine/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_typed_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, const char* family,
                                       const ReferenceOperands& host_operands, Func&& op_func)
        {
            Dispatch::require_domain<Dom>(dtype, family, meta.name);

            const ExecutionMode mode = engine.get_mode();
            require_host_reachable(mode, engine.get_context(), host_operands, family, meta.name);
            const KernelPairDispatcher dispatcher(mode);

            engine.get_graph().add_task(meta, reads, writes,
                Detail::TypedTaskBody<Dom, std::decay_t<Func>, true>(
                    dtype, family, meta.name, std::forward<Func>(op_func), dispatcher));
            return sycl::event();
        }
    }
}
