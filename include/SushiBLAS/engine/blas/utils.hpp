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
 * @file utils.hpp
 * @brief Declares the BLAS helpers that size a result, allocate scratch and record a typed task.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <complex>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/running_solution.hpp>
#include <SushiBLAS/kernels/blas/level3/packed_product_plan.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the dtype a norm or absolute-sum result holds for an operand of dtype.
         * @return The real component type of a complex dtype, dtype itself otherwise.
         */
        [[nodiscard]] inline constexpr Core::DataType real_result_dtype(Core::DataType dtype)
        {
            if (dtype == Core::DataType::COMPLEX32)
                return Core::DataType::FLOAT32;
            if (dtype == Core::DataType::COMPLEX64)
                return Core::DataType::FLOAT64;
            return dtype;
        }

        /**
         * @brief Throws unless a scalar result tensor has storage, the expected dtype and exactly
         *        one element.
         * @throws std::invalid_argument naming the first rule that fails.
         */
        inline void require_scalar_result(const Tensor& result, Core::DataType expected,
                                          const char* op)
        {
            Dispatch::require_storage(result, op, "result");
            Dispatch::require_dtype(result, expected, op, "result");
            if (result.num_elements != 1)
                Dispatch::Detail::throw_operand_error(op, "result",
                                                      "holds {} elements; a scalar result holds "
                                                      "exactly one",
                                                      result.num_elements);
        }

        /**
         * @brief Returns the bytes one element of a substitution's wide solution takes.
         * @return 0 when the solution is carried in the dtype itself.
         */
        [[nodiscard]] inline constexpr std::size_t
        running_solution_bytes(Core::DataType dtype) noexcept
        {
            switch (dtype)
            {
                case Core::DataType::HALF:
                    return Kernels::RunningSolution<sycl::half>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<sycl::half>::Wide) : 0;
                case Core::DataType::FLOAT32:
                    return Kernels::RunningSolution<float>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<float>::Wide) : 0;
                case Core::DataType::FLOAT64:
                    return Kernels::RunningSolution<double>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<double>::Wide) : 0;
                case Core::DataType::COMPLEX32:
                    return Kernels::RunningSolution<std::complex<float>>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<std::complex<float>>::Wide) : 0;
                case Core::DataType::COMPLEX64:
                    return Kernels::RunningSolution<std::complex<double>>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<std::complex<double>>::Wide) : 0;
                case Core::DataType::INT32:
                case Core::DataType::INT64:
                    return 0;
            }
            return 0;
        }

        /**
         * @brief Leases device scratch on the device of anchor and declares it written.
         * @return An empty buffer when anchor has no storage, bytes is zero, or the recorder's mode
         *         is FORCE_REFERENCE, under which no SYCL kernel reads it.
         */
        [[nodiscard]] inline ScratchBuffer allocate_blas_scratch(TaskRecorder& recorder,
                                                                 const Tensor& anchor,
                                                                 std::size_t bytes,
                                                                 OperandSet& operands)
        {
            if (!anchor.storage || bytes == 0 ||
                recorder.get_mode() == ExecutionMode::FORCE_REFERENCE)
                return ScratchBuffer();

            ScratchBuffer scratch(recorder.get_scratch_pool(), bytes,
                                  anchor.storage->get_device_index());
            operands.scratch(scratch);
            return scratch;
        }

        /**
         * @brief Leases the partials scratch a Level 1 reduction over x plans.
         * @param elements Values the reduction folds: n, or 2n for a complex nrm2.
         * @param partial_bytes Bytes one partial takes.
         */
        [[nodiscard]] inline ScratchBuffer allocate_level1_scratch(TaskRecorder& recorder,
                                                                   const Tensor& x,
                                                                   std::size_t elements,
                                                                   std::size_t partial_bytes,
                                                                   OperandSet& operands)
        {
            if (!x.storage || elements == 0)
                return ScratchBuffer();

            const std::size_t partials =
                Reduce::planned_partials(Device::profile_for(x.get_device()), elements);
            return allocate_blas_scratch(recorder, x, partials * partial_bytes, operands);
        }

        /**
         * @brief Returns the scratch bytes a packed product of dtype needs on a device.
         * @return 0 when the profile does not select the packed kernel, or for an integer dtype.
         */
        [[nodiscard]] inline std::size_t
        packed_product_scratch_bytes(Core::DataType dtype, const Device::Profile& profile,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t batch_size) noexcept
        {
            Kernels::PackedProductPlan plan;
            switch (dtype)
            {
                case Core::DataType::HALF:
                    plan = Kernels::plan_packed_product<sycl::half>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::FLOAT32:
                    plan = Kernels::plan_packed_product<float>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::FLOAT64:
                    plan = Kernels::plan_packed_product<double>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::COMPLEX32:
                    plan = Kernels::plan_packed_product<std::complex<float>>(profile, m, n, k,
                                                                             batch_size);
                    break;
                case Core::DataType::COMPLEX64:
                    plan = Kernels::plan_packed_product<std::complex<double>>(profile, m, n, k,
                                                                              batch_size);
                    break;
                case Core::DataType::INT32:
                case Core::DataType::INT64:
                    break;
            }
            return plan.selected ? plan.scratch_bytes : std::size_t{0};
        }

        /**
         * @brief Leases the panels an m x n product of k terms stages, on the device of C.
         * @return An empty buffer when the profile of that device does not select the packed
         *         kernel, and wherever allocate_blas_scratch returns one.
         */
        [[nodiscard]] inline ScratchBuffer
        allocate_packed_product_scratch(TaskRecorder& recorder, const Tensor& C,
                                        int64_t m, int64_t n, int64_t k, int64_t batch_size,
                                        OperandSet& operands)
        {
            if (!C.storage)
                return ScratchBuffer();

            const std::size_t bytes = packed_product_scratch_bytes(
                C.dtype, Device::profile_for(C.get_device()), m, n, k, batch_size);
            return allocate_blas_scratch(recorder, C, bytes, operands);
        }

        /**
         * @brief Records one dtype-dispatched BLAS task whose body runs a reference and SYCL pair.
         * @param op_func Callable taking a value of the element type, a const
         *        KernelPairDispatcher&, the queue and the dependencies.
         * @throws std::invalid_argument at record time on a dtype outside Dom.
         * @throws UnsupportedExecution at record time, as TaskRecorder::record_pair states.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        TaskHandle execute_blas_task(TaskRecorder& recorder,
                                     const SushiRuntime::Graph::TaskMetadata& meta,
                                     const OperandSet& operands, Core::DataType dtype,
                                     Func&& op_func)
        {
            return execute_typed_task<Dom>(recorder, meta, operands, dtype, "BLAS operation",
                                           std::forward<Func>(op_func));
        }
    } // namespace Internal
} // namespace SushiBLAS
