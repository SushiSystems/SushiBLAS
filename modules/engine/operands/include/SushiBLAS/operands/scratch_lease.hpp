/**************************************************************************/
/* scratch_lease.hpp                                                      */
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
 * @file scratch_lease.hpp
 * @brief Declares the helpers that size and lease the device scratch of a task.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the bytes one element of a substitution's wide solution takes.
         * @return 0 when the solution is carried in the dtype itself.
         */
        [[nodiscard]] inline constexpr std::size_t
        running_solution_bytes(Core::DataType dtype) noexcept
        {
            switch (dtype)
            {
                case Core::DataType::Half:
                    return Kernels::RunningSolution<sycl::half>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<sycl::half>::Wide) : 0;
                case Core::DataType::Float32:
                    return Kernels::RunningSolution<float>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<float>::Wide) : 0;
                case Core::DataType::Float64:
                    return Kernels::RunningSolution<double>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<double>::Wide) : 0;
                case Core::DataType::Complex32:
                    return Kernels::RunningSolution<std::complex<float>>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<std::complex<float>>::Wide) : 0;
                case Core::DataType::Complex64:
                    return Kernels::RunningSolution<std::complex<double>>::K_NEEDS_WORKSPACE
                               ? sizeof(Kernels::RunningSolution<std::complex<double>>::Wide) : 0;
                case Core::DataType::Int32:
                case Core::DataType::Int64:
                    return 0;
            }
            return 0;
        }

        /**
         * @brief Leases device scratch on the device of anchor and declares it written.
         * @return An empty buffer when anchor has no storage, bytes is zero, or the recorder's mode
         *         is ForceReference, under which no SYCL kernel reads it.
         */
        [[nodiscard]] inline ScratchBuffer allocate_blas_scratch(TaskRecorder& recorder,
                                                                 const Tensor& anchor,
                                                                 std::size_t bytes,
                                                                 OperandSet& operands)
        {
            if (!anchor.storage || bytes == 0 ||
                recorder.mode() == ExecutionMode::ForceReference)
                return ScratchBuffer();

            ScratchBuffer scratch(recorder.scratch_pool(), bytes,
                                  anchor.storage->device_index());
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
                Reduce::planned_partials(Device::profile_for(x.device()), elements);
            return allocate_blas_scratch(recorder, x, partials * partial_bytes, operands);
        }
    } // namespace Internal
} // namespace SushiBLAS
