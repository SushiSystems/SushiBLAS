/**************************************************************************/
/* packed_product_scratch.hpp                                             */
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
 * @file packed_product_scratch.hpp
 * @brief Declares the helpers that size and lease the panels of a packed product.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
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
                case Core::DataType::Half:
                    plan = Kernels::plan_packed_product<sycl::half>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::Float32:
                    plan = Kernels::plan_packed_product<float>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::Float64:
                    plan = Kernels::plan_packed_product<double>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::Complex32:
                    plan = Kernels::plan_packed_product<std::complex<float>>(profile, m, n, k,
                                                                             batch_size);
                    break;
                case Core::DataType::Complex64:
                    plan = Kernels::plan_packed_product<std::complex<double>>(profile, m, n, k,
                                                                              batch_size);
                    break;
                case Core::DataType::Int32:
                case Core::DataType::Int64:
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
                C.dtype, Device::profile_for(C.device()), m, n, k, batch_size);
            return allocate_blas_scratch(recorder, C, bytes, operands);
        }
    } // namespace Internal
} // namespace SushiBLAS
