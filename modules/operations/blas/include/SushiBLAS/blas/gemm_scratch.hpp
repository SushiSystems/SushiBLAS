/**************************************************************************/
/* gemm_scratch.hpp                                                       */
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
 * @file gemm_scratch.hpp
 * @brief Declares the scratch a GEMM leases: packed panels or the partial sums of depth slices.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/tensor_profile.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_depth_slice_plan.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/blas/packed_product_scratch.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the scratch bytes the depth slices of a GEMM of dtype take on a device.
         * @return 0 when the profile asks for no slices, or for an integer dtype.
         */
        [[nodiscard]] inline std::size_t
        depth_slice_scratch_bytes(Core::DataType dtype, const Device::Profile& profile,
                                  int64_t m, int64_t n, int64_t k, int64_t batch_size) noexcept
        {
            Kernels::GemmDepthSlicePlan plan;
            switch (dtype)
            {
                case Core::DataType::Half:
                    plan = Kernels::plan_gemm_depth_slices<sycl::half>(profile, m, n, k,
                                                                       batch_size);
                    break;
                case Core::DataType::Float32:
                    plan = Kernels::plan_gemm_depth_slices<float>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::Float64:
                    plan = Kernels::plan_gemm_depth_slices<double>(profile, m, n, k, batch_size);
                    break;
                case Core::DataType::Complex32:
                    plan = Kernels::plan_gemm_depth_slices<std::complex<float>>(profile, m, n, k,
                                                                                batch_size);
                    break;
                case Core::DataType::Complex64:
                    plan = Kernels::plan_gemm_depth_slices<std::complex<double>>(profile, m, n,
                                                                                 k, batch_size);
                    break;
                case Core::DataType::Int32:
                case Core::DataType::Int64:
                case Core::DataType::UInt32:
                    break;
            }
            return plan.scratch_bytes;
        }

        /**
         * @brief Returns the scratch bytes a GEMM of dtype leases: enough for whichever of the
         *        packed product and the depth slices the profile selects.
         */
        [[nodiscard]] inline std::size_t gemm_scratch_bytes(Core::DataType dtype,
                                                            const Device::Profile& profile,
                                                            int64_t m, int64_t n, int64_t k,
                                                            int64_t batch_size) noexcept
        {
            const std::size_t packed =
                packed_product_scratch_bytes(dtype, profile, m, n, k, batch_size);
            const std::size_t sliced =
                depth_slice_scratch_bytes(dtype, profile, m, n, k, batch_size);
            return packed > sliced ? packed : sliced;
        }

        /**
         * @brief Leases the scratch an m x n GEMM of k terms takes on a device of profile.
         * @return An empty buffer when the profile selects neither the packed product nor
         *         depth slices, and wherever allocate_blas_scratch returns one.
         */
        [[nodiscard]] inline ScratchBuffer
        allocate_gemm_scratch(TaskRecorder& recorder, const Tensor& C,
                              const Device::Profile& profile, int64_t m, int64_t n, int64_t k,
                              int64_t batch_size, OperandSet& operands)
        {
            if (!C.storage)
                return ScratchBuffer();

            const std::size_t bytes = gemm_scratch_bytes(C.dtype, profile, m, n, k, batch_size);
            return allocate_blas_scratch(recorder, C, bytes, operands);
        }

        /**
         * @brief Leases the scratch an m x n GEMM of k terms takes on the device of C.
         * @return An empty buffer when the profile of that device selects neither the packed
         *         product nor depth slices, and wherever allocate_blas_scratch returns one.
         */
        [[nodiscard]] inline ScratchBuffer
        allocate_gemm_scratch(TaskRecorder& recorder, const Tensor& C, int64_t m, int64_t n,
                              int64_t k, int64_t batch_size, OperandSet& operands)
        {
            if (!C.storage)
                return ScratchBuffer();

            return allocate_gemm_scratch(recorder, C, Dispatch::profile_of(C), m, n, k,
                                         batch_size, operands);
        }
    } // namespace Internal
} // namespace SushiBLAS
