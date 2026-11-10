/**************************************************************************/
/* gemm_recording.hpp                                                     */
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
 * @file gemm_recording.hpp
 * @brief Declares the port that records a checked GEMM call and the in-memory adapter that
 *        records it without a kernel.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/cost/cost_shape.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/blas/shape_check.hpp>
#include <SushiBLAS/blas/packed_product_scratch.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the tensors an epilogue declared and the floating-point operations it does
         *        on each element of C.
         * @see modules/operations/blas/README.md
         */
        struct GemmEpilogueAccess
        {
            OperandSet operands;                 /**< What the epilogue reads and writes. */
            std::uint32_t flops_per_element = 0; /**< Operations the epilogue does per element. */
        };

        /**
         * @brief Holds a GEMM call that passed every record-time check: the task description, the
         *        access set, the leased panels and the operands the kernel is given.
         * @see modules/operations/blas/README.md
         */
        struct PreparedGemm
        {
            Core::DataType dtype = Core::DataType::Float32;     /**< The element type of A, B and C. */
            TaskDescription description;                        /**< The name, scalars, flags and shape. */
            OperandSet operands;                                /**< A, B and C, and the epilogue's. */
            ScratchBuffer pack;                                 /**< The panels of a packed product. */
            GemmGeometry geometry;                              /**< The extents and strides. */
            Core::Layout layout = Core::Layout::RowMajor;       /**< The layout of C. */
            Core::Transpose trans_a = Core::Transpose::NoTrans; /**< How A is read. */
            Core::Transpose trans_b = Core::Transpose::NoTrans; /**< How B is read. */
            BLASScalar alpha = 1.0;                             /**< The scale of the product. */
            BLASScalar beta = 0.0;                              /**< The scale of the old C. */
            const void* read_a = nullptr;                       /**< The data of A. */
            const void* read_b = nullptr;                       /**< The data of B. */
            void* write_c = nullptr;                            /**< The data of C. */
        };

        /**
         * @brief Records a prepared GEMM on a recorder; the adapter chooses what the task runs.
         * @see modules/operations/blas/README.md
         */
        class GemmTaskPort
        {
            public:
                GemmTaskPort() = default;
                GemmTaskPort(const GemmTaskPort&) = delete;
                GemmTaskPort& operator=(const GemmTaskPort&) = delete;
                virtual ~GemmTaskPort() = default;

                /**
                 * @brief Records gemm as one task.
                 * @throws std::invalid_argument at record time on a dtype the adapter refuses.
                 */
                virtual TaskHandle record(TaskRecorder& recorder, const PreparedGemm& gemm) = 0;
        };

        /**
         * @brief Records the task of a GEMM with a body that computes nothing, so the access set,
         *        the description and the order of the task can be read with no kernel.
         * @see modules/operations/blas/README.md
         */
        class InMemoryGemmPort final : public GemmTaskPort
        {
            public:
                /** @brief Records gemm with a body that submits no kernel and returns no event. */
                TaskHandle record(TaskRecorder& recorder, const PreparedGemm& gemm) override
                {
                    return recorder.record_pair(
                        gemm.description, gemm.operands, "BLAS operation",
                        [](const KernelPairDispatcher&,
                           const std::vector<sycl::event>&) -> sycl::event
                        {
                            return sycl::event();
                        });
                }
        };

        /**
         * @brief Checks a GEMM call, builds its access set and leases its panels.
         * @param real_only True when the call refuses a complex tensor.
         * @param epilogue What the epilogue declared; empty for a bare functor.
         * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] inline PreparedGemm
        prepare_gemm(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                     Core::Transpose transA, Core::Transpose transB,
                     const BLASScalar& alpha, const BLASScalar& beta,
                     const char* name, SushiRuntime::OperationID op_id, bool real_only,
                     const GemmEpilogueAccess& epilogue)
        {
            require_flag(transA, name, "transA");
            require_flag(transB, name, "transB");
            require_dtype(B, A.dtype, name, "B");
            require_dtype(C, A.dtype, name, "C");
            require_real_scalar(alpha, A.dtype, name, "alpha");
            require_real_scalar(beta, A.dtype, name, "beta");

            require_gemm_domain(real_only, A, name);

            require_operand_storage(A, name, "A");
            require_operand_storage(B, name, "B");
            require_operand_storage(C, name, "C");

            PreparedGemm gemm;
            gemm.geometry = check_gemm_shapes(A, B, C, transA, transB, name);
            const GemmGeometry& g = gemm.geometry;

            // C is read unless beta is zero; see modules/operations/blas/README.md.
            gemm.operands.reads(A).reads(B);
            if (beta.is_zero())
                gemm.operands.writes(C);
            else
                gemm.operands.updates(C);
            gemm.operands.merge(epilogue.operands);

            gemm.pack = Internal::allocate_packed_product_scratch(
                recorder, C, g.m, g.n, g.k, g.batch_size, gemm.operands);

            gemm.description = Internal::BLASMetadata(name, op_id)
                                   .scalar(alpha).scalar(beta)
                                   .flag(transA).flag(transB)
                                   .shape(GemmShape{g.m, g.n, g.k, g.batch_size, A.dtype,
                                                    epilogue.flops_per_element})
                                   .get();
            gemm.dtype = A.dtype;
            gemm.layout = g.c.layout;
            gemm.trans_a = transA;
            gemm.trans_b = transB;
            gemm.alpha = alpha;
            gemm.beta = beta;
            gemm.read_a = A.data_or_null();
            gemm.read_b = B.data_or_null();
            gemm.write_c = C.data_or_null();
            return gemm;
        }

        /**
         * @brief Checks a GEMM call and records it through port.
         * @see prepare_gemm
         */
        inline TaskHandle record_gemm(GemmTaskPort& port, TaskRecorder& recorder, const Tensor& A,
                                      const Tensor& B, Tensor& C, Core::Transpose transA,
                                      Core::Transpose transB, const BLASScalar& alpha,
                                      const BLASScalar& beta, const char* name,
                                      SushiRuntime::OperationID op_id, bool real_only,
                                      const GemmEpilogueAccess& epilogue)
        {
            return port.record(recorder, prepare_gemm(recorder, A, B, C, transA, transB, alpha,
                                                      beta, name, op_id, real_only, epilogue));
        }
    } // namespace Dispatch
} // namespace SushiBLAS
