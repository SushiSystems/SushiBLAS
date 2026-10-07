/**************************************************************************/
/* dispatch.hpp                                                           */
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
 * @file dispatch.hpp
 * @brief Declares the GEMM executor of the dispatch seam and the epilogue bundle it accepts.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/matrix_operand.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        // The access set travels in the epilogue, so it cannot be lost; see include/SushiBLAS/engine/blas/README.md.

        /**
         * @brief Holds a GEMM epilogue functor together with the regions it reads and writes.
         * @tparam Epilogue The functor type, under the contract execute_gemm states for a bare one.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<typename Epilogue>
        class GemmEpilogue
        {
            public:
                /** @brief Wraps functor, declaring nothing yet. */
                explicit GemmEpilogue(Epilogue functor) : functor_(std::move(functor)) {}

                /**
                 * @brief Declares that the epilogue reads operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::runtime_error if operand has no storage.
                 */
                GemmEpilogue& reads(const Tensor& operand)
                {
                    require_declared_storage(operand, "reads");
                    operands_.reads(operand);
                    return *this;
                }

                /**
                 * @brief Declares that the epilogue writes operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::runtime_error if operand has no storage.
                 * @see include/SushiBLAS/engine/blas/README.md
                 */
                GemmEpilogue& writes(const Tensor& operand)
                {
                    require_declared_storage(operand, "writes");
                    operands_.writes(operand);
                    return *this;
                }

                /** @brief Returns the functor itself, the only part that reaches the device. */
                [[nodiscard]] const Epilogue& functor() const noexcept { return functor_; }

                /** @brief Returns the tensors declared through reads() and writes(). */
                [[nodiscard]] const OperandSet& operands() const noexcept { return operands_; }

                /** @brief Returns the regions declared through reads(). */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                read_regions() const noexcept { return operands_.read_regions(); }

                /** @brief Returns the regions declared through writes(). */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                write_regions() const noexcept { return operands_.write_regions(); }

            private:
                /**
                 * @brief Refuses an operand whose region keys on no allocation.
                 * @param direction "reads" or "writes", for the message.
                 * @throws std::runtime_error if operand has no storage.
                 * @see include/SushiBLAS/engine/blas/README.md
                 */
                static void require_declared_storage(const Tensor& operand, const char* direction)
                {
                    SB_THROW_IF(!operand.region().valid(),
                                "A GEMM epilogue declared a tensor it {}, but that tensor has no "
                                "storage. Its region keys on no allocation, so the declaration "
                                "would order it against nothing - which is the silent hazard "
                                "declaring it was meant to close.",
                                direction);
                }

                /// @brief The epilogue functor.
                Epilogue functor_;

                /// @brief The tensors the functor dereferences, with their storage retained.
                OperandSet operands_;
        };

        /**
         * @brief Bundles functor into a GemmEpilogue that declares nothing yet.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<typename Epilogue>
        [[nodiscard]] inline GemmEpilogue<Epilogue> gemm_epilogue(Epilogue functor)
        {
            return GemmEpilogue<Epilogue>(std::move(functor));
        }

        namespace Detail
        {
            /**
             * @brief The functor of a bare epilogue: itself.
             * @tparam Epilogue The epilogue functor type.
             * @param epilogue The epilogue.
             * @return @p epilogue.
             */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue& epilogue_functor(const Epilogue& epilogue) noexcept
            {
                return epilogue;
            }

            /**
             * @brief Returns the functor a GemmEpilogue bundle carries.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue&
            epilogue_functor(const GemmEpilogue<Epilogue>& bundle) noexcept
            {
                return bundle.functor();
            }

            /**
             * @brief Adds nothing, because a bare epilogue declares no operands.
             * @see include/SushiBLAS/engine/blas/README.md
             */
            template<typename Epilogue>
            inline void append_epilogue_access(const Epilogue&, OperandSet&) noexcept
            {

            }

            /** @brief Adds a bundle's declared operands to a task's operand set. */
            template<typename Epilogue>
            inline void append_epilogue_access(const GemmEpilogue<Epilogue>& bundle,
                                               OperandSet& operands)
            {
                operands.merge(bundle.operands());
            }
        } // namespace Detail

        /**
         * @brief Records `C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)` as one task.
         * @param C Must not overlap A or B.
         * @param epilogue Must be device-copyable, capture by value, and neither allocate nor throw.
         * @return The handle of the recorded task; the handle of no task when nothing was recorded.
         * @throws std::runtime_error on rank < 2, a dtype mismatch, or a dtype outside Dom.
         * @throws std::invalid_argument on a non-contiguous operand, mixed layouts, or a shape or batch mismatch.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<Domain Dom = Domain::REAL, typename Epilogue>
        TaskHandle execute_gemm(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                                bool transA, bool transB, double alpha, double beta,
                                const char* name, SushiRuntime::Graph::OpID op_id,
                                Epilogue epilogue)
        {
            SB_THROW_IF(A.rank < 2 || B.rank < 2 || C.rank < 2,
                        "GEMM operation '{}' requires at least 2D tensors.", name);
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype,
                        "Data type mismatch in GEMM operation '{}'.", name);

            if constexpr (Dom == Domain::REAL)
            {
                // A dtype outside the domain is rejected at record time; see include/SushiBLAS/engine/blas/README.md.
                SB_THROW_IF(Core::is_complex_dtype(A.dtype),
                            "GEMM operation '{}' is not declared for complex tensors (dtype {}): its "
                            "epilogue was instantiated for real scalars only. Pass "
                            "Domain::REAL_AND_COMPLEX if the epilogue is well-formed for std::complex.",
                            name, Core::to_string(A.dtype));
            }

            const MatrixOperand a = describe_matrix_operand(A, name, "A");
            const MatrixOperand b = describe_matrix_operand(B, name, "B");
            const MatrixOperand c = describe_matrix_operand(C, name, "C");
            require_same_layout(a, c, name, "A", "C");
            require_same_layout(b, c, name, "B", "C");

            const int64_t m = c.rows;
            const int64_t n = c.cols;
            const int64_t k = transA ? a.rows : a.cols;
            require_matrix_extent(transA ? a.cols : a.rows, m, "rows in op(A)", name, "A");
            require_matrix_extent(transB ? b.cols : b.rows, k, "rows in op(B)", name, "B");
            require_matrix_extent(transB ? b.rows : b.cols, n, "columns in op(B)", name, "B");

            const int64_t batch_size = c.batch_count;
            const int64_t str_a = batch_stride(a, batch_size, name, "A");
            const int64_t str_b = batch_stride(b, batch_size, name, "B");
            const int64_t str_c = batch_stride(c, batch_size, name, "C");
            require_same_batch_axes(A, a, C, c, name, "A", "C");
            require_same_batch_axes(B, b, C, c, name, "B", "C");

            const int64_t lda = a.leading_dimension;
            const int64_t ldb = b.leading_dimension;
            const int64_t ldc = c.leading_dimension;

            const auto layout = c.layout;
            const void* read_A = A.data_or_null();
            const void* read_B = B.data_or_null();
            void* rw_C = C.data_or_null();

            // C is declared as a read whenever beta is non-zero; see include/SushiBLAS/engine/blas/README.md.
            OperandSet operands;
            operands.reads(A).reads(B);
            if (beta != 0.0)
                operands.reads(C);
            operands.writes(C);

            // The epilogue's declared operands join the operand set here; see include/SushiBLAS/engine/blas/README.md.
            Detail::append_epilogue_access(epilogue, operands);

            // Only the functor enters the kernel closure, not the bundle; see include/SushiBLAS/engine/blas/README.md.
            const auto kernel_epilogue = Detail::epilogue_functor(epilogue);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            meta.set_param(0, alpha);
            meta.set_param(1, beta);
            meta.set_param(2, transA);
            meta.set_param(3, transB);

            return Internal::execute_blas_task<Dom>(recorder, meta, operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);

                    auto pA = static_cast<const T*>(read_A);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    auto alpha_t = static_cast<T>(alpha);
                    auto beta_t = static_cast<T>(beta);

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::gemm<T>(layout, transA, transB, m, n, k, alpha_t,
                                                        pA, lda, str_a, pB, ldb, str_b,
                                                        beta_t, pC, ldc, str_c, batch_size, kernel_epilogue);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n, k, alpha_t,
                                                          pA, lda, str_a, pB, ldb, str_b,
                                                          beta_t, pC, ldc, str_c, batch_size, deps2, kernel_epilogue);
                        });
                });
        }

    } // namespace Dispatch
} // namespace SushiBLAS
