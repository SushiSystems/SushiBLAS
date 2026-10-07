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
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/blas/utils.hpp>
#include <SushiBLAS/engine/blas/scalar.hpp>
#include <SushiBLAS/engine/blas/task_metadata.hpp>
#include <SushiBLAS/engine/blas/argument_check.hpp>
#include <SushiBLAS/engine/blas/matrix_operand.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm_packed.hpp>
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds a GEMM epilogue functor together with the tensors it reads and writes.
         * @tparam Epilogue The functor type, under the contract execute_gemm states for a bare one.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<typename Epilogue>
        class GemmEpilogue
        {
            public:
                /** @brief Wraps functor, declaring nothing yet. */
                explicit GemmEpilogue(Epilogue functor)
                    : functor_(std::move(functor))
                {

                }

                /**
                 * @brief Declares that the epilogue reads operand.
                 * @return `*this`, so declarations chain.
                 * @throws std::invalid_argument if operand has no storage.
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
                 * @throws std::invalid_argument if operand has no storage.
                 */
                GemmEpilogue& writes(const Tensor& operand)
                {
                    require_declared_storage(operand, "writes");
                    operands_.writes(operand);
                    return *this;
                }

                /** @brief Returns the functor, the only part that reaches the device. */
                [[nodiscard]] const Epilogue& functor() const noexcept
                {
                    return functor_;
                }

                /** @brief Returns the tensors declared through reads and writes. */
                [[nodiscard]] const OperandSet& operands() const noexcept
                {
                    return operands_;
                }

                /** @brief Returns the regions declared through reads. */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                read_regions() const noexcept
                {
                    return operands_.read_regions();
                }

                /** @brief Returns the regions declared through writes. */
                [[nodiscard]] const std::vector<SushiRuntime::Core::ResourceRegion>&
                write_regions() const noexcept
                {
                    return operands_.write_regions();
                }

            private:
                /**
                 * @brief Throws if a declared tensor has no storage, so its region keys on nothing.
                 * @param direction "reads" or "writes", for the message.
                 */
                static void require_declared_storage(const Tensor& operand, const char* direction)
                {
                    if (!operand.storage)
                        Detail::throw_operand_error("Dispatch::GemmEpilogue", direction,
                                                    "was given a tensor without storage; its "
                                                    "region keys on no allocation, so the "
                                                    "declaration would order it against nothing");
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
            /** @brief Returns the functor of a bare epilogue: itself. */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue& epilogue_functor(const Epilogue& epilogue) noexcept
            {
                return epilogue;
            }

            /** @brief Returns the functor a GemmEpilogue bundle carries. */
            template<typename Epilogue>
            [[nodiscard]] inline const Epilogue&
            epilogue_functor(const GemmEpilogue<Epilogue>& bundle) noexcept
            {
                return bundle.functor();
            }

            /** @brief Adds nothing, because a bare epilogue declares no operands. */
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
         * @param epilogue Must be device-copyable, capture by value, and never allocate or throw.
         * @throws std::invalid_argument on an operand, flag or scalar the call does not accept.
         * @warning A bare functor that hides a pointer is neither kept alive nor host-checked.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        template<Domain Dom = Domain::Real, typename Epilogue>
        TaskHandle execute_gemm(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                                Core::Transpose transA, Core::Transpose transB,
                                BlasScalar alpha, BlasScalar beta,
                                const char* name, SushiRuntime::Graph::OpID op_id,
                                Epilogue epilogue)
        {
            require_flag(transA, name, "transA");
            require_flag(transB, name, "transB");
            require_dtype(B, A.dtype, name, "B");
            require_dtype(C, A.dtype, name, "C");
            require_real_scalar(alpha, A.dtype, name, "alpha");
            require_real_scalar(beta, A.dtype, name, "beta");

            if constexpr (Dom == Domain::Real)
            {
                if (Core::is_complex_dtype(A.dtype))
                    Detail::throw_operand_error(name, "A",
                                                "has dtype {}, but the epilogue of this call was "
                                                "declared for real tensors; pass "
                                                "Domain::RealAndComplex when the epilogue is "
                                                "well-formed for std::complex",
                                                Core::to_string(A.dtype));
            }

            const MatrixOperand a = describe_matrix_operand(A, name, "A");
            const MatrixOperand b = describe_matrix_operand(B, name, "B");
            const MatrixOperand c = describe_matrix_operand(C, name, "C");
            require_same_layout(a, c, name, "A", "C");
            require_same_layout(b, c, name, "B", "C");

            const bool a_transposed = Core::is_transposed(transA);
            const bool b_transposed = Core::is_transposed(transB);
            const int64_t m = c.rows;
            const int64_t n = c.cols;
            const int64_t k = a_transposed ? a.rows : a.cols;
            require_matrix_extent(a_transposed ? a.cols : a.rows, m, "rows in op(A)", name, "A");
            require_matrix_extent(b_transposed ? b.cols : b.rows, k, "rows in op(B)", name, "B");
            require_matrix_extent(b_transposed ? b.rows : b.cols, n, "columns in op(B)", name, "B");

            const int64_t batch_size = c.batch_count;
            const int64_t str_a = batch_stride(a, batch_size, name, "A");
            const int64_t str_b = batch_stride(b, batch_size, name, "B");
            const int64_t str_c = batch_stride(c, batch_size, name, "C");
            require_same_batch_axes(A, a, C, c, name, "A", "C");
            require_same_batch_axes(B, b, C, c, name, "B", "C");

            // C is read unless beta is zero; see include/SushiBLAS/engine/blas/README.md.
            OperandSet operands;
            operands.reads(A).reads(B);
            if (beta.is_zero())
                operands.writes(C);
            else
                operands.updates(C);
            Detail::append_epilogue_access(epilogue, operands);

            const ScratchBuffer pack = Internal::allocate_packed_product_scratch(
                recorder, C, m, n, k, batch_size, operands);

            // Only the functor enters the kernel closure, never the bundle and its operand set.
            const auto kernel_epilogue = Detail::epilogue_functor(epilogue);

            const Internal::BlasMetadata meta = Internal::BlasMetadata(name, op_id)
                                                    .scalar(alpha).scalar(beta)
                                                    .flag(transA).flag(transB);

            const int64_t lda = a.leading_dimension;
            const int64_t ldb = b.leading_dimension;
            const int64_t ldc = c.leading_dimension;
            const Core::Layout layout = c.layout;
            const void* read_A = A.data_or_null();
            const void* read_B = B.data_or_null();
            void* rw_C = C.data_or_null();

            return Internal::execute_blas_task<Dom>(recorder, meta.get(), operands, A.dtype,
                [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                    sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    using T = decltype(scalar_type);

                    auto pA = static_cast<const T*>(read_A);
                    auto pB = static_cast<const T*>(read_B);
                    auto pC = static_cast<T*>(rw_C);
                    const T alpha_t = alpha.as<T>();
                    const T beta_t = beta.as<T>();

                    return dispatcher.run(q, deps,
                        [=]
                        {
                            Kernels::Reference::gemm<T>(layout, transA, transB, m, n, k, alpha_t,
                                                        pA, lda, str_a, pB, ldb, str_b,
                                                        beta_t, pC, ldc, str_c, batch_size,
                                                        kernel_epilogue);
                        },
                        [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                        {
                            return Kernels::SYCL::gemm<T>(q2, layout, transA, transB, m, n, k,
                                                          alpha_t, pA, lda, str_a, pB, ldb, str_b,
                                                          beta_t, pC, ldc, str_c, batch_size,
                                                          pack.data(), pack.bytes(),
                                                          deps2, kernel_epilogue);
                        });
                });
        }
    } // namespace Dispatch
} // namespace SushiBLAS
