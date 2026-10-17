/**************************************************************************/
/* gemm_packed.hpp                                                        */
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
 * @file gemm_packed.hpp
 * @brief Declares the form of the SYCL GEMM that packs its operands where the profile selects it.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_seam_kernels/blas_seam_kernels_log_category.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Stores `epilogue(alpha * sum + beta * C)` for each element of a packed
                 *        GEMM.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename Epilogue>
                class GemmStore final
                {
                    public:
                        /** @brief Binds the scalars, the addressing of C and the epilogue. */
                        GemmStore(T alpha, T beta, OperandAccess c_access,
                                  Epilogue epilogue) noexcept
                            : alpha_(alpha), beta_(beta), c_access_(c_access), epilogue_(epilogue)
                        {

                        }

                        /** @brief Returns true: a GEMM writes every macro-block of C. */
                        [[nodiscard]] constexpr bool
                        wants([[maybe_unused]] int64_t row0, [[maybe_unused]] int64_t rows,
                              [[maybe_unused]] int64_t col0,
                              [[maybe_unused]] int64_t cols) const noexcept
                        {
                            return true;
                        }

                        /**
                         * @brief Writes element (row, col) of one matrix of C from its sum.
                         * @param c The first element of the matrix.
                         */
                        void operator()(T* c, int64_t row, int64_t col, Accumulator<T> sum) const
                        {
                            using W = Accumulator<T>;

                            const int64_t c_idx = c_access_.index(row, col);

                            // A zero beta must not read C, whose prior value may be uninitialised.
                            const T value = (beta_ == T(0))
                                ? T(multiply(W(alpha_), sum))
                                : T(multiply(W(alpha_), sum) + multiply(W(beta_), W(c[c_idx])));

                            c[c_idx] = epilogue_(value, row, col);
                        }

                    private:
                        /// @brief The scale of the product.
                        T alpha_;

                        /// @brief The scale of the prior C; exactly zero skips reading it.
                        T beta_;

                        /// @brief How element (row, col) of C is addressed.
                        OperandAccess c_access_;

                        /// @brief The functor applied to each element before it is stored.
                        Epilogue epilogue_;
                };

                /**
                 * @brief Launches `C = Epilogue(alpha * X * Y + beta * C)` on the packed product.
                 * @tparam AccessA Brick with `T at(const T*, int64_t, int64_t) const`; AccessB too.
                 * @pre packed_product_is_ready(plan, scratch, the bytes behind scratch) holds.
                 * @param c Must not overlap a, b or scratch.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename AccessA, typename AccessB, typename Epilogue>
                [[nodiscard]] sycl::event packed_gemm(
                    sycl::queue& queue, const PackedProductPlan& plan,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, AccessA a_access, int64_t str_a,
                    const T* b, AccessB b_access, int64_t str_b,
                    T beta, T* c, OperandAccess c_access, int64_t str_c,
                    int64_t batch_size, void* scratch,
                    const std::vector<sycl::event>& deps, Epilogue epilogue)
                {
                    SB_LOG_DEBUG(K_BLAS_SEAM_KERNELS_LOG_CATEGORY, "SYCL GEMM [packed]: batch={} {}x{}x{} block={}x{}x{} blocks={} "
                                "workers={} scratch={}B",
                                batch_size, m, n, k, plan.block_rows, plan.block_cols,
                                plan.block_depth, plan.blocks, plan.workers, plan.scratch_bytes);

                    const GemmStore<T, Epilogue> store(alpha, beta, c_access, epilogue);

                    return packed_product<T>(queue, plan, m, n, k, a, a_access, str_a,
                                             b, b_access, str_b, c, str_c, scratch, deps, store);
                }
            } // namespace Detail

            /**
             * @brief Computes GEMM as the form without scratch does, and packs the operands when
             *        the device profile selects the packed kernel.
             * @param scratch Device memory for the packed panels; may be null.
             * @param scratch_bytes Bytes behind scratch; below the plan's demand nothing is packed.
             * @param c Must not overlap A, B or scratch.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            [[nodiscard]] sycl::event gemm(sycl::queue& queue, Core::Layout layout,
                            Core::Transpose transA, Core::Transpose transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size, void* scratch, std::size_t scratch_bytes,
                            const std::vector<sycl::event>& deps,
                            Epilogue epilogue = Epilogue{})
            {
                const PackedProductPlan plan =
                    plan_packed_product<T>(Device::profile_for(queue), m, n, k, batch_size);

                if (!packed_product_is_ready(plan, scratch, scratch_bytes))
                    return gemm<T, Epilogue>(queue, layout, transA, transB, m, n, k,
                                             alpha, a, lda, str_a, b, ldb, str_b,
                                             beta, c, ldc, str_c, batch_size, deps, epilogue);

                return Detail::packed_gemm<T>(
                    queue, plan, m, n, k,
                    alpha, a, operand_access(layout, transA, lda), str_a,
                    b, operand_access(layout, transB, ldb), str_b,
                    beta, c, operand_access(layout, Core::Transpose::NoTrans, ldc), str_c,
                    batch_size, scratch, deps, epilogue);
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
