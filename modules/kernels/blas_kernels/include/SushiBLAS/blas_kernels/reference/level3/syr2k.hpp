/**************************************************************************/
/* syr2k.hpp                                                              */
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
 * @file syr2k.hpp
 * @brief Defines the reference kernel of SYR2K and, under a conjugation policy, of HER2K.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes C = alpha * op(A) * Conj(op(B))^T
             *        + Conj(alpha) * op(B) * Conj(op(A))^T + beta * C on one triangle of C.
             * @tparam Conj Unconjugated for SYR2K, Conjugated for HER2K.
             * @tparam Scalar The type of beta: T for SYR2K, the real component type for HER2K.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            void syr2k(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans,
                       int64_t n, int64_t k,
                       T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
                       Scalar beta, T* c, int64_t ldc)
            {
                using RW = ReferenceAccumulator<T>;
                using RWS = ReferenceAccumulator<Scalar>;

                const bool upper = (uplo == Core::Uplo::Upper);
                const OperandAccess op_a = operand_access(layout, trans, lda);
                const OperandAccess op_b = operand_access(layout, trans, ldb);
                const OperandAccess at_c = operand_access(layout, Core::Transpose::NoTrans, ldc);

                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        if ((upper && j < i) || (!upper && j > i))
                            continue;

                        ReferenceSum<T> first;
                        ReferenceSum<T> second;
                        for (int64_t l = 0; l < k; ++l)
                        {
                            first.add(RW(op_a.at(a, i, l)) * Conj::apply(RW(op_b.at(b, j, l))));
                            second.add(RW(op_b.at(b, i, l)) * Conj::apply(RW(op_a.at(a, j, l))));
                        }

                        const RW scaled = RW(alpha) * first.value() +
                                          Conj::apply(RW(alpha)) * second.value();

                        // A zero beta must not read the prior value of C.
                        T& out = c[at_c.index(i, j)];
                        const RW updated = (beta == Scalar(0)) ? scaled
                                                               : RW(scaled + RWS(beta) * RW(out));
                        out = T((i == j) ? Conj::diagonal(updated) : updated);
                    }
                }
            }

            /**
             * @brief Computes a batch of SYR2K or HER2K updates with the reference kernel.
             * @param str_a Element stride between consecutive A matrices; str_b and str_c likewise.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            void syr2k(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans,
                       int64_t n, int64_t k,
                       T alpha, const T* a, int64_t lda, int64_t str_a,
                       const T* b, int64_t ldb, int64_t str_b,
                       Scalar beta, T* c, int64_t ldc, int64_t str_c,
                       int64_t batch_size)
            {
                for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx)
                {
                    syr2k<T, Conj>(layout, uplo, trans, n, k, alpha,
                                   a + b_idx * str_a, lda,
                                   b + b_idx * str_b, ldb,
                                   beta, c + b_idx * str_c, ldc);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
