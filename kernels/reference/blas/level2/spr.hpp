/**************************************************************************/
/* spr.hpp                                                                */
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
 * @file spr.hpp
 * @brief Defines the reference kernel of SPR and HPR.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiBLAS/kernels/blas/packed_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SPR, A += alpha * x * x^T, on the uplo triangle of A packed in ap.
             * @tparam Conj Unconjugated for SPR, Conjugated for HPR, which adds alpha * x * x^H.
             * @tparam Scalar T for SPR, the real component type of T for HPR.
             * @see kernels/README.md
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            void spr(Core::Layout layout, Core::Uplo uplo, int64_t n, Scalar alpha, const T* x,
                     int64_t incx, T* ap)
            {
                static_assert(std::is_same_v<Scalar, T> || std::is_same_v<Scalar, RealType<T>>,
                              "alpha is a T for SPR and the real component type of T for HPR.");

                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;

                const PackedAccess access =
                    packed_access(layout, uplo, Core::Transpose::NoTrans, n);

                for (int64_t i = 0; i < n; ++i)
                {
                    const int64_t end = access.end_column(i, n);
                    for (int64_t j = access.first_column(i); j < end; ++j)
                    {
                        T& ref = ap[access.index(i, j)];
                        const W updated = W(ref) + WS(alpha) * W(x[i * incx]) *
                                                       Conj::apply(W(x[j * incx]));
                        ref = T((i == j) ? Conj::diagonal(updated) : updated);
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
