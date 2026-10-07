/**************************************************************************/
/* level1.hpp                                                             */
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
 * @file level1.hpp
 * @brief Defines the reference kernels of the sparse-dense level 1 routines.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include "kernels/sparse/index_list.hpp"
#include "kernels/reference/sparse/structure.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes y[p_k] = alpha * x[k] + y[p_k] for every stored position p_k.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void axpyi(T alpha, const T* x, const IndexList<Index>& positions, T* y, int64_t incy)
            {
                using W = Accumulator<T>;

                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "axpyi was given positions outside its vector or stored twice");

                for (int64_t k = 0; k < positions.count; ++k)
                {
                    T& target = y[positions.at(k) * incy];
                    target = static_cast<T>(W(target) + W(alpha) * W(x[k]));
                }
            }

            /**
             * @brief Computes sum(Conj(x[k]) * y[p_k]) in a ReferenceSum, rounded once on store.
             * @tparam Conj Unconjugated for doti and dotui, Conjugated for dotci.
             * @see kernels/README.md
             */
            template<typename T, typename Index, typename Conj = Unconjugated>
            void doti(const T* x, const IndexList<Index>& positions, const T* y, int64_t incy,
                      T* result)
            {
                using RW = ReferenceAccumulator<T>;

                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "doti was given positions outside its vector or stored twice");

                ReferenceSum<T> sum;
                for (int64_t k = 0; k < positions.count; ++k)
                    sum.add(Conj::apply(static_cast<RW>(x[k])) *
                            static_cast<RW>(y[positions.at(k) * incy]));
                *result = static_cast<T>(sum.value());
            }

            /**
             * @brief Copies y[p_k] into x[k] for every stored position p_k.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void gthr(const T* y, int64_t incy, const IndexList<Index>& positions, T* x)
            {
                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "gthr was given positions outside its vector or stored twice");

                for (int64_t k = 0; k < positions.count; ++k)
                    x[k] = y[positions.at(k) * incy];
            }

            /**
             * @brief Copies y[p_k] into x[k] and then sets y[p_k] to zero.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void gthrz(T* y, int64_t incy, const IndexList<Index>& positions, T* x)
            {
                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "gthrz was given positions outside its vector or stored twice");

                for (int64_t k = 0; k < positions.count; ++k)
                {
                    T& source = y[positions.at(k) * incy];
                    x[k] = source;
                    source = T(0);
                }
            }

            /**
             * @brief Copies x[k] into y[p_k] for every stored position p_k.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void sctr(const T* x, const IndexList<Index>& positions, T* y, int64_t incy)
            {
                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "sctr was given positions outside its vector or stored twice");

                for (int64_t k = 0; k < positions.count; ++k)
                    y[positions.at(k) * incy] = x[k];
            }

            /**
             * @brief Applies the plane rotation x[k] = c * x[k] + s * y[p_k],
             *        y[p_k] = c * y[p_k] - s * x[k], both from the values before the call.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void roti(T* x, const IndexList<Index>& positions, T* y, int64_t incy, RealType<T> c,
                      RealType<T> s)
            {
                using W = Accumulator<T>;
                using WR = Accumulator<RealType<T>>;

                SB_ASSERT(find_defect(positions) == StructureDefect::None,
                          "roti was given positions outside its vector or stored twice");

                for (int64_t k = 0; k < positions.count; ++k)
                {
                    T& target = y[positions.at(k) * incy];
                    const W xk = W(x[k]);
                    const W yk = W(target);
                    x[k] = static_cast<T>(WR(c) * xk + WR(s) * yk);
                    target = static_cast<T>(WR(c) * yk - WR(s) * xk);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
