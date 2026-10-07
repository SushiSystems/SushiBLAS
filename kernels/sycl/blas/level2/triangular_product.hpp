/**************************************************************************/
/* triangular_product.hpp                                                 */
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
 * @file triangular_product.hpp
 * @brief Defines the SYCL triangular matrix-vector product TBMV and TPMV share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Computes x = op(A) * x in place for a triangular op(A) read through
                 *        access, in one sequential task.
                 * @tparam Access BandAccess or PackedAccess of op(A).
                 * @param op_upper Whether op(A) is upper triangular.
                 * @see kernels/README.md
                 */
                template<typename T, typename Access>
                [[nodiscard]] sycl::event triangular_product(sycl::queue& queue,
                                                             const Access& access, bool op_upper,
                                                             bool unit_diag, int64_t n, const T* a,
                                                             T* x, int64_t incx,
                                                             const std::vector<sycl::event>& deps)
                {
                    using W = Accumulator<T>;

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.single_task([=]()
                        {
                            auto row = [=](int64_t i)
                            {
                                const int64_t end = access.end_column(i, n);

                                W sum(0);
                                for (int64_t k = access.first_column(i); k < end; ++k)
                                {
                                    const T aval =
                                        (k == i && unit_diag) ? T(1) : access.at(a, i, k);
                                    sum += multiply(W(aval), W(x[k * incx]));
                                }
                                x[i * incx] = T(sum);
                            };

                            // Row i reads only elements of x that no earlier row has overwritten.
                            if (op_upper)
                            {
                                for (int64_t i = 0; i < n; ++i)
                                    row(i);
                            }
                            else
                            {
                                for (int64_t i = n - 1; i >= 0; --i)
                                    row(i);
                            }
                        });
                    });
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
