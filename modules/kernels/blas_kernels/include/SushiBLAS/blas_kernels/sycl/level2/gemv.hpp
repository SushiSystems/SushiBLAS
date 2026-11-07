/**************************************************************************/
/* gemv.hpp                                                               */
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
 * @file gemv.hpp
 * @brief Defines the SYCL kernel of GEMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/blas_kernels/blas_kernels_log_category.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/blend_index.hpp>
#include <SushiBLAS/blas_kernels/matrix_vector_index.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/row_reduction.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes GEMV, y = alpha*op(A)*x + beta*y, one output element per work-item
             *        or per work-group.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event gemv(sycl::queue& queue, Core::Layout layout,
                                           Core::Transpose trans, int64_t m, int64_t n, T alpha,
                                           const T* a, int64_t lda, const T* x, int64_t incx,
                                           T beta, T* y, int64_t incy,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;
                const OperandAccess access = operand_access(layout, trans, lda);
                const bool transposed = Core::is_transposed(trans);
                const Indexing::VectorExtents extents = Indexing::vector_extents(transposed, m, n);
                const int64_t out_len = extents.out_len;
                const int64_t in_len = extents.in_len;

                const Device::Profile& profile = Device::profile_for(queue);
                const bool row_contiguous =
                    Indexing::rows_are_contiguous(layout == Core::Layout::RowMajor, transposed);

                // Fewer rows than one work-group per compute unit leave units idle; see kernels/README.md.
                const bool few_rows = !profile.is_cpu &&
                    Indexing::rows_underfill_device(out_len, profile.compute_units,
                                                    Device::work_group_size(profile));
                const std::size_t width = ((row_contiguous || few_rows) && out_len > 0)
                    ? Detail::row_reduction_width(profile, in_len)
                    : std::size_t{0};

                if (width != 0)
                {
                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        Detail::reduce_rows_by_group<W>(h, out_len, in_len, width,
                            [=](int64_t o, int64_t k) -> W
                            {
                                return multiply(W(access.at(a, o, k)), W(x[k * incx]));
                            },
                            [=](int64_t o, W sum)
                            {
                                y[o * incy] = T(Indexing::blend_with_prior<W, W>(
                                    multiply(W(alpha), sum), beta, y[o * incy]));
                            });
                    });
                }

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(out_len), [=](size_t k)
                    {
                        const int64_t o = static_cast<int64_t>(k);
                        W sum(0);
                        for (int64_t k = 0; k < in_len; ++k)
                            sum += multiply(W(access.at(a, o, k)), W(x[k * incx]));
                        y[o * incy] = T(Indexing::blend_with_prior<W, W>(
                            multiply(W(alpha), sum), beta, y[o * incy]));
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
