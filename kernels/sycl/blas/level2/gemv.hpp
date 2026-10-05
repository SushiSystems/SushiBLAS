/****************************************************************/
/* gemv.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes GEMV, y = alpha*op(A)*x + beta*y, one work-item per output element.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event gemv(sycl::queue& queue, Core::Layout layout, bool transA,
                             int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
                             const T* x, int64_t incx, T beta, T* y, int64_t incy,
                             const std::vector<sycl::event>& deps)
            {
                const int64_t out_len = transA ? n : m;
                const int64_t in_len = transA ? m : n;

                SB_LOG_INFO("SYCL GEMV: {}x{}", m, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(out_len), [=](size_t k)
                    {
                        const int64_t o = static_cast<int64_t>(k);
                        auto A_at = [=](int64_t r, int64_t c) -> T
                        {
                            return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                        };
                        T sum(0);
                        for (int64_t k = 0; k < in_len; ++k)
                        {
                            T aval = transA ? A_at(k, o) : A_at(o, k);
                            sum += aval * x[k * incx];
                        }
                        y[o * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[o * incy];
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
