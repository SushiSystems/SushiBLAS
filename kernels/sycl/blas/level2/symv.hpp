/****************************************************************/
/* symv.hpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file symv.hpp
 * @brief Defines the SYCL kernel of SYMV.
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
            * @brief SYMV: y = alpha*A*x + beta*y, A symmetric, only the requested
            * triangle physically read. One work-item per output element.
            *
            * Same launch shape and the same reasoning as `gemv` above: `Launch::indexed`
            * for the device-dependent grid, and not `Launch::map` because each output is
            * a reduction over a row rather than a function of one input element.
            */
            template<typename T>
            [[nodiscard]] sycl::event symv(sycl::queue& queue, Core::Layout layout, bool upper,
                             int64_t n, T alpha, const T* a, int64_t lda,
                             const T* x, int64_t incx, T beta, T* y, int64_t incy,
                             const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL SYMV: {}x{}", n, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        auto A_at = [=](int64_t r, int64_t c) -> T
                        {
                            return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                        };
                        auto sym_at = [=](int64_t r, int64_t c) -> T
                        {
                            if (upper) return (r <= c) ? A_at(r, c) : A_at(c, r);
                            return (r >= c) ? A_at(r, c) : A_at(c, r);
                        };
                        T sum(0);
                        for (int64_t j = 0; j < n; ++j) sum += sym_at(i, j) * x[j * incx];
                        y[i * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i * incy];
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
