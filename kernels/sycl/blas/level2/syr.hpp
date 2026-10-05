/****************************************************************/
/* syr.hpp                                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file syr.hpp
 * @brief Defines the SYCL kernel of SYR.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes SYR, A += alpha * x * x^T, writing only the requested triangle.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event syr(sycl::queue& queue, Core::Layout layout, bool upper,
                            int64_t n, T alpha, const T* x, int64_t incx,
                            T* a, int64_t lda, const std::vector<sycl::event>& deps)
            {
                SB_LOG_INFO("SYCL SYR: {}x{}", n, n);

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(n)), [=](sycl::id<2> idx)
                    {
                        int64_t i = idx[0], j = idx[1];
                        bool in_triangle = upper ? (j >= i) : (j <= i);
                        if (!in_triangle) return;
                        T& ref = (layout == Core::Layout::ROW_MAJOR) ? a[i * lda + j] : a[i + j * lda];
                        ref += alpha * x[i * incx] * x[j * incx];
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
