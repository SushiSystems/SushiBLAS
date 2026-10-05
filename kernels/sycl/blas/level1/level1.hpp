/****************************************************************/
/* level1.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// SYCL Level 1 kernels. The elementwise ops (axpy/copy/scal/swap/rot) are
// embarrassingly parallel over the vector, one work-item per element. The
// reductions (asum/nrm2/dot/iamax) run as a single serial device task -
// Level 1 vectors are small relative to Level 3 workloads in this library,
// so a tree reduction isn't worth the complexity yet (can revisit if a
// profiling need for large-vector Level 1 reductions ever shows up).
//
// The parallel ops launch through `Launch::indexed` rather than a bare
// `sycl::range<1>`, so they get the same device-dependent geometry as every
// elementwise op - flat on a CPU, grid-stride on a GPU. `Launch::map` would
// additionally give them the 128-bit wide path, but it addresses its operands
// as `in[i]`, and a Level 1 kernel's contract is `x[i * incx]`: the increments
// are part of the BLAS signature, so the access pattern is not the contiguous
// one `map` assumes. Specialising each kernel on `incx == incy == 1` to reach
// the wide path is a worthwhile follow-up, not something to write blind - the
// wide path only ever engages on a GPU, and this project has no GPU device to
// measure it on today.

#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include <kernels/reference/blas/level1/level1.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            template<typename T>
            [[nodiscard]] sycl::event axpy(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        y[i * incy] += alpha * x[i * incx];
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event copy(sycl::queue& queue, int64_t n, const T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        y[i * incy] = x[i * incx];
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event scal(sycl::queue& queue, int64_t n, T alpha, T* x, int64_t incx, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        x[i * incx] *= alpha;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event swap(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        T tmp = x[i * incx];
                        x[i * incx] = y[i * incy];
                        y[i * incy] = tmp;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event rot(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy,
                             typename Reference::RealOf<T>::type c, typename Reference::RealOf<T>::type s,
                             const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<size_t>(n), [=](size_t k)
                    {
                        const int64_t i = static_cast<int64_t>(k);
                        T xi = x[i * incx];
                        T yi = y[i * incy];
                        x[i * incx] = c * xi + s * yi;
                        y[i * incy] = c * yi - s * xi;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event asum(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        typename Reference::RealOf<T>::type sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += Reference::abs1(x[i * incx]);
                        *result = sum;
                    });
                });
            }

            // The scaling 2-norm, through the library's own accumulator rather than a
            // second copy of the algorithm - see Reduce::ScaledSumOfSquares for what it
            // does and why `sqrt(sum(x*x))`, which is what this used to be, is wrong at
            // both ends of the range. The reference kernel next door deliberately spells
            // the same mathematics a different way; see the comment there.
            //
            // A complex element folds as two reals. `abs1` (|Re| + |Im|) is the magnitude
            // xASUM and IxAMAX are defined on, not the one a 2-norm needs, and using it
            // here was a plain wrong answer for complex input.
            template<typename T>
            [[nodiscard]] sycl::event nrm2(sycl::queue& queue, int64_t n, const T* x, int64_t incx, typename Reference::RealOf<T>::type* result, const std::vector<sycl::event>& deps)
            {
                using R = typename Reference::RealOf<T>::type;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        Reduce::ScaledSumOfSquares<R> acc;
                        for (int64_t i = 0; i < n; ++i)
                        {
                            const T v = x[i * incx];
                            if constexpr (Core::is_complex_v<T>)
                            {
                                acc.accumulate(v.real());
                                acc.accumulate(v.imag());
                            }
                            else
                            {
                                acc.accumulate(v);
                            }
                        }
                        *result = acc.value();
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event dot(sycl::queue& queue, int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        T sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
                        *result = sum;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<T>* x, int64_t incx, const std::complex<T>* y, int64_t incy, std::complex<T>* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        std::complex<T> sum(0);
                        for (int64_t i = 0; i < n; ++i) sum += std::conj(x[i * incx]) * y[i * incy];
                        *result = sum;
                    });
                });
            }

            template<typename T>
            [[nodiscard]] sycl::event iamax(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t* result, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        int64_t best_idx = 0;
                        typename Reference::RealOf<T>::type best_val(0);
                        for (int64_t i = 0; i < n; ++i)
                        {
                            auto v = Reference::abs1(x[i * incx]);
                            if (i == 0 || v > best_val)
                            {
                                best_val = v;
                                best_idx = i;
                            }
                        }
                        *result = best_idx;
                    });
                });
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
