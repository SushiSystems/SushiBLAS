/**************************************************************************/
/* rotg.hpp                                                               */
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
 * @file rotg.hpp
 * @brief Defines the SYCL kernel of ROTG.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/blas_seam_kernels/givens.hpp>
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
                /** @brief Takes square roots with the SYCL built-in, which a device can run. */
                struct DeviceRoot
                {
                    /** @brief Returns the square root of value. */
                    template<typename R>
                    [[nodiscard]] static R apply(R value) noexcept
                    {
                        return sycl::sqrt(value);
                    }
                };
            } // namespace Detail

            /**
             * @brief Computes ROTG, the rotation that maps (a, b) to (r, 0), in one device task.
             * @param a Receives r.
             * @param b Receives z for a real T; a complex b is read and left as it is.
             * @param c Receives the cosine, in the real component type of T.
             * @param s Receives the sine.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event rotg(sycl::queue& queue, T* a, T* b, RealType<T>* c, T* s,
                                           const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        const GivensRotation<W> g =
                            givens_rotation<Detail::DeviceRoot>(W(*a), W(*b));
                        *a = T(g.r);
                        if constexpr (!Core::is_complex_v<T>)
                            *b = T(g.z);
                        *c = RealType<T>(g.c);
                        *s = T(g.s);
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
