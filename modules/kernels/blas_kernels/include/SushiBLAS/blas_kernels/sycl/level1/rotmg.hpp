/**************************************************************************/
/* rotmg.hpp                                                              */
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
 * @file rotmg.hpp
 * @brief Defines the SYCL kernel of ROTMG.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/modified_givens.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Computes ROTMG, the modified Givens transformation that zeroes the second
             *        component of (sqrt(d1) * x1, sqrt(d2) * y1), in one device task.
             * @param d1 Receives the first scale factor; d2 receives the second and x1 the first
             *        component after the transformation.
             * @param param Five elements; receives the flag and the entries of H it does not imply.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event rotmg(sycl::queue& queue, T* d1, T* d2, T* x1, const T* y1,
                                            T* param, const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]()
                    {
                        const ModifiedGivens<W> rotation =
                            modified_givens(W(*d1), W(*d2), W(*x1), W(*y1));
                        *d1 = T(rotation.d1);
                        *d2 = T(rotation.d2);
                        *x1 = T(rotation.x1);
                        store_modified_givens(rotation, param);
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
