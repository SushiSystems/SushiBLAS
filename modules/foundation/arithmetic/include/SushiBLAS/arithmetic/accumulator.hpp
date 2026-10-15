/**************************************************************************/
/* accumulator.hpp                                                        */
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
 * @file accumulator.hpp
 * @brief Declares AccumulatorOf, the type a BLAS kernel accumulates an element type in.
 * @author Mustafa Garip
 */

#pragma once

#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names the type a sum of T is accumulated in: T itself for every type but HALF.
         * @see modules/foundation/arithmetic/README.md
         */
        template<typename T>
        struct AccumulatorOf
        {
            /** @brief The accumulation type. */
            using type = T;
        };

        /** @brief Accumulates HALF in FLOAT32. */
        template<>
        struct AccumulatorOf<sycl::half>
        {
            /** @brief The accumulation type. */
            using type = float;
        };

        /** @brief Shorthand for `AccumulatorOf<T>::type`. */
        template<typename T>
        using Accumulator = typename AccumulatorOf<T>::type;

    } // namespace Kernels
} // namespace SushiBLAS
