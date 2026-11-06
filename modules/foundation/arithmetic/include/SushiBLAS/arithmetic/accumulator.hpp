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
