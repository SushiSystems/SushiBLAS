/**************************************************************************/
/* device_math.hpp                                                        */
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
 * @file device_math.hpp
 * @brief Declares the scalar functions a kernel may call, std on the host and sycl on the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>

#if defined(SYCL_LANGUAGE_VERSION)
#    include <sycl/sycl.hpp>
#endif

namespace SushiBLAS
{
    /**
     * @brief Names the scalar functions a kernel may call, std on the host and sycl on the device.
     * @see modules/foundation/support/README.md
     */
    namespace Math
    {
#if defined(__SYCL_DEVICE_ONLY__)
        using ::sycl::asin;
        using ::sycl::atan2;
        using ::sycl::cbrt;
        using ::sycl::copysign;
        using ::sycl::cos;
        using ::sycl::exp;
        using ::sycl::floor;
        using ::sycl::fmod;
        using ::sycl::isfinite;
        using ::sycl::log;
        using ::sycl::pow;
        using ::sycl::sin;
        using ::sycl::sqrt;

        /** @brief Returns the magnitude of a floating-point @p value on the device. */
        template<typename T>
        inline T abs(T value) noexcept
        {
            return ::sycl::fabs(value);
        }
#else
        using std::abs;
        using std::asin;
        using std::atan2;
        using std::cbrt;
        using std::copysign;
        using std::cos;
        using std::exp;
        using std::floor;
        using std::fmod;
        using std::isfinite;
        using std::log;
        using std::pow;
        using std::sin;
        using std::sqrt;
#endif
    } // namespace Math
} // namespace SushiBLAS
