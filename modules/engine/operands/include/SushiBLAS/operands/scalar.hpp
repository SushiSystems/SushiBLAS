/**************************************************************************/
/* scalar.hpp                                                             */
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
 * @file scalar.hpp
 * @brief Declares BLASScalar, the real or complex scalar argument of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds one scalar argument of a BLAS call as a complex double, on the host.
     * @see modules/operations/blas/README.md
     */
    class BLASScalar final
    {
        public:
            /** @brief Converts a real value, so a call site passes a literal. */
            BLASScalar(double value) noexcept
                : real_(value), imag_(0.0)
            {

            }

            /** @brief Converts a complex value. */
            BLASScalar(const std::complex<double>& value) noexcept
                : real_(value.real()), imag_(value.imag())
            {

            }

            /** @brief Returns the real part as given. */
            [[nodiscard]] double real() const noexcept
            {
                return real_;
            }

            /** @brief Returns the imaginary part as given. */
            [[nodiscard]] double imag() const noexcept
            {
                return imag_;
            }

            /** @brief Returns whether the imaginary part is zero. */
            [[nodiscard]] bool is_real() const noexcept
            {
                return imag_ == 0.0;
            }

            /** @brief Returns whether both parts are zero. */
            [[nodiscard]] bool is_zero() const noexcept
            {
                return real_ == 0.0 && imag_ == 0.0;
            }

            /**
             * @brief Returns the scalar in element type T.
             * @tparam T sycl::half, float, double, std::complex<float> or std::complex<double>.
             * @pre is_real() holds when T is a real type.
             * @return The value rounded once to T; HALF is rounded to binary32 first.
             */
            template<typename T>
            [[nodiscard]] T as() const noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    using R = typename T::value_type;
                    return T(static_cast<R>(real_), static_cast<R>(imag_));
                }
                else if constexpr (std::is_same_v<T, sycl::half>)
                {
                    return static_cast<sycl::half>(static_cast<float>(real_));
                }
                else
                {
                    static_assert(std::is_floating_point_v<T>,
                                  "BLASScalar serves HALF, FLOAT32, FLOAT64, COMPLEX32 and "
                                  "COMPLEX64.");
                    return static_cast<T>(real_);
                }
            }

        private:
            /// @brief The real part as given.
            double real_;

            /// @brief The imaginary part as given.
            double imag_;
    };
} // namespace SushiBLAS
