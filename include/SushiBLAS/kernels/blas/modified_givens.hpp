/**************************************************************************/
/* modified_givens.hpp                                                    */
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
/*                                                                        */
/* Transcribed from the reference BLAS of LAPACK, Copyright (c) 1992-2025 */
/* The University of Tennessee and others; modified BSD, see NOTICE.md    */
/**************************************************************************/

/**
 * @file modified_givens.hpp
 * @brief Declares ModifiedGivens and the function that builds one as xROTMG does.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /** @brief Holds the element count of the parameter array xROTMG writes and xROTM reads. */
        inline constexpr int64_t K_MODIFIED_GIVENS_PARAMETERS = 5;

        /**
         * @brief Holds the modified Givens transformation H with the scale factors it leaves.
         * @tparam R float or double.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename R>
        struct ModifiedGivens
        {
            R flag = R(-2); /**< Which entries of H are implied: -2, -1, 0 or 1. */
            R h11 = R(0);   /**< H(1, 1); implied as 1 when flag is 0. */
            R h21 = R(0);   /**< H(2, 1); implied as -1 when flag is 1. */
            R h12 = R(0);   /**< H(1, 2); implied as 1 when flag is 1. */
            R h22 = R(0);   /**< H(2, 2); implied as 1 when flag is 0. */
            R d1 = R(0);    /**< The first scale factor after the transformation. */
            R d2 = R(0);    /**< The second scale factor after the transformation. */
            R x1 = R(0);    /**< The first component after the transformation. */
        };

        namespace Detail
        {
            /** @brief Holds the factor xROTMG rescales by, 2^12. */
            template<typename R>
            inline constexpr R K_ROTMG_GAMMA = R(4096);

            /** @brief Holds the square of K_ROTMG_GAMMA, the top of the scale window. */
            template<typename R>
            inline constexpr R K_ROTMG_GAMMA_SQUARED = R(16777216);

            /** @brief Holds the bottom of the scale window as the reference routine writes it. */
            template<typename R>
            inline constexpr R K_ROTMG_GAMMA_SQUARED_INVERSE = R(5.9604645e-8);

            /** @brief Holds a pass count that no finite scale factor reaches. */
            inline constexpr int K_ROTMG_RESCALE_LIMIT = 128;

            /** @brief Returns what xROTMG gives up with: H and the factors all zero. */
            template<typename R>
            [[nodiscard]] constexpr ModifiedGivens<R> rejected_modified_givens() noexcept
            {
                ModifiedGivens<R> h;
                h.flag = R(-1);
                return h;
            }

            /** @brief Writes the entries flag 0 or 1 implies into h and marks all stored. */
            template<typename R>
            constexpr void store_implied_entries(ModifiedGivens<R>& h) noexcept
            {
                if (h.flag == R(0))
                {
                    h.h11 = R(1);
                    h.h22 = R(1);
                }
                else if (h.flag == R(1))
                {
                    h.h21 = R(-1);
                    h.h12 = R(1);
                }
                h.flag = R(-1);
            }

            /** @brief Returns whether a scale factor magnitude lies outside the scale window. */
            template<typename R>
            [[nodiscard]] constexpr bool is_outside_rotmg_window(R magnitude) noexcept
            {
                return magnitude <= K_ROTMG_GAMMA_SQUARED_INVERSE<R> ||
                       magnitude >= K_ROTMG_GAMMA_SQUARED<R>;
            }

            /** @brief Rescales d1, x1 and the first row of H until d1 lies inside the window. */
            template<typename R>
            constexpr void rescale_first_factor(ModifiedGivens<R>& h) noexcept
            {
                if (h.d1 == R(0))
                    return;

                for (int pass = 0;
                     pass < K_ROTMG_RESCALE_LIMIT && is_outside_rotmg_window(h.d1); ++pass)
                {
                    store_implied_entries(h);
                    if (h.d1 <= K_ROTMG_GAMMA_SQUARED_INVERSE<R>)
                    {
                        h.d1 = h.d1 * K_ROTMG_GAMMA_SQUARED<R>;
                        h.x1 = h.x1 / K_ROTMG_GAMMA<R>;
                        h.h11 = h.h11 / K_ROTMG_GAMMA<R>;
                        h.h12 = h.h12 / K_ROTMG_GAMMA<R>;
                    }
                    else
                    {
                        h.d1 = h.d1 / K_ROTMG_GAMMA_SQUARED<R>;
                        h.x1 = h.x1 * K_ROTMG_GAMMA<R>;
                        h.h11 = h.h11 * K_ROTMG_GAMMA<R>;
                        h.h12 = h.h12 * K_ROTMG_GAMMA<R>;
                    }
                }
            }

            /** @brief Rescales d2 and the second row of H until |d2| lies inside the window. */
            template<typename R>
            constexpr void rescale_second_factor(ModifiedGivens<R>& h) noexcept
            {
                if (h.d2 == R(0))
                    return;

                for (int pass = 0; pass < K_ROTMG_RESCALE_LIMIT; ++pass)
                {
                    const R magnitude = (h.d2 < R(0)) ? -h.d2 : h.d2;
                    if (!is_outside_rotmg_window(magnitude))
                        return;

                    store_implied_entries(h);
                    if (magnitude <= K_ROTMG_GAMMA_SQUARED_INVERSE<R>)
                    {
                        h.d2 = h.d2 * K_ROTMG_GAMMA_SQUARED<R>;
                        h.h21 = h.h21 / K_ROTMG_GAMMA<R>;
                        h.h22 = h.h22 / K_ROTMG_GAMMA<R>;
                    }
                    else
                    {
                        h.d2 = h.d2 / K_ROTMG_GAMMA_SQUARED<R>;
                        h.h21 = h.h21 * K_ROTMG_GAMMA<R>;
                        h.h22 = h.h22 * K_ROTMG_GAMMA<R>;
                    }
                }
            }
        } // namespace Detail

        /**
         * @brief Returns the modified Givens transformation that zeroes the second component of
         *        (sqrt(d1) * x1, sqrt(d2) * y1), by the algorithm of the reference xROTMG.
         * @return flag -2 with d1, d2 and x1 as given when d2 * y1 is zero.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename R>
        [[nodiscard]] constexpr ModifiedGivens<R> modified_givens(R d1, R d2, R x1, R y1) noexcept
        {
            if (d1 < R(0))
                return Detail::rejected_modified_givens<R>();

            ModifiedGivens<R> h;
            h.d1 = d1;
            h.d2 = d2;
            h.x1 = x1;

            const R p2 = d2 * y1;
            if (p2 == R(0))
                return h;

            const R p1 = d1 * x1;
            const R q2 = p2 * y1;
            const R q1 = p1 * x1;
            const R q1_magnitude = (q1 < R(0)) ? -q1 : q1;
            const R q2_magnitude = (q2 < R(0)) ? -q2 : q2;

            if (q1_magnitude > q2_magnitude)
            {
                h.h21 = -y1 / x1;
                h.h12 = p2 / p1;
                const R u = R(1) - h.h12 * h.h21;
                if (!(u > R(0)))
                    return Detail::rejected_modified_givens<R>();

                h.flag = R(0);
                h.d1 = d1 / u;
                h.d2 = d2 / u;
                h.x1 = x1 * u;
            }
            else
            {
                if (q2 < R(0))
                    return Detail::rejected_modified_givens<R>();

                h.flag = R(1);
                h.h11 = p1 / p2;
                h.h22 = x1 / y1;
                const R u = R(1) + h.h11 * h.h22;
                h.d1 = d2 / u;
                h.d2 = d1 / u;
                h.x1 = y1 * u;
            }

            Detail::rescale_first_factor(h);
            Detail::rescale_second_factor(h);
            return h;
        }

        /**
         * @brief Writes a transformation to the five-element parameter array xROTM reads.
         * @param param Receives the flag, and of h11, h21, h12, h22 only those the flag does not
         *        imply; the other elements keep their values.
         */
        template<typename R, typename T>
        constexpr void store_modified_givens(const ModifiedGivens<R>& h, T* param) noexcept
        {
            if (h.flag == R(-1))
            {
                param[1] = T(h.h11);
                param[2] = T(h.h21);
                param[3] = T(h.h12);
                param[4] = T(h.h22);
            }
            else if (h.flag == R(0))
            {
                param[2] = T(h.h21);
                param[3] = T(h.h12);
            }
            else if (h.flag == R(1))
            {
                param[1] = T(h.h11);
                param[4] = T(h.h22);
            }
            param[0] = T(h.flag);
        }
    } // namespace Kernels
} // namespace SushiBLAS
