/**************************************************************************/
/* test_multiply_add.cpp                                                  */
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
 * @file test_multiply_add.cpp
 * @brief Tests that the multiply-add brick rounds once and adds complex terms part by part.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/multiply_add.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the multiply-add tests once per real type. */
        template<typename R>
        class Small_MultiplyAdd : public ::testing::Test
        {
            protected:
                /**
                 * @brief Returns 1 + 2^-k, whose square is not representable, with k the half
                 *        of R's digits plus one, so the square rounds the 2^-2k term away.
                 */
                [[nodiscard]] static R near_one() noexcept
                {
                    return R(1) + std::ldexp(R(1), -half_digits());
                }

                /** @brief Returns half of R's digits plus one. */
                [[nodiscard]] static int half_digits() noexcept
                {
                    return std::numeric_limits<R>::digits / 2 + 1;
                }
        };

        using MultiplyAddTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_MultiplyAdd, MultiplyAddTypes);

        TYPED_TEST(Small_MultiplyAdd, ARealMultiplyAddRoundsOnce)
        {
            using R = TypeParam;
            const R a = this->near_one();
            const R c = -(R(1) + std::ldexp(R(1), 1 - this->half_digits()));

            EXPECT_EQ(Kernels::Detail::multiply_add(a, a, c),
                      std::ldexp(R(1), -2 * this->half_digits()));
        }

        TYPED_TEST(Small_MultiplyAdd, AVectorMultiplyAddRoundsEachLaneOnce)
        {
            using R = TypeParam;
            using Row = sycl::vec<R, 4>;
            const R c = -(R(1) + std::ldexp(R(1), 1 - this->half_digits()));

            const Row sum =
                Kernels::Detail::multiply_add(Row(this->near_one()), Row(this->near_one()), Row(c));
            for (int lane = 0; lane < 4; ++lane)
                EXPECT_EQ(sum[lane], std::ldexp(R(1), -2 * this->half_digits()));
        }

        TYPED_TEST(Small_MultiplyAdd, AComplexMultiplyAddAddsTheProduct)
        {
            using R = TypeParam;
            const std::complex<R> a(R(1), R(2));
            const std::complex<R> b(R(3), R(-4));
            const std::complex<R> c(R(5), R(6));

            EXPECT_EQ(Kernels::Detail::multiply_add(a, b, c), std::complex<R>(R(16), R(8)));
        }

        TYPED_TEST(Small_MultiplyAdd, TheComplexOverloadMatchesTheVectorParts)
        {
            using R = TypeParam;
            using Row = sycl::vec<R, 2>;
            const std::complex<R> a(this->near_one(), R(-0.75));
            const std::complex<R> b(R(0.5), this->near_one());
            const std::complex<R> c(R(-1), R(0.25));

            Row sum_re(c.real());
            Row sum_im(c.imag());
            Kernels::Detail::multiply_add_parts(sum_re, sum_im, Row(a.real()), Row(a.imag()),
                                        Row(b.real()), Row(b.imag()));
            const std::complex<R> scalar = Kernels::Detail::multiply_add(a, b, c);

            EXPECT_EQ(sum_re[0], scalar.real());
            EXPECT_EQ(sum_im[0], scalar.imag());
        }

        TEST(Small_MultiplyAddInteger, AnIntegerMultiplyAddIsExact)
        {
            EXPECT_EQ(Kernels::Detail::multiply_add(std::int32_t{6}, std::int32_t{7},
                                                    std::int32_t{-2}),
                      std::int32_t{40});
        }
    } // namespace Test
} // namespace SushiBLAS
