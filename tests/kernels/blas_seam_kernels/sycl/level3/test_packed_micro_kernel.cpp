/**************************************************************************/
/* test_packed_micro_kernel.cpp                                           */
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
 * @file test_packed_micro_kernel.cpp
 * @brief Tests the micro-kernel of a packed product on one register block per accumulation type.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <random>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/arithmetic/multiply_add.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_micro_kernel.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the most terms a test accumulates.
            constexpr int64_t K_MAX_DEPTH = 37;

            /// @brief Holds the terms of the tests that do not need K_MAX_DEPTH.
            constexpr int64_t K_SHORT_DEPTH = 5;

            /// @brief Holds the seed of the slices compared against the bare product.
            constexpr std::uint32_t K_BARE_PRODUCT_SEED = 20261007u;

            /// @brief Holds the seed of the slices compared against Kernels::Detail::multiply_add.
            constexpr std::uint32_t K_MULTIPLY_SEED = 20261008u;

            /// @brief Holds the seed of the tile a depth of zero must leave.
            constexpr std::uint32_t K_UNTOUCHED_SEED = 20261009u;

            /// @brief Holds the exponent p of the 2^p that absorbs a 1 in FLOAT32.
            constexpr int K_FLOAT_ABSORBING_EXPONENT = 30;

            /// @brief Holds the exponent p of the 2^p that absorbs a 1 in FLOAT64.
            constexpr int K_DOUBLE_ABSORBING_EXPONENT = 60;

            /// @brief Holds the value the real after the tile has before the kernel runs.
            constexpr float K_SENTINEL = -777.0f;

            /**
             * @brief Returns a value of W from two parts; a real W drops the imaginary one.
             */
            template<typename W>
            [[nodiscard]] W value_of(Kernels::RealType<W> real,
                                     [[maybe_unused]] Kernels::RealType<W> imaginary) noexcept
            {
                if constexpr (Core::is_complex_v<W>)
                    return W(real, imaginary);
                else
                    return real;
            }

            /**
             * @brief Owns the aligned slices of both factors and one micro-tile of type W.
             * @tparam W The accumulation type: float, double or the std::complex of either.
             */
            template<typename W>
            class MicroProblem final
            {
                public:
                    /** @brief The type the slices and the tile store. */
                    using R = Kernels::RealType<W>;

                    /** @brief The register block of W. */
                    using Block = Kernels::PackedRegisterBlock<W>;

                    /** @brief The sums of one register block, row by row. */
                    using Sums = std::array<W, Block::K_ROWS * Block::K_COLS>;

                    /** @brief Zeroes the slices and the tile and sets the sentinel after it. */
                    MicroProblem() noexcept
                    {
                        tile_[Block::K_TILE_REALS] = static_cast<R>(K_SENTINEL);
                    }

                    /** @brief Sets element i of slice l of the first factor. */
                    void set_a(int64_t l, int64_t i, W value) noexcept
                    {
                        store(a_.data() + l * Block::K_PARTS * Block::K_ROWS, Block::K_ROWS, i,
                              value);
                    }

                    /** @brief Sets element j of slice l of the second factor. */
                    void set_b(int64_t l, int64_t j, W value) noexcept
                    {
                        store(b_.data() + l * Block::K_PARTS * Block::K_COLS, Block::K_COLS, j,
                              value);
                    }

                    /** @brief Sets the sum at (i, j) of the tile. */
                    void set_sum(int64_t i, int64_t j, W value) noexcept
                    {
                        store(tile_.data(), Block::K_ROWS * Block::K_COLS, i * Block::K_COLS + j,
                              value);
                    }

                    /** @brief Returns element i of slice l of the first factor. */
                    [[nodiscard]] W a(int64_t l, int64_t i) const noexcept
                    {
                        return load(a_.data() + l * Block::K_PARTS * Block::K_ROWS, Block::K_ROWS,
                                    i);
                    }

                    /** @brief Returns element j of slice l of the second factor. */
                    [[nodiscard]] W b(int64_t l, int64_t j) const noexcept
                    {
                        return load(b_.data() + l * Block::K_PARTS * Block::K_COLS, Block::K_COLS,
                                    j);
                    }

                    /** @brief Returns the sum at (i, j) of the tile. */
                    [[nodiscard]] W sum(int64_t i, int64_t j) const noexcept
                    {
                        return load(tile_.data(), Block::K_ROWS * Block::K_COLS,
                                    i * Block::K_COLS + j);
                    }

                    /** @brief Returns every sum of the tile, row by row. */
                    [[nodiscard]] Sums sums() const noexcept
                    {
                        Sums all;
                        for (int64_t i = 0; i < Block::K_ROWS; ++i)
                            for (int64_t j = 0; j < Block::K_COLS; ++j)
                                all[static_cast<std::size_t>(i * Block::K_COLS + j)] = sum(i, j);
                        return all;
                    }

                    /**
                     * @brief Returns the sums the tile holds after depth terms are added in
                     *        ascending order, each by Kernels::Detail::multiply_add.
                     */
                    [[nodiscard]] Sums expected_sums(int64_t depth) const
                    {
                        Sums expected = sums();
                        for (int64_t i = 0; i < Block::K_ROWS; ++i)
                            for (int64_t j = 0; j < Block::K_COLS; ++j)
                                for (int64_t l = 0; l < depth; ++l)
                                {
                                    W& total = expected[static_cast<std::size_t>(
                                        i * Block::K_COLS + j)];
                                    total = Kernels::Detail::multiply_add(a(l, i), b(l, j), total);
                                }
                        return expected;
                    }

                    /** @brief Fills depth slices of both factors and the tile from a seed. */
                    void fill_uniform(std::uint32_t seed, int64_t depth)
                    {
                        std::mt19937 generator(seed);
                        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                        const auto draw = [&]() -> W
                        {
                            const R real = static_cast<R>(distribution(generator));
                            return value_of<W>(real, static_cast<R>(distribution(generator)));
                        };

                        for (int64_t l = 0; l < depth; ++l)
                        {
                            for (int64_t i = 0; i < Block::K_ROWS; ++i)
                                set_a(l, i, draw());
                            for (int64_t j = 0; j < Block::K_COLS; ++j)
                                set_b(l, j, draw());
                        }
                        for (int64_t i = 0; i < Block::K_ROWS; ++i)
                            for (int64_t j = 0; j < Block::K_COLS; ++j)
                                set_sum(i, j, draw());
                    }

                    /** @brief Runs the micro-kernel over the first depth slices. */
                    void run(int64_t depth) noexcept
                    {
                        Kernels::SYCL::Detail::packed_micro_kernel<W>(depth, a_.data(), b_.data(),
                                                                      tile_.data());
                    }

                    /** @brief Returns the real stored directly after the tile. */
                    [[nodiscard]] R real_after_tile() const noexcept
                    {
                        return tile_[Block::K_TILE_REALS];
                    }

                private:
                    /** @brief Writes a value at an offset of a block whose parts are planes. */
                    static void store(R* block, [[maybe_unused]] int64_t plane_reals,
                                      int64_t offset, W value) noexcept
                    {
                        if constexpr (Core::is_complex_v<W>)
                        {
                            block[offset] = value.real();
                            block[plane_reals + offset] = value.imag();
                        }
                        else
                        {
                            block[offset] = value;
                        }
                    }

                    /** @brief Returns the value at an offset of a block whose parts are planes. */
                    [[nodiscard]] static W load(const R* block,
                                                [[maybe_unused]] int64_t plane_reals,
                                                int64_t offset) noexcept
                    {
                        if constexpr (Core::is_complex_v<W>)
                        {
                            return W(block[offset], block[plane_reals + offset]);
                        }
                        else
                        {
                            return block[offset];
                        }
                    }

                    /// @brief K_MAX_DEPTH slices of the first factor.
                    alignas(Kernels::K_PACKED_ALIGNMENT)
                    std::array<R, K_MAX_DEPTH * Block::K_PARTS * Block::K_ROWS> a_{};

                    /// @brief K_MAX_DEPTH slices of the second factor.
                    alignas(Kernels::K_PACKED_ALIGNMENT)
                    std::array<R, K_MAX_DEPTH * Block::K_PARTS * Block::K_COLS> b_{};

                    /// @brief One micro-tile and the sentinel after it.
                    alignas(Kernels::K_PACKED_ALIGNMENT)
                    std::array<R, Block::K_TILE_REALS + 1> tile_{};
            };

            /** @brief Returns how many sums differ in their bits between two blocks. */
            template<typename W, std::size_t N>
            [[nodiscard]] std::size_t count_bit_mismatches(const std::array<W, N>& got,
                                                           const std::array<W, N>& want) noexcept
            {
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < N; ++i)
                    if (std::memcmp(&got[i], &want[i], sizeof(W)) != 0)
                        ++mismatches;
                return mismatches;
            }
        }

        /**
         * @brief Names the accumulation type a micro-kernel runs in.
         * @tparam W float, double or the std::complex of either.
         */
        template<typename W>
        class Medium_PackedMicroKernel : public ::testing::Test
        {

        };

        using PackedMicroKernelTypes =
            ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Medium_PackedMicroKernel, PackedMicroKernelTypes);

        /** @brief Tests that every sum gains the outer products of the slices, exactly. */
        TYPED_TEST(Medium_PackedMicroKernel, TheTileGainsTheOuterProducts)
        {
            using W = TypeParam;
            using R = Kernels::RealType<W>;
            using Block = Kernels::PackedRegisterBlock<W>;

            MicroProblem<W> problem;
            for (int64_t l = 0; l < K_SHORT_DEPTH; ++l)
            {
                for (int64_t i = 0; i < Block::K_ROWS; ++i)
                    problem.set_a(l, i, value_of<W>(static_cast<R>((l + 2 * i) % 5 - 2),
                                                    static_cast<R>((2 * l + i) % 3 - 1)));
                for (int64_t j = 0; j < Block::K_COLS; ++j)
                    problem.set_b(l, j, value_of<W>(static_cast<R>((3 * l + j) % 7 - 3),
                                                    static_cast<R>((l + 2 * j) % 5 - 2)));
            }
            for (int64_t i = 0; i < Block::K_ROWS; ++i)
                for (int64_t j = 0; j < Block::K_COLS; ++j)
                    problem.set_sum(i, j, value_of<W>(static_cast<R>(10 * i + j + 1),
                                                      static_cast<R>(-(i + j + 1))));

            const auto expected = problem.expected_sums(K_SHORT_DEPTH);
            problem.run(K_SHORT_DEPTH);

            for (int64_t i = 0; i < Block::K_ROWS; ++i)
                for (int64_t j = 0; j < Block::K_COLS; ++j)
                    EXPECT_EQ(problem.sum(i, j),
                              expected[static_cast<std::size_t>(i * Block::K_COLS + j)])
                        << "row " << i << ", column " << j;
        }

        /** @brief Tests that term l is added before term l + 1, in each part of a sum. */
        TYPED_TEST(Medium_PackedMicroKernel, TheTermsAreAddedInAscendingOrder)
        {
            using W = TypeParam;
            using R = Kernels::RealType<W>;
            using Block = Kernels::PackedRegisterBlock<W>;

            const R absorbing = std::ldexp(R(1), std::is_same_v<R, float>
                                                     ? K_FLOAT_ABSORBING_EXPONENT
                                                     : K_DOUBLE_ABSORBING_EXPONENT);

            MicroProblem<W> cancelling;
            cancelling.set_a(0, 0, value_of<W>(absorbing, absorbing));
            cancelling.set_a(1, 0, value_of<W>(R(1), R(1)));
            cancelling.set_a(2, 0, value_of<W>(-absorbing, -absorbing));
            for (int64_t l = 0; l < 3; ++l)
                for (int64_t j = 0; j < Block::K_COLS; ++j)
                    cancelling.set_b(l, j, value_of<W>(R(1), R(0)));
            cancelling.run(3);

            for (int64_t j = 0; j < Block::K_COLS; ++j)
                EXPECT_EQ(cancelling.sum(0, j), W(0)) << "column " << j;

            if constexpr (!Core::is_complex_v<W>)
            {
                MicroProblem<W> problem;
                problem.fill_uniform(K_BARE_PRODUCT_SEED, K_MAX_DEPTH);
                const auto expected = problem.expected_sums(K_MAX_DEPTH);
                problem.run(K_MAX_DEPTH);

                EXPECT_EQ(count_bit_mismatches(problem.sums(), expected), std::size_t{0});
            }
        }

        /** @brief Tests that each term is added by Kernels::Detail::multiply_add, to the bit. */
        TYPED_TEST(Medium_PackedMicroKernel, EachTermIsAddedByTheFusedMultiplyAdd)
        {
            using W = TypeParam;

            MicroProblem<W> problem;
            problem.fill_uniform(K_MULTIPLY_SEED, K_MAX_DEPTH);
            const auto expected = problem.expected_sums(K_MAX_DEPTH);
            problem.run(K_MAX_DEPTH);

            EXPECT_EQ(count_bit_mismatches(problem.sums(), expected), std::size_t{0});
        }

        /** @brief Tests that a depth of zero leaves every bit of the tile. */
        TYPED_TEST(Medium_PackedMicroKernel, ADepthOfZeroLeavesTheTile)
        {
            using W = TypeParam;

            MicroProblem<W> problem;
            problem.fill_uniform(K_UNTOUCHED_SEED, K_SHORT_DEPTH);
            const auto before = problem.sums();
            problem.run(0);

            EXPECT_EQ(count_bit_mismatches(problem.sums(), before), std::size_t{0});
        }

        /** @brief Tests that the real after the tile is left alone. */
        TYPED_TEST(Medium_PackedMicroKernel, NothingIsWrittenPastTheTile)
        {
            using W = TypeParam;
            using R = Kernels::RealType<W>;

            MicroProblem<W> problem;
            problem.fill_uniform(K_UNTOUCHED_SEED, K_SHORT_DEPTH);
            problem.run(K_SHORT_DEPTH);

            EXPECT_EQ(problem.real_after_tile(), static_cast<R>(K_SENTINEL));
        }
    } // namespace Test
} // namespace SushiBLAS
