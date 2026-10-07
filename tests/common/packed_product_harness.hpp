/**************************************************************************/
/* packed_product_harness.hpp                                             */
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
 * @file packed_product_harness.hpp
 * @brief Declares the problem, the data and the blockings the packed product tests share.
 * @author Mustafa Garip
 * @see tests/common/README.md
 */

#pragma once

#include <new>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>
#include <SushiBLAS/kernels/blas/level3/packed_product_plan.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm_packed.hpp>
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the alignment, in bytes, of every array a packed product test allocates. */
        inline constexpr std::size_t K_PACKED_TEST_ALIGNMENT = 128;

        /** @brief Holds the elements a leading dimension exceeds its extent by. */
        inline constexpr int64_t K_PACKED_TEST_LD_PADDING = 3;

        /** @brief Holds the elements between two matrices of a batch beyond one matrix. */
        inline constexpr int64_t K_PACKED_TEST_BATCH_GAP = 5;

        /** @brief Holds the bytes a test moves the scratch off its alignment by. */
        inline constexpr std::size_t K_PACKED_TEST_SCRATCH_SKEW = 8;

        /**
         * @brief Owns one aligned shared USM array and frees it with `sycl::free`.
         * @tparam T The element type; trivially copyable.
         */
        template<typename T>
        class UsmArray final
        {
            public:
                /** @brief Allocates count elements, at least one, on the device of q. */
                UsmArray(sycl::queue& q, std::size_t count)
                    : queue_(q), count_(count),
                      data_(sycl::aligned_alloc_shared<T>(K_PACKED_TEST_ALIGNMENT,
                                                          count ? count : std::size_t{1}, q))
                {
                    if (data_ == nullptr)
                        throw std::bad_alloc();
                }

                /** @brief Frees the array. */
                ~UsmArray()
                {
                    sycl::free(data_, queue_);
                }

                UsmArray(const UsmArray&) = delete;
                UsmArray& operator=(const UsmArray&) = delete;
                UsmArray(UsmArray&&) = delete;
                UsmArray& operator=(UsmArray&&) = delete;

                /** @brief Returns the first element. */
                [[nodiscard]] T* get() const noexcept
                {
                    return data_;
                }

                /** @brief Returns the number of elements asked for. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return count_;
                }

            private:
                /// @brief The queue whose context owns the array.
                sycl::queue& queue_;

                /// @brief The number of elements asked for.
                std::size_t count_;

                /// @brief The first element.
                T* data_;
        };

        /**
         * @brief Returns an integer-valued element that depends on both indices, unsymmetrically.
         * @return A real part in [-3, 3] and, for a complex T, an imaginary part in [-2, 2].
         */
        template<typename T>
        [[nodiscard]] T patterned_element(int64_t row, int64_t col, int64_t salt) noexcept
        {
            const float real = static_cast<float>((3 * row + 5 * col + salt) % 7 - 3);
            if constexpr (Core::is_complex_v<T>)
            {
                using R = typename T::value_type;
                return T(static_cast<R>(real),
                         static_cast<R>((2 * row + 3 * col + salt) % 5 - 2));
            }
            else
            {
                return static_cast<T>(real);
            }
        }

        /** @brief Returns a value whose every part is a quiet NaN. */
        template<typename T>
        [[nodiscard]] T poison_element() noexcept
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            if constexpr (Core::is_complex_v<T>)
            {
                using R = typename T::value_type;
                return T(static_cast<R>(nan), static_cast<R>(nan));
            }
            else
            {
                return static_cast<T>(nan);
            }
        }

        /** @brief Returns how many of count elements differ in their bits between two arrays. */
        template<typename T>
        [[nodiscard]] std::size_t count_bit_mismatches(const T* got, const T* want,
                                                       std::size_t count) noexcept
        {
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < count; ++i)
                if (std::memcmp(got + i, want + i, sizeof(T)) != 0)
                    ++mismatches;
            return mismatches;
        }

        /** @brief Returns the blocking that makes every block one register block, one term deep. */
        [[nodiscard]] inline Device::CacheBlocking smallest_blocking() noexcept
        {
            Device::CacheBlocking blocking;
            blocking.inner_bytes = 1;
            blocking.outer_bytes = 1;
            blocking.stream_bytes = 1;
            blocking.min_operand_bytes = 0;
            blocking.min_register_blocks = 0;
            return blocking;
        }

        /** @brief Returns the blocking whose every block is larger than any matrix. */
        [[nodiscard]] inline Device::CacheBlocking widest_blocking() noexcept
        {
            Device::CacheBlocking blocking = smallest_blocking();
            blocking.inner_bytes = std::numeric_limits<std::size_t>::max();
            blocking.outer_bytes = std::numeric_limits<std::size_t>::max();
            blocking.stream_bytes = std::numeric_limits<std::size_t>::max();
            return blocking;
        }

        /**
         * @brief Returns the blocking of depth 7, two register blocks of rows and three of columns.
         * @tparam T The element type the product runs in.
         */
        template<typename T>
        [[nodiscard]] Device::CacheBlocking ragged_blocking() noexcept
        {
            using W = Kernels::Accumulator<T>;
            using Block = Kernels::PackedRegisterBlock<W>;
            constexpr std::size_t depth = 7;

            Device::CacheBlocking blocking = smallest_blocking();
            blocking.inner_bytes = depth * (Block::K_ROWS + Block::K_COLS) * sizeof(W);
            blocking.outer_bytes = depth * 2 * Block::K_ROWS * sizeof(W);
            blocking.stream_bytes = depth * 3 * Block::K_COLS * sizeof(W);
            return blocking;
        }

        /**
         * @brief Owns the operands of one batched GEMM and runs both kernels on them.
         * @tparam T The element type.
         * @see tests/common/README.md
         */
        template<typename T>
        class GemmProblem final
        {
            public:
                /** @brief Allocates A, B and C with padded leading dimensions and batch strides. */
                GemmProblem(sycl::queue& q, Core::Layout layout,
                            Core::Transpose transA, Core::Transpose transB,
                            int64_t m, int64_t n, int64_t k, int64_t batch_size)
                    : queue_(q), layout_(layout), transA_(transA), transB_(transB),
                      m_(m), n_(n), k_(k), batch_size_(batch_size),
                      a_rows_(Core::is_transposed(transA) ? k : m),
                      a_cols_(Core::is_transposed(transA) ? m : k),
                      b_rows_(Core::is_transposed(transB) ? n : k),
                      b_cols_(Core::is_transposed(transB) ? k : n),
                      lda_(leading(a_rows_, a_cols_)), ldb_(leading(b_rows_, b_cols_)),
                      ldc_(leading(m, n)),
                      str_a_(stride(a_rows_, a_cols_, lda_)),
                      str_b_(stride(b_rows_, b_cols_, ldb_)), str_c_(stride(m, n, ldc_)),
                      a_(q, static_cast<std::size_t>(str_a_ * batch_size)),
                      b_(q, static_cast<std::size_t>(str_b_ * batch_size)),
                      c_(q, static_cast<std::size_t>(str_c_ * batch_size)),
                      initial_c_(c_.size())
                {
                    fill_patterned(0);
                }

                /** @brief Fills A, B and C with integer-valued elements, padding included. */
                void fill_patterned(int64_t salt)
                {
                    for (std::size_t i = 0; i < a_.size(); ++i)
                        a_.get()[i] = patterned_element<T>(static_cast<int64_t>(i) / lda_,
                                                           static_cast<int64_t>(i) % lda_, salt);
                    for (std::size_t i = 0; i < b_.size(); ++i)
                        b_.get()[i] = patterned_element<T>(static_cast<int64_t>(i) / ldb_,
                                                           static_cast<int64_t>(i) % ldb_,
                                                           salt + 1);
                    for (std::size_t i = 0; i < initial_c_.size(); ++i)
                        initial_c_[i] = patterned_element<T>(static_cast<int64_t>(i) / ldc_,
                                                             static_cast<int64_t>(i) % ldc_,
                                                             salt + 2);
                }

                /** @brief Fills A, B and C with values in [-1, 1] drawn from seed. */
                void fill_uniform(std::uint32_t seed)
                {
                    std::mt19937 generator(seed);
                    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                    const auto draw = [&]() -> T
                    {
                        if constexpr (Core::is_complex_v<T>)
                        {
                            using R = typename T::value_type;
                            const R real = static_cast<R>(distribution(generator));
                            return T(real, static_cast<R>(distribution(generator)));
                        }
                        else
                        {
                            return static_cast<T>(distribution(generator));
                        }
                    };

                    for (std::size_t i = 0; i < a_.size(); ++i)
                        a_.get()[i] = draw();
                    for (std::size_t i = 0; i < b_.size(); ++i)
                        b_.get()[i] = draw();
                    for (T& element : initial_c_)
                        element = draw();
                }

                /** @brief Makes every element of C a NaN before each run. */
                void poison_output()
                {
                    for (T& element : initial_c_)
                        element = poison_element<T>();
                }

                /** @brief Returns the whole C buffer the reference kernel leaves, padding too. */
                template<typename Epilogue = Kernels::GemmIdentityEpilogue>
                [[nodiscard]] std::vector<T> expected(T alpha, T beta,
                                                      Epilogue epilogue = Epilogue{}) const
                {
                    std::vector<T> c = initial_c_;
                    Kernels::Reference::gemm<T>(layout_, transA_, transB_, m_, n_, k_, alpha,
                                                a_.get(), lda_, str_a_, b_.get(), ldb_, str_b_,
                                                beta, c.data(), ldc_, str_c_, batch_size_,
                                                epilogue);
                    return c;
                }

                /**
                 * @brief Returns the whole C buffer the packed kernel leaves under a blocking.
                 * @param plan Receives the plan the run used, when not null.
                 */
                template<typename Epilogue = Kernels::GemmIdentityEpilogue>
                [[nodiscard]] std::vector<T> packed(const Device::CacheBlocking& blocking,
                                                    std::size_t compute_units, T alpha, T beta,
                                                    Kernels::PackedProductPlan* plan = nullptr,
                                                    Epilogue epilogue = Epilogue{})
                {
                    std::memcpy(c_.get(), initial_c_.data(), initial_c_.size() * sizeof(T));

                    const Kernels::PackedProductPlan used = Kernels::plan_packed_blocks<T>(
                        blocking, compute_units, m_, n_, k_, batch_size_);
                    if (plan != nullptr)
                        *plan = used;

                    if (used.blocks > 0)
                    {
                        // The scratch starts off its alignment, so the kernel has to align it.
                        const UsmArray<unsigned char> scratch(
                            queue_, used.scratch_bytes + K_PACKED_TEST_SCRATCH_SKEW);
                        Kernels::SYCL::Detail::packed_gemm<T>(
                            queue_, used, m_, n_, k_, alpha,
                            a_.get(), Kernels::operand_access(layout_, transA_, lda_), str_a_,
                            b_.get(), Kernels::operand_access(layout_, transB_, ldb_), str_b_,
                            beta, c_.get(),
                            Kernels::operand_access(layout_, Core::Transpose::NO_TRANS, ldc_),
                            str_c_, batch_size_, scratch.get() + K_PACKED_TEST_SCRATCH_SKEW, {},
                            epilogue).wait();
                    }

                    return std::vector<T>(c_.get(), c_.get() + c_.size());
                }

                /** @brief Returns the whole C buffer the kernel that takes no scratch leaves. */
                template<typename Epilogue = Kernels::GemmIdentityEpilogue>
                [[nodiscard]] std::vector<T> unpacked(T alpha, T beta,
                                                      Epilogue epilogue = Epilogue{})
                {
                    std::memcpy(c_.get(), initial_c_.data(), initial_c_.size() * sizeof(T));

                    Kernels::SYCL::gemm<T, Epilogue>(queue_, layout_, transA_, transB_, m_, n_, k_,
                                                     alpha, a_.get(), lda_, str_a_,
                                                     b_.get(), ldb_, str_b_,
                                                     beta, c_.get(), ldc_, str_c_, batch_size_, {},
                                                     epilogue).wait();

                    return std::vector<T>(c_.get(), c_.get() + c_.size());
                }

                /** @brief Returns the number of elements of the C buffer, padding included. */
                [[nodiscard]] std::size_t output_size() const noexcept
                {
                    return initial_c_.size();
                }

            private:
                /** @brief Returns the padded leading dimension of a rows x cols matrix. */
                [[nodiscard]] int64_t leading(int64_t rows, int64_t cols) const noexcept
                {
                    return ((layout_ == Core::Layout::COLUMN_MAJOR) ? rows : cols) +
                           K_PACKED_TEST_LD_PADDING;
                }

                /** @brief Returns the elements between two matrices of a batch. */
                [[nodiscard]] int64_t stride(int64_t rows, int64_t cols, int64_t ld) const noexcept
                {
                    return ld * ((layout_ == Core::Layout::COLUMN_MAJOR) ? cols : rows) +
                           K_PACKED_TEST_BATCH_GAP;
                }

                /// @brief The queue both kernels' memory lives on.
                sycl::queue& queue_;

                /// @brief The storage order of the three operands.
                Core::Layout layout_;

                /// @brief op() of A.
                Core::Transpose transA_;

                /// @brief op() of B.
                Core::Transpose transB_;

                /// @brief Rows of C.
                int64_t m_;

                /// @brief Columns of C.
                int64_t n_;

                /// @brief Terms of each sum.
                int64_t k_;

                /// @brief Matrices in the batch.
                int64_t batch_size_;

                /// @brief Rows of A as stored.
                int64_t a_rows_;

                /// @brief Columns of A as stored.
                int64_t a_cols_;

                /// @brief Rows of B as stored.
                int64_t b_rows_;

                /// @brief Columns of B as stored.
                int64_t b_cols_;

                /// @brief Leading dimension of A.
                int64_t lda_;

                /// @brief Leading dimension of B.
                int64_t ldb_;

                /// @brief Leading dimension of C.
                int64_t ldc_;

                /// @brief Elements between two matrices of A.
                int64_t str_a_;

                /// @brief Elements between two matrices of B.
                int64_t str_b_;

                /// @brief Elements between two matrices of C.
                int64_t str_c_;

                /// @brief The batch of A.
                UsmArray<T> a_;

                /// @brief The batch of B.
                UsmArray<T> b_;

                /// @brief The batch of C the packed kernel writes.
                UsmArray<T> c_;

                /// @brief The contents C starts every run from.
                std::vector<T> initial_c_;
        };
    } // namespace Test
} // namespace SushiBLAS
