/**************************************************************************/
/* test_packed_syrk.cpp                                                   */
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
 * @file test_packed_syrk.cpp
 * @brief Tests SYRK and HERK on the packed product against the reference kernel.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <kernels/sycl/blas/level3/syrk.hpp>
#include <kernels/reference/blas/level3/syrk.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>
#include <SushiBLAS/kernels/blas/level3/packed_product_plan.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/packed_product.hpp>
#include "../common/test_utils.hpp"
#include "../common/blas_truth.hpp"
#include "../common/packed_product_harness.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the order of C in the exact-data tests.
            constexpr int64_t K_EXACT_ORDER = 29;

            /// @brief Holds the terms of each sum in the exact-data tests.
            constexpr int64_t K_EXACT_TERMS = 13;

            /// @brief Holds alpha of the exact-data tests.
            constexpr int K_EXACT_ALPHA = 2;

            /// @brief Holds beta of the exact-data tests.
            constexpr int K_EXACT_BETA = -1;

            /// @brief Holds the rounded operations a parity bound allows beyond one per term.
            constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Holds the step between the orders tried for an update the profile packs.
            constexpr int64_t K_SEAM_ORDER_STEP = 64;

            /// @brief Holds the largest order tried for an update the profile packs.
            constexpr int64_t K_LARGEST_SEAM_ORDER = 768;

            /**
             * @brief Holds the relative error of storing a part of T: 2^-11 for HALF, the unit
             *        roundoff of the accumulation type otherwise.
             */
            template<typename T>
            constexpr double K_STORE_ROUNDOFF =
                std::is_same_v<T, sycl::half>
                    ? 0x1p-11
                    : unit_roundoff<Kernels::RealType<Kernels::Accumulator<T>>>();

            /** @brief Holds one injected blocking and the compute units it is planned for. */
            struct InjectedBlocking
            {
                const char* name;               /**< Names the blocking in a failure message. */
                Device::CacheBlocking blocking; /**< The budgets handed to the plan. */
                std::size_t compute_units;      /**< The compute units handed to the plan. */
            };

            /** @brief Returns the three blockings the exact-data tests run under, for type T. */
            template<typename T>
            [[nodiscard]] std::vector<InjectedBlocking> injected_blockings()
            {
                return {
                    {"smallest, five units", smallest_blocking(), 5},
                    {"widest, one unit", widest_blocking(), 1},
                    {"ragged, three units", ragged_blocking<T>(), 3},
                };
            }

            /** @brief Returns the real and the imaginary part of value; a real T has no second. */
            template<typename T>
            [[nodiscard]] std::array<double, 2> parts_of(T value) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
                else
                    return {Detail::to_double(value), 0.0};
            }

            /**
             * @brief Returns how far a part of one updated element may lie from the reference.
             * @param terms The terms of each sum; alpha is 1, beta 0 and operand parts at most 1.
             */
            template<typename T>
            [[nodiscard]] double parity_bound(int64_t terms) noexcept
            {
                using Real = Kernels::RealType<Kernels::Accumulator<T>>;

                const double parts = Core::is_complex_v<T> ? 2.0 : 1.0;
                const double magnitude = parts * static_cast<double>(terms);
                return rounding_bound(static_cast<std::size_t>(terms) + K_FIXED_ROUNDINGS,
                                      unit_roundoff<Real>(), magnitude) +
                       K_STORE_ROUNDOFF<T> * magnitude;
            }

            /** @brief Returns how many parts of got lie beyond bound from want, a NaN included. */
            template<typename T>
            [[nodiscard]] std::size_t count_parts_beyond(const std::vector<T>& got,
                                                         const std::vector<T>& want,
                                                         double bound) noexcept
            {
                std::size_t beyond = 0;
                for (std::size_t i = 0; i < want.size(); ++i)
                {
                    const std::array<double, 2> got_parts = parts_of(got[i]);
                    const std::array<double, 2> want_parts = parts_of(want[i]);
                    for (std::size_t part = 0; part < got_parts.size(); ++part)
                        if (!(std::fabs(got_parts[part] - want_parts[part]) <= bound))
                            ++beyond;
                }
                return beyond;
            }

            /** @brief Returns whether the device computes in the precision T needs. */
            template<typename T>
            [[nodiscard]] bool device_runs(const Device::Profile& profile) noexcept
            {
                if (std::is_same_v<T, sycl::half>)
                    return profile.supports_fp16;
                if (std::is_same_v<Kernels::RealType<T>, double>)
                    return profile.supports_fp64;
                return true;
            }

            /**
             * @brief Returns the smallest order the profile packs a square update of T at.
             * @return Zero when no order up to K_LARGEST_SEAM_ORDER is packed.
             */
            template<typename T>
            [[nodiscard]] int64_t packed_order(const Device::Profile& profile) noexcept
            {
                for (int64_t s = K_SEAM_ORDER_STEP; s <= K_LARGEST_SEAM_ORDER;
                     s += K_SEAM_ORDER_STEP)
                    if (Kernels::plan_packed_product<T>(profile, s, s, s, 1).selected)
                        return s;
                return 0;
            }

            /**
             * @brief Owns A and C of one rank-k update and runs both kernels on them.
             * @tparam T The element type.
             */
            template<typename T>
            class SyrkProblem final
            {
                public:
                    /** @brief Allocates A and C with padded leading dimensions and fills them. */
                    SyrkProblem(sycl::queue& q, Core::Layout layout, Core::Uplo uplo,
                                Core::Transpose trans, int64_t n, int64_t k)
                        : queue_(q), layout_(layout), uplo_(uplo), trans_(trans), n_(n), k_(k),
                          a_rows_(Core::is_transposed(trans) ? k : n),
                          a_cols_(Core::is_transposed(trans) ? n : k),
                          lda_(leading(a_rows_, a_cols_)), ldc_(leading(n, n)),
                          str_a_(stride(a_rows_, a_cols_, lda_)), str_c_(stride(n, n, ldc_)),
                          a_(q, static_cast<std::size_t>(str_a_)),
                          c_(q, static_cast<std::size_t>(str_c_)),
                          initial_c_(c_.size())
                    {
                        for (std::size_t i = 0; i < a_.size(); ++i)
                            a_.get()[i] = patterned_element<T>(static_cast<int64_t>(i) / lda_,
                                                               static_cast<int64_t>(i) % lda_, 0);
                        for (std::size_t i = 0; i < initial_c_.size(); ++i)
                            initial_c_[i] = patterned_element<T>(static_cast<int64_t>(i) / ldc_,
                                                                 static_cast<int64_t>(i) % ldc_,
                                                                 2);
                    }

                    /** @brief Returns the C buffer the reference kernel leaves, padding too. */
                    template<typename Conj, typename Scalar>
                    [[nodiscard]] std::vector<T> expected(Scalar alpha, Scalar beta) const
                    {
                        std::vector<T> c = initial_c_;
                        Kernels::Reference::syrk<T, Conj, Scalar>(layout_, uplo_, trans_, n_, k_,
                                                                  alpha, a_.get(), lda_, str_a_,
                                                                  beta, c.data(), ldc_, str_c_, 1);
                        return c;
                    }

                    /** @brief Returns the C buffer the packed product leaves under a blocking. */
                    template<typename Conj, typename Scalar>
                    [[nodiscard]] std::vector<T> packed(const Device::CacheBlocking& blocking,
                                                        std::size_t compute_units,
                                                        Scalar alpha, Scalar beta)
                    {
                        std::memcpy(c_.get(), initial_c_.data(), initial_c_.size() * sizeof(T));

                        const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<T>(
                            blocking, compute_units, n_, n_, k_, 1);

                        const Kernels::OperandAccess a_access =
                            Kernels::operand_access(layout_, trans_, lda_);
                        Kernels::OperandAccess second = a_access.transposed();
                        second.conjugated =
                            (a_access.conjugated != std::is_same_v<Conj, Kernels::Conjugated>);

                        // The scratch starts off its alignment, so the kernel has to align it.
                        const UsmArray<unsigned char> scratch(
                            queue_, plan.scratch_bytes + K_PACKED_TEST_SCRATCH_SKEW);
                        Kernels::SYCL::Detail::packed_product<T>(
                            queue_, plan, n_, n_, k_, a_.get(), a_access, str_a_,
                            a_.get(), second, str_a_, c_.get(), str_c_,
                            scratch.get() + K_PACKED_TEST_SCRATCH_SKEW, {},
                            Kernels::SYCL::Detail::SyrkStore<T, Conj, Scalar>(
                                uplo_ == Core::Uplo::Upper, alpha, beta,
                                Kernels::operand_access(layout_, Core::Transpose::NoTrans, ldc_)))
                            .wait();

                        return std::vector<T>(c_.get(), c_.get() + c_.size());
                    }

                    /** @brief Returns the contents C starts every run from. */
                    [[nodiscard]] const std::vector<T>& initial() const noexcept
                    {
                        return initial_c_;
                    }

                    /** @brief Returns whether an element of the C buffer is in the triangle. */
                    [[nodiscard]] bool is_stored(std::size_t index) const noexcept
                    {
                        const int64_t major = static_cast<int64_t>(index) / ldc_;
                        const int64_t minor = static_cast<int64_t>(index) % ldc_;
                        if (major >= n_ || minor >= n_)
                            return false;

                        const bool row_major = (layout_ == Core::Layout::RowMajor);
                        const int64_t row = row_major ? major : minor;
                        const int64_t col = row_major ? minor : major;
                        return (uplo_ == Core::Uplo::Upper) ? (col >= row) : (col <= row);
                    }

                    /** @brief Returns the index of diagonal element i in the C buffer. */
                    [[nodiscard]] std::size_t diagonal_index(int64_t i) const noexcept
                    {
                        return static_cast<std::size_t>(i * ldc_ + i);
                    }

                private:
                    /** @brief Returns the padded leading dimension of a rows x cols matrix. */
                    [[nodiscard]] int64_t leading(int64_t rows, int64_t cols) const noexcept
                    {
                        return ((layout_ == Core::Layout::ColumnMajor) ? rows : cols) +
                               K_PACKED_TEST_LD_PADDING;
                    }

                    /** @brief Returns the elements of one matrix and the gap after it. */
                    [[nodiscard]] int64_t stride(int64_t rows, int64_t cols,
                                                 int64_t ld) const noexcept
                    {
                        return ld * ((layout_ == Core::Layout::ColumnMajor) ? cols : rows) +
                               K_PACKED_TEST_BATCH_GAP;
                    }

                    /// @brief The queue the packed product's memory lives on.
                    sycl::queue& queue_;

                    /// @brief The storage order of A and C.
                    Core::Layout layout_;

                    /// @brief The triangle of C the update stores.
                    Core::Uplo uplo_;

                    /// @brief op() of A.
                    Core::Transpose trans_;

                    /// @brief Order of C.
                    int64_t n_;

                    /// @brief Terms of each sum.
                    int64_t k_;

                    /// @brief Rows of A as stored.
                    int64_t a_rows_;

                    /// @brief Columns of A as stored.
                    int64_t a_cols_;

                    /// @brief Leading dimension of A.
                    int64_t lda_;

                    /// @brief Leading dimension of C.
                    int64_t ldc_;

                    /// @brief Elements of the A buffer.
                    int64_t str_a_;

                    /// @brief Elements of the C buffer.
                    int64_t str_c_;

                    /// @brief The matrix A.
                    UsmArray<T> a_;

                    /// @brief The matrix C the packed product writes.
                    UsmArray<T> c_;

                    /// @brief The contents C starts every run from.
                    std::vector<T> initial_c_;
            };
        } // namespace

        /** @brief Gives the packed SYRK tests an engine and the queue they submit to. */
        template<typename T>
        class Unit_PackedSyrk : public SushiBLASTest
        {
            protected:
                /** @brief Skips the test when the device does not compute in T's precision. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>(Device::profile_for(queue())))
                        GTEST_SKIP() << "the device does not compute in this precision";
                }

                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::get_context().get_queue();
                }

                /**
                 * @brief Expects the packed update to equal the reference under every blocking
                 *        and to leave every element outside the triangle as it was.
                 */
                template<typename Conj, typename Scalar>
                static void expect_exact(SyrkProblem<T>& problem)
                {
                    const Scalar alpha = Scalar(K_EXACT_ALPHA);
                    const Scalar beta = Scalar(K_EXACT_BETA);
                    const std::vector<T> want =
                        problem.template expected<Conj, Scalar>(alpha, beta);
                    const std::vector<T>& initial = problem.initial();

                    for (const InjectedBlocking& injected : injected_blockings<T>())
                    {
                        SCOPED_TRACE(injected.name);
                        const std::vector<T> got = problem.template packed<Conj, Scalar>(
                            injected.blocking, injected.compute_units, alpha, beta);
                        EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                                  std::size_t{0});

                        std::size_t changed_outside = 0;
                        for (std::size_t i = 0; i < got.size(); ++i)
                            if (!problem.is_stored(i) &&
                                std::memcmp(&got[i], &initial[i], sizeof(T)) != 0)
                                ++changed_outside;
                        EXPECT_EQ(changed_outside, std::size_t{0});
                    }
                }

                /**
                 * @brief Expects one recorded update of order s to pack under ForceSycl and to
                 *        lie within the bound of the ForceReference result.
                 * @param record Records the update of C by A on the fixture's engine.
                 */
                template<typename Record>
                void expect_seam_parity(int64_t s, Record record)
                {
                    const Core::DataType dtype = get_sushiblas_dtype<T>();
                    Tensor A = engine->create_tensor({s, s}, dtype);
                    Tensor sycl_result = engine->create_tensor({s, s}, dtype);
                    Tensor reference_result = engine->create_tensor({s, s}, dtype);
                    fill_random<T>(A);
                    fill_random<T>(sycl_result);

                    // Both results start equal, so the triangle left alone compares too.
                    const std::vector<T> start = read_tensor<T>(sycl_result);
                    queue().memcpy(reference_result.data(), start.data(),
                                   start.size() * sizeof(T)).wait();

                    auto& context = TestEnvironment::get_context();

                    engine->set_mode(ExecutionMode::ForceSycl);
                    const std::size_t live_before = context.live_usm_count();
                    record(A, sycl_result);
                    EXPECT_EQ(context.live_usm_count(), live_before + 1);
                    engine->execute();

                    engine->set_mode(ExecutionMode::ForceReference);
                    record(A, reference_result);
                    engine->execute();

                    const std::vector<T> got = read_tensor<T>(sycl_result);
                    const std::vector<T> want = read_tensor<T>(reference_result);
                    ASSERT_EQ(got.size(), want.size());
                    EXPECT_EQ(count_parts_beyond(got, want, parity_bound<T>(s)), std::size_t{0});
                }
        };

        using PackedSyrkTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                 std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_PackedSyrk, PackedSyrkTypes);

        /** @brief Tests SYRK on both triangles, transposes and layouts for exact parity. */
        TYPED_TEST(Unit_PackedSyrk, BothTrianglesEqualTheReferenceOnExactData)
        {
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
            {
                for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                {
                    for (const Core::Transpose trans : {Core::Transpose::NoTrans,
                                                        Core::Transpose::Trans})
                    {
                        SCOPED_TRACE(layout == Core::Layout::RowMajor ? "row-major"
                                                                       : "column-major");
                        SCOPED_TRACE(uplo == Core::Uplo::Upper ? "upper" : "lower");
                        SCOPED_TRACE(Core::to_string(trans));
                        SyrkProblem<TypeParam> problem(this->queue(), layout, uplo, trans,
                                                       K_EXACT_ORDER, K_EXACT_TERMS);
                        this->template expect_exact<Kernels::Unconjugated, TypeParam>(problem);
                    }
                }
            }
        }

        /** @brief Tests HERK for exact parity and for a diagonal with no imaginary part. */
        TYPED_TEST(Unit_PackedSyrk, HerkConjugatesTheSecondFactor)
        {
            if constexpr (!Core::is_complex_v<TypeParam>)
            {
                GTEST_SKIP() << "HERK of a real type is SYRK, which the test above covers";
            }
            else
            {
                using Scalar = Kernels::RealType<TypeParam>;

                for (const Core::Layout layout : {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor})
                {
                    for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                    {
                        for (const Core::Transpose trans : {Core::Transpose::NoTrans,
                                                            Core::Transpose::ConjTrans})
                        {
                            SCOPED_TRACE(layout == Core::Layout::RowMajor ? "row-major"
                                                                           : "column-major");
                            SCOPED_TRACE(uplo == Core::Uplo::Upper ? "upper" : "lower");
                            SCOPED_TRACE(Core::to_string(trans));
                            SyrkProblem<TypeParam> problem(this->queue(), layout, uplo, trans,
                                                           K_EXACT_ORDER, K_EXACT_TERMS);
                            this->template expect_exact<Kernels::Conjugated, Scalar>(problem);

                            const std::vector<TypeParam> got =
                                problem.template packed<Kernels::Conjugated, Scalar>(
                                    ragged_blocking<TypeParam>(), 3, Scalar(K_EXACT_ALPHA),
                                    Scalar(K_EXACT_BETA));
                            std::size_t complex_diagonals = 0;
                            for (int64_t i = 0; i < K_EXACT_ORDER; ++i)
                                if (parts_of(got[problem.diagonal_index(i)])[1] != 0.0)
                                    ++complex_diagonals;
                            EXPECT_EQ(complex_diagonals, std::size_t{0});
                        }
                    }
                }
            }
        }

        /** @brief Tests which macro-blocks each triangle's store asks the kernel to compute. */
        TYPED_TEST(Unit_PackedSyrk, AStoreWantsOnlyBlocksThatTouchItsTriangle)
        {
            using Store = Kernels::SYCL::Detail::SyrkStore<TypeParam, Kernels::Unconjugated,
                                                           TypeParam>;

            /** @brief Holds one macro-block and which triangles hold part of it. */
            struct Row
            {
                const char* name;  /**< Names the block in a failure message. */
                int64_t row0;      /**< The first row of the block. */
                int64_t rows;      /**< The rows of the block. */
                int64_t col0;      /**< The first column of the block. */
                int64_t cols;      /**< The columns of the block. */
                bool upper_wants;  /**< Whether the upper triangle holds part of it. */
                bool lower_wants;  /**< Whether the lower triangle holds part of it. */
            };

            const std::array<Row, 6> table = {{
                {"above the diagonal", 0, 4, 8, 4, true, false},
                {"below the diagonal", 8, 4, 0, 4, false, true},
                {"one column above the diagonal", 0, 4, 4, 4, true, false},
                {"one row below the diagonal", 4, 4, 0, 4, false, true},
                {"across the diagonal", 4, 4, 2, 4, true, true},
                {"one diagonal element", 5, 1, 5, 1, true, true},
            }};

            const Store upper(true, TypeParam(1), TypeParam(0), Kernels::OperandAccess{});
            const Store lower(false, TypeParam(1), TypeParam(0), Kernels::OperandAccess{});

            for (const Row& row : table)
            {
                SCOPED_TRACE(row.name);
                EXPECT_EQ(upper.wants(row.row0, row.rows, row.col0, row.cols), row.upper_wants);
                EXPECT_EQ(lower.wants(row.row0, row.rows, row.col0, row.cols), row.lower_wants);
            }
        }

        /** @brief Tests SYRK and HERK through the engine at the order the device profile packs. */
        TYPED_TEST(Unit_PackedSyrk, TheKernelPacksWhenTheProfileSelectsIt)
        {
            const Device::Profile& profile = Device::profile_for(this->queue());
            const int64_t s = packed_order<TypeParam>(profile);
            if (s == 0)
                GTEST_SKIP() << "no order up to " << K_LARGEST_SEAM_ORDER
                             << " selects the packed kernel; cache_bytes=" << profile.cache_bytes;

            for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
            {
                SCOPED_TRACE(uplo == Core::Uplo::Upper ? "upper" : "lower");

                this->expect_seam_parity(s, [&](const Tensor& A, Tensor& C)
                {
                    SushiBLAS::BLASOps(*this->engine).syrk(A, C, uplo);
                });
                this->expect_seam_parity(s, [&](const Tensor& A, Tensor& C)
                {
                    SushiBLAS::BLASOps(*this->engine).herk(A, C, uplo);
                });
            }
        }
    } // namespace Test
} // namespace SushiBLAS
