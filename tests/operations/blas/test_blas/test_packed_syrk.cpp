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
 * @file test_packed_syrk.cpp
 * @brief Tests SYRK and HERK on the packed product against the reference kernel.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syrk.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include "blas_rows.hpp"

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

            /// @brief Holds the two triangles C can be stored in.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /**
             * @brief Holds the relative error of storing a part of T: 2^-11 for HALF, the unit
             *        roundoff of the accumulation type otherwise.
             */
            template<typename T>
            constexpr double K_STORE_ROUNDOFF =
                std::is_same_v<T, sycl::half>
                    ? K_HALF_UNIT_ROUNDOFF
                    : unit_roundoff<Kernels::RealType<Kernels::Accumulator<T>>>();

            /** @brief Names one of the three blockings the exact-data rows inject. */
            enum class BlockingKind : std::uint8_t
            {
                Smallest, /**< The smallest blocking, planned for five compute units. */
                Widest,   /**< The widest blocking, planned for one compute unit. */
                Ragged    /**< A blocking that leaves ragged edges, planned for three compute units. */
            };

            /** @brief Holds the three blockings, in the order of BlockingKind. */
            constexpr BlockingKind K_BLOCKINGS[] = {BlockingKind::Smallest, BlockingKind::Widest,
                                                    BlockingKind::Ragged};

            /** @brief Names the routine a seam row records. */
            enum class Routine : std::uint8_t
            {
                Syrk, /**< The symmetric rank-k update. */
                Herk  /**< The Hermitian rank-k update. */
            };

            /** @brief Holds one injected blocking and the compute units it is planned for. */
            struct InjectedBlocking
            {
                Device::CacheBlocking blocking; /**< The budgets handed to the plan. */
                std::size_t compute_units;      /**< The compute units handed to the plan. */
            };

            /** @brief Returns the blocking of a kind for type T. */
            template<typename T>
            [[nodiscard]] InjectedBlocking injected_blocking(BlockingKind kind)
            {
                switch (kind)
                {
                    case BlockingKind::Smallest:
                        return {smallest_blocking(), 5};
                    case BlockingKind::Widest:
                        return {widest_blocking(), 1};
                    case BlockingKind::Ragged:
                        break;
                }
                return {ragged_blocking<T>(), 3};
            }

            /** @brief Returns the word that names a blocking in a row name. */
            [[nodiscard]] std::string blocking_word(BlockingKind kind)
            {
                return kind == BlockingKind::Smallest ? "Smallest"
                       : kind == BlockingKind::Widest ? "Widest"
                                                      : "Ragged";
            }

            /** @brief Holds one exact-data row: the type, the storage, the triangle, op(A) and the blocking. */
            struct ExactRow
            {
                Core::DataType type;   /**< The element type of A and C. */
                Core::Layout layout;   /**< The storage order of A and C. */
                Core::Uplo uplo;       /**< The triangle of C the update stores. */
                Core::Transpose trans; /**< The form of A the update multiplies. */
                BlockingKind blocking; /**< The blocking handed to the plan. */
            };

            /** @brief Holds one seam row: the type, the triangle and the routine. */
            struct SeamRow
            {
                Core::DataType type; /**< The element type of A and C. */
                Core::Uplo uplo;     /**< The triangle of C the update stores. */
                Routine routine;     /**< The routine the seam records. */
            };

            /** @brief Returns every exact-data row over the given types and the two forms of A. */
            [[nodiscard]] std::vector<ExactRow> exact_rows(const std::vector<Core::DataType>& types,
                                                           Core::Transpose second_form)
            {
                std::vector<ExactRow> rows;
                for (const Core::DataType type : types)
                    for (const Core::Layout layout : K_ROW_LAYOUTS)
                        for (const Core::Uplo uplo : K_TRIANGLES)
                            for (const Core::Transpose trans :
                                 {Core::Transpose::NoTrans, second_form})
                                for (const BlockingKind blocking : K_BLOCKINGS)
                                    rows.push_back({type, layout, uplo, trans, blocking});
                return rows;
            }

            /** @brief Returns every seam row. */
            [[nodiscard]] std::vector<SeamRow> seam_rows()
            {
                std::vector<SeamRow> rows;
                for (const Core::DataType type : K_ALL_TYPES)
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        for (const Routine routine : {Routine::Syrk, Routine::Herk})
                            rows.push_back({type, uplo, routine});
                return rows;
            }

            /** @brief Returns the name of an exact-data row. */
            [[nodiscard]] std::string exact_row_name(const ::testing::TestParamInfo<ExactRow>& info)
            {
                const ExactRow& row = info.param;
                return std::string(Core::to_string(row.type)) + "_" + word(row.layout) + "_" +
                       word(row.uplo) + "_" + word(row.trans) + "_" + blocking_word(row.blocking);
            }

            /** @brief Returns the name of a seam row. */
            [[nodiscard]] std::string seam_row_name(const ::testing::TestParamInfo<SeamRow>& info)
            {
                const SeamRow& row = info.param;
                return std::string(Core::to_string(row.type)) + "_" + word(row.uplo) + "_" +
                       (row.routine == Routine::Syrk ? "Syrk" : "Herk");
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
                        const USMArray<unsigned char> scratch(
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
                    USMArray<T> a_;

                    /// @brief The matrix C the packed product writes.
                    USMArray<T> c_;

                    /// @brief The contents C starts every run from.
                    std::vector<T> initial_c_;
            };

            /**
             * @brief Expects the packed update to equal the reference under the blocking of a row
             *        and to leave every element outside the triangle as it was.
             * @return The C buffer the packed update left.
             */
            template<typename T, typename Conj, typename Scalar>
            std::vector<T> expect_exact(SyrkProblem<T>& problem, BlockingKind kind)
            {
                const Scalar alpha = Scalar(K_EXACT_ALPHA);
                const Scalar beta = Scalar(K_EXACT_BETA);
                const std::vector<T> want = problem.template expected<Conj, Scalar>(alpha, beta);
                const std::vector<T>& initial = problem.initial();

                const InjectedBlocking injected = injected_blocking<T>(kind);
                std::vector<T> got = problem.template packed<Conj, Scalar>(
                    injected.blocking, injected.compute_units, alpha, beta);
                EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), want.size()),
                          std::size_t{0});

                std::size_t changed_outside = 0;
                for (std::size_t i = 0; i < got.size(); ++i)
                    if (!problem.is_stored(i) && std::memcmp(&got[i], &initial[i], sizeof(T)) != 0)
                        ++changed_outside;
                EXPECT_EQ(changed_outside, std::size_t{0});
                return got;
            }
        } // namespace

        /** @brief Gives the packed SYRK tests an engine and the queue they submit to. */
        class PackedSyrkEngine : public SushiBLASTest
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }

                /**
                 * @brief Expects one recorded update of order s to pack under ForceSYCL and to
                 *        lie within the bound of the ForceReference result.
                 * @param record Records the update of C by A on the fixture's engine.
                 */
                template<typename T, typename Record>
                void expect_seam_parity(int64_t s, Record record)
                {
                    const Core::DataType dtype = sushiblas_dtype<T>();
                    Tensor A = engine->create_tensor({s, s}, dtype);
                    Tensor sycl_result = engine->create_tensor({s, s}, dtype);
                    Tensor reference_result = engine->create_tensor({s, s}, dtype);
                    fill_random<T>(A);
                    fill_random<T>(sycl_result);

                    // Both results start equal, so the triangle left alone compares too.
                    const std::vector<T> start = read_tensor<T>(sycl_result);
                    queue().memcpy(reference_result.data(), start.data(),
                                   start.size() * sizeof(T)).wait();

                    set_mode(ExecutionMode::ForceSYCL);
                    const std::size_t live_before = TestEnvironment::live_allocation_count();
                    record(A, sycl_result);
                    EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before + 1);
                    recording->run().value();

                    set_mode(ExecutionMode::ForceReference);
                    record(A, reference_result);
                    recording->run().value();

                    const std::vector<T> got = read_tensor<T>(sycl_result);
                    const std::vector<T> want = read_tensor<T>(reference_result);
                    ASSERT_EQ(got.size(), want.size());
                    EXPECT_EQ(count_parts_beyond(got, want, parity_bound<T>(s)), std::size_t{0});
                }
        };

        /**
         * @brief Runs a case once per row, and skips a device that does not compute in the type of
         *        the row.
         * @tparam Row One of the row types above; it holds the element type as `type`.
         */
        template<typename Row>
        class PackedSyrkRowCase : public PackedSyrkEngine, public ::testing::WithParamInterface<Row>
        {
            protected:
                /** @brief Skips the case when the device does not compute in the type of the row. */
                void SetUp() override
                {
                    PackedSyrkEngine::SetUp();
                    if (!device_runs(this->GetParam().type))
                        GTEST_SKIP() << "the device does not compute in this precision";
                }
        };

        /** @brief Compares the packed SYRK with the reference on exact data. */
        class Medium_PackedSyrk : public PackedSyrkRowCase<ExactRow>
        {
        };

        /** @brief Compares the packed HERK with the reference on exact data. */
        class Medium_PackedHerk : public PackedSyrkRowCase<ExactRow>
        {
        };

        /** @brief Compares the seam of syrk and herk with the reference on an update the profile packs. */
        class Medium_PackedSyrkSeam : public PackedSyrkRowCase<SeamRow>
        {
            protected:
                /** @brief Skips the case when no order up to the largest tried selects the packed kernel. */
                void SetUp() override
                {
                    PackedSyrkRowCase<SeamRow>::SetUp();
                    if (IsSkipped())
                        return;

                    const Device::Profile& profile = Device::profile_for(queue());
                    with_element(GetParam().type, [&](auto element)
                    {
                        using T = typename decltype(element)::type;
                        order_ = packed_order<T>(profile);
                    });
                    if (order_ == 0)
                        GTEST_SKIP() << "no order up to " << K_LARGEST_SEAM_ORDER
                                     << " selects the packed kernel; cache_bytes="
                                     << profile.cache_bytes;
                }

                /// @brief The smallest order the profile packs an update of the type of the row at.
                int64_t order_ = 0;
        };

        TEST_P(Medium_PackedSyrk, BothTrianglesEqualTheReferenceOnExactData)
        {
            const ExactRow& row = GetParam();
            with_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                SyrkProblem<T> problem(queue(), row.layout, row.uplo, row.trans, K_EXACT_ORDER,
                                       K_EXACT_TERMS);
                (void)expect_exact<T, Kernels::Unconjugated, T>(problem, row.blocking);
            });
        }

        TEST_P(Medium_PackedHerk, HerkConjugatesTheSecondFactor)
        {
            const ExactRow& row = GetParam();
            with_complex_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Scalar = Kernels::RealType<T>;

                SyrkProblem<T> problem(queue(), row.layout, row.uplo, row.trans, K_EXACT_ORDER,
                                       K_EXACT_TERMS);
                const std::vector<T> got =
                    expect_exact<T, Kernels::Conjugated, Scalar>(problem, row.blocking);

                std::size_t complex_diagonals = 0;
                for (int64_t i = 0; i < K_EXACT_ORDER; ++i)
                    if (parts_of(got[problem.diagonal_index(i)])[1] != 0.0)
                        ++complex_diagonals;
                EXPECT_EQ(complex_diagonals, std::size_t{0});
            });
        }

        TEST_P(Medium_PackedSyrkSeam, TheKernelPacksWhenTheProfileSelectsIt)
        {
            const SeamRow& row = GetParam();
            with_element(row.type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                ASSERT_TRUE(Kernels::selects_packed_product<T>(
                    Device::profile_for(queue()), order_, order_, order_))
                    << "the profile does not choose the packed kernel for the order " << order_;
                expect_seam_parity<T>(order_, [&](const Tensor& A, Tensor& C)
                {
                    if (row.routine == Routine::Syrk)
                        SushiBLAS::BLASOps(recorder()).syrk(A, C, row.uplo);
                    else
                        SushiBLAS::BLASOps(recorder()).herk(A, C, row.uplo);
                });
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_PackedSyrk,
            ::testing::ValuesIn(exact_rows(K_ALL_TYPES, Core::Transpose::Trans)), exact_row_name);

        INSTANTIATE_TEST_SUITE_P(
            , Medium_PackedHerk,
            ::testing::ValuesIn(exact_rows(K_COMPLEX_TYPES, Core::Transpose::ConjTrans)),
            exact_row_name);

        INSTANTIATE_TEST_SUITE_P(, Medium_PackedSyrkSeam,
                                 ::testing::ValuesIn(seam_rows()), seam_row_name);
    } // namespace Test
} // namespace SushiBLAS
