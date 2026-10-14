/**************************************************************************/
/* test_packed_symm.cpp                                                   */
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
 * @file test_packed_symm.cpp
 * @brief Tests SYMM and HEMM on the packed product against the reference kernel.
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
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include "test_utils.hpp"
#include "blas_truth.hpp"
#include "packed_product_harness.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the rows of B and C in the exact-data tests.
            constexpr int64_t K_EXACT_ROWS = 29;

            /// @brief Holds the columns of B and C in the exact-data tests.
            constexpr int64_t K_EXACT_COLS = 21;

            /// @brief Holds the imaginary part written on the stored diagonal of a Hermitian A.
            constexpr double K_DIAGONAL_IMAGINARY = 7.0;

            /// @brief Holds the rounded operations a parity bound allows beyond one per term.
            constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Holds the step between the orders tried for a product the profile packs.
            constexpr int64_t K_SEAM_ORDER_STEP = 64;

            /// @brief Holds the largest order tried for a product the profile packs.
            constexpr int64_t K_LARGEST_SEAM_ORDER = 768;

            /// @brief Holds the two storage orders every exact-data test runs in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief Holds the two sides A can stand on.
            constexpr Core::Side K_SIDES[] = {Core::Side::Left, Core::Side::Right};

            /// @brief Holds the two triangles A can be stored in.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /**
             * @brief Holds the relative error of storing a part of T: 2^-11 for HALF, the unit
             *        roundoff of the accumulation type otherwise.
             */
            template<typename T>
            constexpr double K_STORE_ROUNDOFF =
                std::is_same_v<T, sycl::half>
                    ? 0x1p-11
                    : unit_roundoff<Kernels::RealType<Kernels::Accumulator<T>>>();

            /** @brief Names the routine a seam case records. */
            enum class Routine : std::uint8_t
            {
                Symm, /**< The symmetric product. */
                Hemm  /**< The Hermitian product. */
            };

            /// @brief Holds the two routines the seam test records.
            constexpr Routine K_ROUTINES[] = {Routine::Symm, Routine::Hemm};

            /** @brief Holds one injected blocking and the compute units it is planned for. */
            struct InjectedBlocking
            {
                const char* name;               /**< Names the blocking in a failure message. */
                Device::CacheBlocking blocking; /**< The budgets handed to the plan. */
                std::size_t compute_units;      /**< The compute units handed to the plan. */
            };

            /** @brief Returns the blockings every exact-data test runs under, for type T. */
            template<typename T>
            [[nodiscard]] std::vector<InjectedBlocking> injected_blockings()
            {
                return {
                    {"smallest, one unit", smallest_blocking(), 1},
                    {"widest, one unit", widest_blocking(), 1},
                    {"ragged, three units", ragged_blocking<T>(), 3},
                };
            }

            /** @brief Returns the name of a layout for a failure message. */
            [[nodiscard]] const char* layout_name(Core::Layout layout) noexcept
            {
                return layout == Core::Layout::RowMajor ? "row-major" : "column-major";
            }

            /** @brief Returns the name of a routine for a failure message. */
            [[nodiscard]] const char* routine_name(Routine routine) noexcept
            {
                return routine == Routine::Symm ? "symm" : "hemm";
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

            /** @brief Returns how many elements of values hold a NaN in either part. */
            template<typename T>
            [[nodiscard]] std::size_t count_nans(const std::vector<T>& values) noexcept
            {
                std::size_t nans = 0;
                for (const T& value : values)
                {
                    const std::array<double, 2> parts = parts_of(value);
                    if (std::isnan(parts[0]) || std::isnan(parts[1]))
                        ++nans;
                }
                return nans;
            }

            /** @brief Returns value with its imaginary part replaced; a real value is unchanged. */
            template<typename T>
            [[nodiscard]] T with_imaginary(T value, [[maybe_unused]] double imaginary) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return T(value.real(), static_cast<typename T::value_type>(imaginary));
                else
                    return value;
            }

            /**
             * @brief Returns how far a part of one product element may lie from the reference.
             * @param terms The terms of each sum.
             * @param alpha_abs The absolute value of alpha; operand parts are at most 1.
             * @param added_abs The largest magnitude added to the product.
             */
            template<typename T>
            [[nodiscard]] double parity_bound(int64_t terms, double alpha_abs,
                                              double added_abs) noexcept
            {
                using Real = Kernels::RealType<Kernels::Accumulator<T>>;

                const double parts = Core::is_complex_v<T> ? 2.0 : 1.0;
                const double magnitude =
                    parts * (alpha_abs * static_cast<double>(terms) + added_abs);
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
             * @brief Returns the smallest order the profile packs a square product of T at.
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
             * @brief Owns the operands of one symmetric or Hermitian product and runs both kernels.
             * @tparam T The element type.
             * @tparam Conj Kernels::Unconjugated for SYMM, Kernels::Conjugated for HEMM.
             */
            template<typename T, typename Conj>
            class SymmetricProblem final
            {
                public:
                    /**
                     * @brief Allocates A, B and C with padded leading dimensions, fills them with
                     *        integer-valued elements and poisons the triangle of A never read.
                     */
                    SymmetricProblem(sycl::queue& q, Core::Layout layout, Core::Side side,
                                     Core::Uplo uplo, int64_t m, int64_t n)
                        : queue_(q), left_(side == Core::Side::Left), m_(m), n_(n),
                          order_(left_ ? m : n),
                          lda_(order_ + K_PACKED_TEST_LD_PADDING),
                          ldb_(((layout == Core::Layout::ColumnMajor) ? m : n) +
                               K_PACKED_TEST_LD_PADDING),
                          str_a_(lda_ * order_ + K_PACKED_TEST_BATCH_GAP),
                          str_b_(ldb_ * ((layout == Core::Layout::ColumnMajor) ? n : m) +
                                 K_PACKED_TEST_BATCH_GAP),
                          at_a_(Kernels::symmetric_access<Conj>(layout, uplo, lda_)),
                          at_b_(Kernels::operand_access(layout, Core::Transpose::NoTrans, ldb_)),
                          a_(q, static_cast<std::size_t>(str_a_)),
                          b_(q, static_cast<std::size_t>(str_b_)),
                          c_(q, static_cast<std::size_t>(str_b_)),
                          initial_c_(c_.size())
                    {
                        for (std::size_t i = 0; i < a_.size(); ++i)
                            a_.get()[i] = patterned_element<T>(static_cast<int64_t>(i) / lda_,
                                                               static_cast<int64_t>(i) % lda_, 0);
                        for (std::size_t i = 0; i < b_.size(); ++i)
                            b_.get()[i] = patterned_element<T>(static_cast<int64_t>(i) / ldb_,
                                                               static_cast<int64_t>(i) % ldb_, 1);
                        for (std::size_t i = 0; i < initial_c_.size(); ++i)
                            initial_c_[i] = patterned_element<T>(static_cast<int64_t>(i) / ldb_,
                                                                 static_cast<int64_t>(i) % ldb_,
                                                                 2);

                        for (int64_t r = 0; r < order_; ++r)
                            for (int64_t c = 0; c < order_; ++c)
                                if (at_a_.upper ? (r > c) : (r < c))
                                    a_.get()[at_a_.stored.index(r, c)] = poison_element<T>();
                    }

                    /** @brief Replaces the imaginary part of every stored diagonal element of A. */
                    void set_diagonal_imaginary(double imaginary)
                    {
                        for (int64_t d = 0; d < order_; ++d)
                        {
                            T& diagonal = a_.get()[at_a_.stored.index(d, d)];
                            diagonal = with_imaginary(diagonal, imaginary);
                        }
                    }

                    /** @brief Returns the C buffer the reference kernel leaves, padding too. */
                    [[nodiscard]] std::vector<T> expected(T alpha, T beta) const
                    {
                        std::vector<T> c = initial_c_;
                        const Kernels::OperandAccess at_c = at_b_;
                        const int64_t str_c = str_b_;

                        multiply([&](const T* x, auto at_x, int64_t str_x,
                                     const T* y, auto at_y, int64_t str_y)
                        {
                            Kernels::Reference::gemm_accessed<T>(
                                m_, n_, order_, alpha, x, at_x, str_x, y, at_y, str_y,
                                beta, c.data(), at_c, str_c, 1, Kernels::GemmIdentityEpilogue{});
                        });
                        return c;
                    }

                    /** @brief Returns the C buffer the packed kernel leaves under a blocking. */
                    [[nodiscard]] std::vector<T> packed(const Device::CacheBlocking& blocking,
                                                        std::size_t compute_units, T alpha, T beta)
                    {
                        std::memcpy(c_.get(), initial_c_.data(), initial_c_.size() * sizeof(T));

                        const Kernels::PackedProductPlan plan = Kernels::plan_packed_blocks<T>(
                            blocking, compute_units, m_, n_, order_, 1);

                        // The scratch starts off its alignment, so the kernel has to align it.
                        const USMArray<unsigned char> scratch(
                            queue_, plan.scratch_bytes + K_PACKED_TEST_SCRATCH_SKEW);
                        void* const panels = scratch.get() + K_PACKED_TEST_SCRATCH_SKEW;
                        const Kernels::OperandAccess at_c = at_b_;
                        const int64_t str_c = str_b_;

                        multiply([&](const T* x, auto at_x, int64_t str_x,
                                     const T* y, auto at_y, int64_t str_y)
                        {
                            Kernels::SYCL::Detail::packed_gemm<T>(
                                queue_, plan, m_, n_, order_, alpha, x, at_x, str_x,
                                y, at_y, str_y, beta, c_.get(), at_c, str_c, 1, panels, {},
                                Kernels::GemmIdentityEpilogue{}).wait();
                        });

                        return std::vector<T>(c_.get(), c_.get() + c_.size());
                    }

                private:
                    /**
                     * @brief Calls product with A first under LEFT and with B first under RIGHT.
                     * @param product Callable taking, per factor, a pointer, a brick and a stride.
                     */
                    template<typename Product>
                    void multiply(Product&& product) const
                    {
                        if (left_)
                            product(a_.get(), at_a_, str_a_, b_.get(), at_b_, str_b_);
                        else
                            product(b_.get(), at_b_, str_b_, a_.get(), at_a_, str_a_);
                    }

                    /// @brief The queue the packed kernel's memory lives on.
                    sycl::queue& queue_;

                    /// @brief Whether A is the first factor.
                    bool left_;

                    /// @brief Rows of B and C.
                    int64_t m_;

                    /// @brief Columns of B and C.
                    int64_t n_;

                    /// @brief Order of A, which is also the terms of each sum.
                    int64_t order_;

                    /// @brief Leading dimension of A.
                    int64_t lda_;

                    /// @brief Leading dimension of B and of C.
                    int64_t ldb_;

                    /// @brief Elements of the A buffer.
                    int64_t str_a_;

                    /// @brief Elements of the B buffer and of the C buffer.
                    int64_t str_b_;

                    /// @brief How A is read from its stored triangle.
                    Kernels::SymmetricAccess<Conj> at_a_;

                    /// @brief How B and C are addressed.
                    Kernels::OperandAccess at_b_;

                    /// @brief The matrix A.
                    USMArray<T> a_;

                    /// @brief The matrix B.
                    USMArray<T> b_;

                    /// @brief The matrix C the packed kernel writes.
                    USMArray<T> c_;

                    /// @brief The contents C starts every run from.
                    std::vector<T> initial_c_;
            };

            /**
             * @brief Expects the packed kernel to leave the bits the reference leaves and no NaN,
             *        for each layout, side, triangle and blocking.
             * @tparam Conj Kernels::Unconjugated for SYMM, Kernels::Conjugated for HEMM.
             * @param diagonal_imaginary Written on the diagonal of A when not zero.
             */
            template<typename T, typename Conj>
            void expect_packed_equals_reference(sycl::queue& q, double diagonal_imaginary)
            {
                const T alpha = T(2);
                const T beta = T(-1);

                for (const Core::Layout layout : K_LAYOUTS)
                {
                    for (const Core::Side side : K_SIDES)
                    {
                        for (const Core::Uplo uplo : K_TRIANGLES)
                        {
                            SCOPED_TRACE(std::string(layout_name(layout)) + " " +
                                         Core::to_string(side) + " " + Core::to_string(uplo));

                            SymmetricProblem<T, Conj> problem(q, layout, side, uplo,
                                                              K_EXACT_ROWS, K_EXACT_COLS);
                            if (diagonal_imaginary != 0.0)
                                problem.set_diagonal_imaginary(diagonal_imaginary);

                            const std::vector<T> want = problem.expected(alpha, beta);
                            for (const InjectedBlocking& injected : injected_blockings<T>())
                            {
                                SCOPED_TRACE(injected.name);
                                const std::vector<T> got = problem.packed(
                                    injected.blocking, injected.compute_units, alpha, beta);
                                ASSERT_EQ(got.size(), want.size());
                                EXPECT_EQ(count_bit_mismatches(got.data(), want.data(),
                                                               want.size()),
                                          std::size_t{0});
                                EXPECT_EQ(count_nans(got), std::size_t{0});
                            }
                        }
                    }
                }
            }

            /** @brief Records the product of a routine with its default scalars. */
            TaskHandle record_product(Engine& engine, Routine routine, const Tensor& A,
                                      const Tensor& B, Tensor& C, Core::Side side, Core::Uplo uplo)
            {
                SushiBLAS::BLASOps ops(engine);
                return routine == Routine::Hemm ? ops.hemm(A, B, C, side, uplo)
                                                : ops.symm(A, B, C, side, uplo);
            }
        } // namespace

        /** @brief Gives the packed SYMM tests an engine and the queue, per element type. */
        template<typename T>
        class Unit_PackedSymm : public SushiBLASTest
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
                    return TestEnvironment::queue();
                }

                /** @brief Returns a square tensor of T of order s with elements in [-1, 1]. */
                [[nodiscard]] Tensor random_square(int64_t s)
                {
                    Tensor tensor = engine->create_tensor({s, s}, sushiblas_dtype<T>());
                    fill_random<T>(tensor);
                    return tensor;
                }
        };

        using PackedSymmTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                 std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_PackedSymm, PackedSymmTypes);

        /** @brief Tests a symmetric brick as either factor against the reference, bit for bit. */
        TYPED_TEST(Unit_PackedSymm, BothSidesEqualTheReferenceOnExactData)
        {
            expect_packed_equals_reference<TypeParam, Kernels::Unconjugated>(this->queue(), 0.0);
        }

        /** @brief Tests a Hermitian brick whose stored diagonal carries an imaginary part. */
        TYPED_TEST(Unit_PackedSymm, HemmReadsTheMirrorConjugatedAndTheDiagonalReal)
        {
            if constexpr (Core::is_complex_v<TypeParam>)
            {
                expect_packed_equals_reference<TypeParam, Kernels::Conjugated>(
                    this->queue(), K_DIAGONAL_IMAGINARY);
            }
            else
            {
                GTEST_SKIP() << "a real element type has no conjugate and no imaginary diagonal";
            }
        }

        /** @brief Tests that symm and hemm lease scratch on SYCL alone and keep parity. */
        TYPED_TEST(Unit_PackedSymm, TheSeamLeasesScratchForASymmetricProduct)
        {
            using T = TypeParam;
            const Device::Profile& profile = Device::profile_for(this->queue());
            const int64_t s = packed_order<T>(profile);
            if (s == 0)
                GTEST_SKIP() << "no order up to " << K_LARGEST_SEAM_ORDER
                             << " selects the packed kernel; cache_bytes=" << profile.cache_bytes;

            const Core::DataType dtype = sushiblas_dtype<T>();

            for (const Routine routine : K_ROUTINES)
            {
                for (const Core::Side side : K_SIDES)
                {
                    SCOPED_TRACE(std::string(routine_name(routine)) + " " +
                                 Core::to_string(side));

                    const Tensor A = this->random_square(s);
                    const Tensor B = this->random_square(s);
                    Tensor sycl_result = this->engine->create_tensor({s, s}, dtype);
                    Tensor reference_result = this->engine->create_tensor({s, s}, dtype);

                    this->engine->set_mode(ExecutionMode::ForceSYCL);
                    const std::size_t live_before_sycl = TestEnvironment::live_allocation_count();
                    record_product(*this->engine, routine, A, B, sycl_result, side,
                                   Core::Uplo::Upper);
                    EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_sycl + 1);
                    this->engine->execute().value();

                    this->engine->set_mode(ExecutionMode::ForceReference);
                    const std::size_t live_before_reference =
                        TestEnvironment::live_allocation_count();
                    record_product(*this->engine, routine, A, B, reference_result, side,
                                   Core::Uplo::Upper);
                    EXPECT_EQ(TestEnvironment::live_allocation_count(), live_before_reference);
                    this->engine->execute().value();

                    const std::vector<T> got = read_tensor<T>(sycl_result);
                    const std::vector<T> want = read_tensor<T>(reference_result);
                    ASSERT_EQ(got.size(), want.size());
                    EXPECT_EQ(count_parts_beyond(got, want, parity_bound<T>(s, 1.0, 0.0)),
                              std::size_t{0});
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
