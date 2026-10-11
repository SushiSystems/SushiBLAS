/**************************************************************************/
/* test_syr2k.cpp                                                         */
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
 * @file test_syr2k.cpp
 * @brief Tests syr2k and her2k against hand-computed updates, a host integer loop, a truth and
 *        refusals.
 * @author Mustafa Garip
 */

#include <cmath>
#include <future>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syr2k.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/blas/local_tile.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include "test_utils.hpp"
#include "blas_truth.hpp"
#include "graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The type the host loop computes in; exact on the integers it is given.
            using Wide = std::complex<double>;

            /// @brief Order of C in the multi-tile cases: three tiles with a partial last one.
            constexpr int64_t K_ORDER = 70;

            /// @brief Columns of op(A) in the multi-tile cases: three k-steps, the last partial.
            constexpr int64_t K_INNER = 37;

            /// @brief Matrices in the batched case.
            constexpr int64_t K_BATCH = 3;

            /// @brief Order of one C of the batched case.
            constexpr int64_t K_BATCH_ORDER = 5;

            /// @brief Columns of one op(A) of the batched case.
            constexpr int64_t K_BATCH_INNER = 3;

            /// @brief Index of the integer pattern at which the elements of A start.
            constexpr std::size_t K_SEED_A = 0;

            /// @brief Index of the integer pattern at which the elements of B start.
            constexpr std::size_t K_SEED_B = 11;

            /// @brief Index of the integer pattern at which the elements of C start.
            constexpr std::size_t K_SEED_C = 23;

            /// @brief Seed of the random operands of this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Scales both products of the integer cases of syr2k.
            constexpr double K_INTEGER_ALPHA = 2.0;

            /// @brief Scales the prior C of the integer cases.
            constexpr double K_INTEGER_BETA = -1.0;

            /// @brief Scales both products of the rounded cases.
            constexpr double K_ROUNDED_ALPHA = 1.5;

            /// @brief Scales the prior C of the rounded cases; a power of two, so exact.
            constexpr double K_ROUNDED_BETA = -0.5;

            /// @brief Roundings a kernel makes whatever the length, and those of the truth itself.
            constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Roundings counted for one compensated sum, which does not grow with k.
            constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

            /// @brief Marks an element of C outside the updated triangle in a hand-computed case.
            constexpr double K_UNTOUCHED = -77.0;

            /// @brief The two modes every case runs in.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceSycl,
                                                 ExecutionMode::ForceReference};

            /// @brief The two layouts the exact cases run in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief The two triangles of C an update can write.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BlasScalar K_COMPLEX_SCALAR = std::complex<double>(1.0, 2.0);

            /** @brief Returns the name of a mode for a failure message. */
            [[nodiscard]] const char* mode_name(ExecutionMode mode) noexcept
            {
                return mode == ExecutionMode::ForceSycl ? "ForceSycl" : "ForceReference";
            }

            /** @brief Returns the names of a layout, a triangle and a transpose for a message. */
            [[nodiscard]] std::string case_name(Core::Layout layout, Core::Uplo uplo,
                                                Core::Transpose trans)
            {
                const char* const layout_name =
                    layout == Core::Layout::RowMajor ? "ROW_MAJOR" : "COLUMN_MAJOR";
                return std::string(layout_name) + " " + Core::to_string(uplo) + " " +
                       Core::to_string(trans);
            }

            /** @brief Returns value in the type the host loop computes in. */
            template<typename T>
            [[nodiscard]] Wide widen(T value) noexcept
            {
                if constexpr (Core::is_complex_v<T>)
                    return Wide(static_cast<double>(value.real()),
                                static_cast<double>(value.imag()));
                else
                    return Wide(static_cast<double>(value), 0.0);
            }

            /** @brief Returns value as an element of T; a real T takes the real part. */
            template<typename T>
            [[nodiscard]] T narrow(Wide value) noexcept
            {
                using Real = Kernels::RealType<T>;
                if constexpr (Core::is_complex_v<T>)
                    return T(static_cast<Real>(value.real()), static_cast<Real>(value.imag()));
                else
                    return static_cast<T>(value.real());
            }

            /** @brief Returns a scalar of the public interface, real when value is. */
            [[nodiscard]] BlasScalar blas_scalar(Wide value) noexcept
            {
                return value.imag() == 0.0 ? BlasScalar(value.real()) : BlasScalar(value);
            }

            /** @brief Returns whether (r, c) lies in the updated triangle. */
            [[nodiscard]] constexpr bool is_updated(Core::Uplo uplo, int64_t r, int64_t c) noexcept
            {
                return uplo == Core::Uplo::Upper ? r <= c : r >= c;
            }

            /** @brief Holds a matrix on the host in the element order of a layout. */
            template<typename T>
            struct HostMatrix
            {
                int64_t rows = 0;                              /**< Rows of the matrix. */
                int64_t cols = 0;                              /**< Columns of the matrix. */
                Core::Layout layout = Core::Layout::RowMajor; /**< Orders values. */
                std::vector<T> values;                         /**< rows * cols elements. */

                /** @brief Returns the position of element (r, c) in values. */
                [[nodiscard]] std::size_t index(int64_t r, int64_t c) const noexcept
                {
                    return static_cast<std::size_t>(layout == Core::Layout::RowMajor
                                                        ? r * cols + c
                                                        : c * rows + r);
                }

                /** @brief Returns element (r, c). */
                [[nodiscard]] const T& at(int64_t r, int64_t c) const noexcept
                {
                    return values[index(r, c)];
                }

                /** @brief Returns element (r, c) for writing. */
                [[nodiscard]] T& at(int64_t r, int64_t c) noexcept
                {
                    return values[index(r, c)];
                }
            };

            /** @brief Returns a rows x cols matrix in layout with every element zero. */
            template<typename T>
            [[nodiscard]] HostMatrix<T> zero_matrix(int64_t rows, int64_t cols, Core::Layout layout)
            {
                return HostMatrix<T>{rows, cols, layout,
                                     std::vector<T>(static_cast<std::size_t>(rows * cols))};
            }

            /**
             * @brief Returns a rows x cols matrix in layout from its elements in reading order.
             * @param row_by_row rows * cols values, the first row first.
             */
            template<typename T>
            [[nodiscard]] HostMatrix<T> matrix_of(int64_t rows, int64_t cols, Core::Layout layout,
                                                  std::initializer_list<Wide> row_by_row)
            {
                HostMatrix<T> out = zero_matrix<T>(rows, cols, layout);
                EXPECT_EQ(row_by_row.size(), out.values.size());
                int64_t position = 0;
                for (const Wide value : row_by_row)
                {
                    out.at(position / cols, position % cols) = narrow<T>(value);
                    ++position;
                }
                return out;
            }

            /** @brief Returns a matrix of integers from -3 to 3, Gaussian for a complex T. */
            template<typename T>
            [[nodiscard]] HostMatrix<T> integer_matrix(std::size_t seed, int64_t rows, int64_t cols,
                                                       Core::Layout layout)
            {
                HostMatrix<T> out = zero_matrix<T>(rows, cols, layout);
                for (std::size_t i = 0; i < out.values.size(); ++i)
                {
                    const double real =
                        static_cast<double>(static_cast<int64_t>(((i + seed) * 37u) % 7u) - 3);
                    const double imaginary = static_cast<double>(
                        static_cast<int64_t>(((i + seed) * 53u + 2u) % 7u) - 3);
                    out.values[i] = narrow<T>(Wide(real, imaginary));
                }
                return out;
            }

            /** @brief Returns a row-major matrix of a real T drawn uniformly from [-1, 1]. */
            template<typename T>
            [[nodiscard]] HostMatrix<T> random_matrix(std::mt19937& generator, int64_t rows,
                                                      int64_t cols)
            {
                std::uniform_real_distribution<T> distribution(T(-1), T(1));
                HostMatrix<T> out = zero_matrix<T>(rows, cols, Core::Layout::RowMajor);
                for (T& element : out.values)
                    element = distribution(generator);
                return out;
            }

            /** @brief Returns op(x) as a matrix of its own, in the layout of x. */
            template<typename T>
            [[nodiscard]] HostMatrix<T> applied(const HostMatrix<T>& x, Core::Transpose trans)
            {
                if (!Core::is_transposed(trans))
                    return x;

                HostMatrix<T> out = zero_matrix<T>(x.cols, x.rows, x.layout);
                for (int64_t r = 0; r < out.rows; ++r)
                {
                    for (int64_t c = 0; c < out.cols; ++c)
                        out.at(r, c) = Core::is_conjugated(trans) ? Kernels::conjugate(x.at(c, r))
                                                                  : x.at(c, r);
                }
                return out;
            }

            /**
             * @brief Returns C after the rank-2k update, computed by a host loop in Wide.
             * @tparam Conj Kernels::Unconjugated for SYR2K, Kernels::Conjugated for HER2K.
             * @pre Every operand holds integers small enough for each sum to be exact in T.
             */
            template<typename T, typename Conj>
            [[nodiscard]] HostMatrix<T> expected_update(const HostMatrix<T>& a,
                                                        const HostMatrix<T>& b,
                                                        const HostMatrix<T>& c, Core::Uplo uplo,
                                                        Core::Transpose trans, Wide alpha,
                                                        double beta)
            {
                constexpr bool hermitian = std::is_same_v<Conj, Kernels::Conjugated>;
                const HostMatrix<T> op_a = applied(a, trans);
                const HostMatrix<T> op_b = applied(b, trans);
                const auto second_factor = [](Wide value)
                {
                    return hermitian ? std::conj(value) : value;
                };

                HostMatrix<T> out = c;
                for (int64_t i = 0; i < c.rows; ++i)
                {
                    for (int64_t j = 0; j < c.cols; ++j)
                    {
                        if (!is_updated(uplo, i, j))
                            continue;

                        Wide first(0.0, 0.0);
                        Wide second(0.0, 0.0);
                        for (int64_t l = 0; l < op_a.cols; ++l)
                        {
                            first += widen(op_a.at(i, l)) * second_factor(widen(op_b.at(j, l)));
                            second += widen(op_b.at(i, l)) * second_factor(widen(op_a.at(j, l)));
                        }

                        Wide value = alpha * first + second_factor(alpha) * second +
                                     beta * widen(c.at(i, j));
                        if (hermitian && i == j)
                            value = Wide(value.real(), 0.0);
                        out.at(i, j) = narrow<T>(value);
                    }
                }
                return out;
            }

            /** @brief Returns the values of every matrix of a batch, one matrix after another. */
            template<typename T>
            [[nodiscard]] std::vector<T> concatenated(const std::vector<HostMatrix<T>>& batch)
            {
                std::vector<T> out;
                for (const HostMatrix<T>& matrix : batch)
                    out.insert(out.end(), matrix.values.begin(), matrix.values.end());
                return out;
            }

            /** @brief Expects two results to hold the same bytes. */
            template<typename T>
            void expect_same_bytes(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(),
                                         actual.size() * sizeof(T)));
            }

            /** @brief Expects every element to equal its expectation with no tolerance. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    EXPECT_EQ(actual[i], expected[i]) << "index " << i;
            }

            /**
             * @brief Records syr2k under Kernels::Unconjugated and her2k under Kernels::Conjugated.
             * @param beta Real, as her2k takes it.
             */
            template<typename Conj>
            TaskHandle record_update(SushiBLAS::BLASOps& ops, const Tensor& A, const Tensor& B,
                                     Tensor& C, Core::Uplo uplo, Core::Transpose trans,
                                     BlasScalar alpha, double beta)
            {
                if constexpr (std::is_same_v<Conj, Kernels::Conjugated>)
                    return ops.her2k(A, B, C, uplo, trans, alpha, beta);
                else
                    return ops.syr2k(A, B, C, uplo, trans, alpha, beta);
            }

            /** @brief Holds a queued event that stays incomplete until the gate is opened. */
            class DependencyGate final
            {
                public:
                    /** @brief Submits a host task to queue that blocks until open() is called. */
                    explicit DependencyGate(sycl::queue& queue)
                        : event_(queue.submit(
                              [opened = opened_.get_future().share()](sycl::handler& h)
                              {
                                  h.host_task([opened]() { opened.wait(); });
                              }))
                    {

                    }

                    DependencyGate(const DependencyGate&) = delete;
                    DependencyGate& operator=(const DependencyGate&) = delete;
                    DependencyGate(DependencyGate&&) = delete;
                    DependencyGate& operator=(DependencyGate&&) = delete;

                    /** @brief Opens the gate, so that a test leaving early blocks no host task. */
                    ~DependencyGate()
                    {
                        open();
                    }

                    /** @brief Returns the event a kernel is handed as its dependency. */
                    [[nodiscard]] const sycl::event& event() const noexcept { return event_; }

                    /** @brief Lets the host task finish; a second call does nothing. */
                    void open()
                    {
                        if (is_open_)
                            return;

                        is_open_ = true;
                        opened_.set_value();
                    }

                private:
                    /// @brief Fulfilled by open(); declared first, because event_ is built from it.
                    std::promise<void> opened_;

                    /// @brief The event of the blocked host task.
                    sycl::event event_;

                    /// @brief Whether open() has fulfilled the promise.
                    bool is_open_ = false;
            };

            /** @brief Returns whether the device of profile hosts the SYR2K tile of T. */
            template<typename T>
            [[nodiscard]] bool hosts_syr2k_tile(const Device::Profile& profile) noexcept
            {
                return Device::hosts_local_tile(profile, Kernels::SYCL::Syr2kTile::demand<T>());
            }

            /** @brief Returns whether the tile of T has the given geometry. */
            template<typename T>
            [[nodiscard]] constexpr bool tile_is(int64_t register_block, int64_t group_edge,
                                                 std::size_t local_bytes) noexcept
            {
                using Tile = Kernels::SYCL::Detail::Syr2kTileConfig<T>;
                return Tile::K_ITEM_ROWS == register_block && Tile::K_ITEM_COLS == register_block &&
                       Tile::K_BLOCK_M == group_edge && Tile::K_BLOCK_N == group_edge &&
                       Tile::local_mem_bytes() == local_bytes;
            }
        } // namespace

        /** @brief Uploads host matrices and runs rank-2k updates on them in a chosen mode. */
        class RankTwoUpdateTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a tensor of dims that holds values. */
                template<typename T>
                [[nodiscard]] Tensor upload(std::initializer_list<int64_t> dims,
                                            const std::vector<T>& values)
                {
                    Tensor tensor = engine->create_tensor(dims, get_sushiblas_dtype<T>());
                    EXPECT_EQ(values.size(), static_cast<std::size_t>(tensor.num_elements));
                    if (!values.empty())
                        TestEnvironment::get_context().devices().queue(0)
                            .memcpy(tensor.data(), values.data(), values.size() * sizeof(T)).wait();
                    return tensor;
                }

                /**
                 * @brief Returns a tensor that holds matrix.
                 * @pre The engine was built with the layout of matrix.
                 */
                template<typename T>
                [[nodiscard]] Tensor upload(const HostMatrix<T>& matrix)
                {
                    Tensor tensor = upload({matrix.rows, matrix.cols}, matrix.values);
                    EXPECT_EQ(tensor.layout, matrix.layout);
                    return tensor;
                }

                /**
                 * @brief Returns the elements of C after the update of a routine, run in mode.
                 * @tparam Conj Kernels::Unconjugated for syr2k, Kernels::Conjugated for her2k.
                 */
                template<typename T, typename Conj>
                [[nodiscard]] std::vector<T> run_update(ExecutionMode mode, const HostMatrix<T>& a,
                                                        const HostMatrix<T>& b,
                                                        const HostMatrix<T>& c, Core::Uplo uplo,
                                                        Core::Transpose trans, Wide alpha,
                                                        double beta)
                {
                    engine->set_mode(mode);
                    const Tensor A = upload(a);
                    const Tensor B = upload(b);
                    Tensor C = upload(c);
                    SushiBLAS::BLASOps ops(*engine);
                    record_update<Conj>(ops, A, B, C, uplo, trans, blas_scalar(alpha), beta);
                    engine->execute();
                    return read_tensor<T>(C);
                }

                /**
                 * @brief Expects both modes to equal the host loop exactly on integer operands,
                 *        for both layouts, both triangles and each of transposes.
                 */
                template<typename T, typename Conj>
                void expect_exact_on_integers(std::initializer_list<Core::Transpose> transposes,
                                              Wide alpha)
                {
                    for (const Core::Layout layout : K_LAYOUTS)
                    {
                        reinit_engine(layout);
                        for (const Core::Transpose trans : transposes)
                        {
                            const bool transposed = Core::is_transposed(trans);
                            const int64_t rows = transposed ? K_INNER : K_ORDER;
                            const int64_t cols = transposed ? K_ORDER : K_INNER;
                            const auto a = integer_matrix<T>(K_SEED_A, rows, cols, layout);
                            const auto b = integer_matrix<T>(K_SEED_B, rows, cols, layout);
                            const auto c = integer_matrix<T>(K_SEED_C, K_ORDER, K_ORDER, layout);

                            for (const Core::Uplo uplo : K_TRIANGLES)
                            {
                                const HostMatrix<T> expected = expected_update<T, Conj>(
                                    a, b, c, uplo, trans, alpha, K_INTEGER_BETA);
                                for (const ExecutionMode mode : K_MODES)
                                {
                                    SCOPED_TRACE(case_name(layout, uplo, trans) + " " +
                                                 mode_name(mode));
                                    expect_equal(run_update<T, Conj>(mode, a, b, c, uplo, trans,
                                                                     alpha, K_INTEGER_BETA),
                                                 expected.values);
                                }
                            }
                        }
                    }
                }
        };

        /** @brief Runs syr2k over the real and complex element types. */
        template<typename T>
        class Unit_Syr2k : public RankTwoUpdateTest
        {

        };

        using Syr2kTypes = ::testing::Types<float, double, std::complex<float>,
                                            std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_Syr2k, Syr2kTypes);

        TYPED_TEST(Unit_Syr2k, MatchesAHandComputedUpdate)
        {
            using T = TypeParam;
            using Conj = Kernels::Unconjugated;

            for (const Core::Layout layout : K_LAYOUTS)
            {
                this->reinit_engine(layout);
                const auto a = matrix_of<T>(2, 3, layout, {1, 2, 3, 4, 5, 6});
                const auto b = matrix_of<T>(2, 3, layout, {1, 0, -1, 2, 1, 0});
                const auto c = matrix_of<T>(2, 2, layout, {1, 10, 20, 1});

                // A * B^T + B * A^T is {-4 2, 2 26}; twice that minus C is {-9 -6, -16 51}.
                const auto upper = matrix_of<T>(2, 2, layout, {-9, -6, 20, 51});
                const auto lower = matrix_of<T>(2, 2, layout, {-9, 10, -16, 51});

                for (const Core::Transpose trans :
                     {Core::Transpose::NoTrans, Core::Transpose::Trans})
                {
                    // Under TRANS the stored operands are the 3 x 2 transposes.
                    const auto stored_a = applied(a, trans);
                    const auto stored_b = applied(b, trans);

                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(case_name(layout, Core::Uplo::Upper, trans) + " " +
                                     mode_name(mode));
                        expect_equal(this->template run_update<T, Conj>(
                                         mode, stored_a, stored_b, c, Core::Uplo::Upper, trans,
                                         K_INTEGER_ALPHA, K_INTEGER_BETA),
                                     upper.values);
                        expect_equal(this->template run_update<T, Conj>(
                                         mode, stored_a, stored_b, c, Core::Uplo::Lower, trans,
                                         K_INTEGER_ALPHA, K_INTEGER_BETA),
                                     lower.values);
                    }
                }
            }
        }

        TYPED_TEST(Unit_Syr2k, IsExactOnSmallIntegers)
        {
            this->template expect_exact_on_integers<TypeParam, Kernels::Unconjugated>(
                {Core::Transpose::NoTrans, Core::Transpose::Trans}, K_INTEGER_ALPHA);
        }

        TYPED_TEST(Unit_Syr2k, ABatchIsUpdatedMatrixByMatrix)
        {
            using T = TypeParam;
            using Conj = Kernels::Unconjugated;
            const Core::Layout layout = Core::Layout::RowMajor;
            const int64_t n = K_BATCH_ORDER;
            const int64_t k = K_BATCH_INNER;

            std::vector<HostMatrix<T>> a;
            std::vector<HostMatrix<T>> b;
            std::vector<HostMatrix<T>> c;
            for (int64_t index = 0; index < K_BATCH; ++index)
            {
                const std::size_t shift = static_cast<std::size_t>(index) * 5u;
                a.push_back(integer_matrix<T>(K_SEED_A + shift, n, k, layout));
                b.push_back(integer_matrix<T>(K_SEED_B + shift, n, k, layout));
                c.push_back(integer_matrix<T>(K_SEED_C + shift, n, n, layout));
            }

            for (const bool a_is_batched : {true, false})
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    std::vector<HostMatrix<T>> expected;
                    for (std::size_t index = 0; index < c.size(); ++index)
                        expected.push_back(expected_update<T, Conj>(
                            a[a_is_batched ? index : 0], b[index], c[index], uplo,
                            Core::Transpose::NoTrans, K_INTEGER_ALPHA, K_INTEGER_BETA));

                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(case_name(layout, uplo, Core::Transpose::NoTrans) + " " +
                                     mode_name(mode) +
                                     (a_is_batched ? ", A batched" : ", A of rank 2"));
                        this->engine->set_mode(mode);
                        const Tensor A = a_is_batched
                            ? this->upload({K_BATCH, n, k}, concatenated(a))
                            : this->upload(a[0]);
                        const Tensor B = this->upload({K_BATCH, n, k}, concatenated(b));
                        Tensor C = this->upload({K_BATCH, n, n}, concatenated(c));

                        SushiBLAS::BLASOps(*this->engine).syr2k(A, B, C, uplo,
                                                                Core::Transpose::NoTrans,
                                                                K_INTEGER_ALPHA, K_INTEGER_BETA);
                        this->engine->execute();

                        expect_equal(read_tensor<T>(C), concatenated(expected));
                    }
                }
            }
        }

        /** @brief Measures syr2k against a truth, over the real element types that have one. */
        template<typename T>
        class Unit_Syr2kBound : public RankTwoUpdateTest
        {
            protected:
                /** @brief Returns the unit roundoff of the type the kernel of mode sums T in. */
                [[nodiscard]] static double summing_unit(ExecutionMode mode) noexcept
                {
                    return mode == ExecutionMode::ForceReference
                               ? unit_roundoff<Kernels::ReferenceAccumulator<T>>()
                               : unit_roundoff<Kernels::Accumulator<T>>();
                }

                /** @brief Returns the roundings the kernel of mode makes on two sums of k terms. */
                [[nodiscard]] static std::size_t roundings(ExecutionMode mode, int64_t k) noexcept
                {
                    const bool compensated = mode == ExecutionMode::ForceReference &&
                                             Kernels::ReferenceSum<T>::K_IS_COMPENSATED;
                    return 2 * (compensated ? K_COMPENSATED_ROUNDINGS
                                            : static_cast<std::size_t>(k)) +
                           K_FIXED_ROUNDINGS;
                }

                /** @brief Returns the exact element (i, j) of the update of op(A), op(B) and C. */
                [[nodiscard]] static TruthValue truth_element(const HostMatrix<T>& op_a,
                                                              const HostMatrix<T>& op_b,
                                                              const HostMatrix<T>& c, int64_t i,
                                                              int64_t j)
                {
                    TruthSum sum;
                    for (int64_t l = 0; l < op_a.cols; ++l)
                    {
                        sum.add_product(K_ROUNDED_ALPHA * static_cast<double>(op_a.at(i, l)),
                                        static_cast<double>(op_b.at(j, l)));
                        sum.add_product(K_ROUNDED_ALPHA * static_cast<double>(op_b.at(i, l)),
                                        static_cast<double>(op_a.at(j, l)));
                    }
                    sum.add_product(K_ROUNDED_BETA, static_cast<double>(c.at(i, j)));
                    return sum.value();
                }
        };

        using Syr2kBoundTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_Syr2kBound, Syr2kBoundTypes);

        TYPED_TEST(Unit_Syr2kBound, IsWithinItsBoundOfTheTruth)
        {
            using T = TypeParam;
            std::mt19937 generator(K_SEED);

            for (const Core::Transpose trans : {Core::Transpose::NoTrans, Core::Transpose::Trans})
            {
                const bool transposed = Core::is_transposed(trans);
                const auto a = random_matrix<T>(generator, transposed ? K_INNER : K_ORDER,
                                                transposed ? K_ORDER : K_INNER);
                const auto b = random_matrix<T>(generator, a.rows, a.cols);
                const auto c = random_matrix<T>(generator, K_ORDER, K_ORDER);
                const auto op_a = applied(a, trans);
                const auto op_b = applied(b, trans);

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(case_name(Core::Layout::RowMajor, uplo, trans) + " " +
                                     mode_name(mode));
                        const std::vector<T> actual =
                            this->template run_update<T, Kernels::Unconjugated>(
                                mode, a, b, c, uplo, trans, K_ROUNDED_ALPHA, K_ROUNDED_BETA);
                        ASSERT_EQ(actual.size(), c.values.size());

                        const double unit = this->summing_unit(mode);
                        const std::size_t count = this->roundings(mode, K_INNER);
                        for (int64_t i = 0; i < K_ORDER; ++i)
                        {
                            for (int64_t j = 0; j < K_ORDER; ++j)
                            {
                                if (!is_updated(uplo, i, j))
                                    continue;

                                const TruthValue truth =
                                    this->truth_element(op_a, op_b, c, i, j);
                                const double bound =
                                    rounding_bound(count, unit, truth.magnitude) +
                                    unit_roundoff<T>() * std::fabs(truth.value);
                                EXPECT_NEAR(static_cast<double>(actual[c.index(i, j)]),
                                            truth.value, bound)
                                    << "(" << i << ", " << j << ")";
                            }
                        }
                    }
                }
            }
        }

        /** @brief Runs the cases of syr2k that fix their element type. */
        class Unit_Syr2kFixedType : public RankTwoUpdateTest
        {

        };

        TEST_F(Unit_Syr2kFixedType, HalfSumsInFloat)
        {
            const sycl::queue& queue = TestEnvironment::get_context().devices().queue(0);
            if (!queue.get_device().has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            const std::vector<sycl::half> a = {sycl::half(2048.0f), sycl::half(1.0f),
                                               sycl::half(-2048.0f)};
            const std::vector<sycl::half> ones(3, sycl::half(1.0f));

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                engine->set_mode(mode);
                const Tensor A = upload({1, 3}, a);
                const Tensor B = upload({1, 3}, ones);
                Tensor C = upload({1, 1}, std::vector<sycl::half>(1, sycl::half(0.0f)));

                SushiBLAS::BLASOps(*engine).syr2k(A, B, C, Core::Uplo::Upper,
                                                  Core::Transpose::NoTrans, 1.0, 0.0);
                engine->execute();

                // Each product sums 2048 + 1 - 2048, which is 1 in FLOAT32 and 0 in HALF.
                const std::vector<sycl::half> result = read_tensor<sycl::half>(C);
                ASSERT_EQ(1u, result.size());
                EXPECT_EQ(static_cast<float>(result[0]), 2.0f);
            }
        }

        TEST_F(Unit_Syr2kFixedType, WaitsForItsDependencies)
        {
            const Core::Layout layout = Core::Layout::RowMajor;
            const Tensor A = upload(matrix_of<float>(2, 3, layout, {1, 2, 3, 4, 5, 6}));
            const Tensor B = upload(matrix_of<float>(2, 3, layout, {1, 0, -1, 2, 1, 0}));
            Tensor C = upload(matrix_of<float>(2, 2, layout, {1, 10, 20, 1}));

            sycl::queue& queue = TestEnvironment::get_context().devices().queue(0);
            DependencyGate gate(queue);

            sycl::event done = Kernels::SYCL::syr2k<float>(
                queue, layout, Core::Uplo::Upper, Core::Transpose::NoTrans, 2, 3,
                static_cast<float>(K_INTEGER_ALPHA), static_cast<const float*>(A.data()), 3, 0,
                static_cast<const float*>(B.data()), 3, 0,
                static_cast<float>(K_INTEGER_BETA), static_cast<float*>(C.data()), 2, 0,
                1, std::vector<sycl::event>{gate.event()});

            const auto status = done.get_info<sycl::info::event::command_execution_status>();
            EXPECT_TRUE(status != sycl::info::event_command_status::complete)
                << "The kernel's event completed while its dependency was still running, so "
                   "its command group does not depend on the events it was given.";

            gate.open();
            done.wait_and_throw();
            expect_equal(read_tensor<float>(C), std::vector<float>{-9.0f, -6.0f, 20.0f, 51.0f});
        }

        TEST_F(Unit_Syr2kFixedType, TheTileFitsTheTestDevice)
        {
            static_assert(tile_is<sycl::half>(4, 32, 4096), "The HALF tile changed.");
            static_assert(tile_is<float>(4, 32, 8192), "The FLOAT32 tile changed.");
            static_assert(tile_is<double>(4, 32, 16384), "The FLOAT64 tile changed.");
            static_assert(tile_is<std::complex<float>>(4, 32, 16384),
                          "The COMPLEX32 tile changed.");
            static_assert(tile_is<std::complex<double>>(2, 16, 16384),
                          "The COMPLEX64 tile changed.");

            const Device::Profile& profile =
                Device::profile_for(TestEnvironment::get_context().devices().queue(0));
            EXPECT_TRUE(hosts_syr2k_tile<sycl::half>(profile));
            EXPECT_TRUE(hosts_syr2k_tile<float>(profile));
            EXPECT_TRUE(hosts_syr2k_tile<double>(profile));
            EXPECT_TRUE(hosts_syr2k_tile<std::complex<float>>(profile));
            EXPECT_TRUE(hosts_syr2k_tile<std::complex<double>>(profile));
        }

        /** @brief Runs her2k over the complex element types. */
        template<typename T>
        class Unit_Her2k : public RankTwoUpdateTest
        {
            protected:
                /// @brief The complex alpha of every case of this suite.
                static constexpr Wide K_ALPHA{1.0, 2.0};
        };

        using Her2kTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_Her2k, Her2kTypes);

        TYPED_TEST(Unit_Her2k, MatchesAHandComputedUpdate)
        {
            using T = TypeParam;
            const Wide alpha = this->K_ALPHA;
            const Wide untouched(K_UNTOUCHED, K_UNTOUCHED);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                this->reinit_engine(layout);
                const auto a = matrix_of<T>(2, 1, layout, {Wide(1, 1), Wide(2, 0)});
                const auto b = matrix_of<T>(2, 1, layout, {Wide(0, 1), Wide(1, -1)});
                const auto c = matrix_of<T>(2, 2, layout,
                                            {Wide(1, 5), Wide(3, -1), untouched, Wide(4, 0)});
                const auto expected = matrix_of<T>(
                    2, 2, layout, {Wide(8, 0), Wide(6, 2), untouched, Wide(4, 0)});

                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(case_name(layout, Core::Uplo::Upper, Core::Transpose::NoTrans) +
                                 " " + mode_name(mode));
                    const std::vector<T> actual =
                        this->template run_update<T, Kernels::Conjugated>(
                            mode, a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans, alpha,
                            2.0);

                    // C(0, 0) is 6 + 2 * (1 + 5i) with its imaginary part dropped.
                    expect_equal(actual, expected.values);
                    EXPECT_EQ(actual[expected.index(0, 0)].imag(), 0);
                    EXPECT_EQ(actual[expected.index(1, 1)].imag(), 0);
                }
            }
        }

        TYPED_TEST(Unit_Her2k, IsExactOnGaussianIntegers)
        {
            this->template expect_exact_on_integers<TypeParam, Kernels::Conjugated>(
                {Core::Transpose::NoTrans, Core::Transpose::ConjTrans}, this->K_ALPHA);
        }

        TYPED_TEST(Unit_Her2k, ConjTransEqualsNoTransOnTheConjugateTransposes)
        {
            using T = TypeParam;
            using Conj = Kernels::Conjugated;
            const Wide alpha = this->K_ALPHA;
            const Core::Layout layout = Core::Layout::RowMajor;
            const auto a = integer_matrix<T>(K_SEED_A, K_INNER, K_ORDER, layout);
            const auto b = integer_matrix<T>(K_SEED_B, K_INNER, K_ORDER, layout);
            const auto c = integer_matrix<T>(K_SEED_C, K_ORDER, K_ORDER, layout);
            const auto a_h = applied(a, Core::Transpose::ConjTrans);
            const auto b_h = applied(b, Core::Transpose::ConjTrans);

            for (const Core::Uplo uplo : K_TRIANGLES)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(std::string(Core::to_string(uplo)) + " " + mode_name(mode));
                    expect_equal(this->template run_update<T, Conj>(
                                     mode, a, b, c, uplo, Core::Transpose::ConjTrans, alpha,
                                     K_INTEGER_BETA),
                                 this->template run_update<T, Conj>(
                                     mode, a_h, b_h, c, uplo, Core::Transpose::NoTrans, alpha,
                                     K_INTEGER_BETA));
                }
            }
        }

        /** @brief Runs the cases of syr2k and her2k that are not typed over the element type. */
        class Unit_Syr2kHer2kEdges : public RankTwoUpdateTest
        {
            protected:
                /**
                 * @brief Expects record to throw std::invalid_argument and to add no task.
                 * @param record Callable `(SushiBLAS::BLASOps&) -> TaskHandle`.
                 */
                template<typename Record>
                void expect_refused(const char* what, Record&& record)
                {
                    SCOPED_TRACE(what);
                    SushiBLAS::BLASOps ops(probe.get_recorder());
                    EXPECT_THROW((void)record(ops), std::invalid_argument);
                    EXPECT_TRUE(probe.take_nodes().empty());
                }

                /// @brief The graph the refusal cases record on and read.
                GraphProbe probe;
        };

        TEST_F(Unit_Syr2kHer2kEdges, Her2kOnARealTensorIsSyr2k)
        {
            std::mt19937 generator(K_SEED);
            const auto a = random_matrix<float>(generator, K_ORDER, K_INNER);
            const auto b = random_matrix<float>(generator, K_ORDER, K_INNER);
            const auto c = random_matrix<float>(generator, K_ORDER, K_ORDER);

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                expect_same_bytes(
                    run_update<float, Kernels::Conjugated>(
                        mode, a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans,
                        K_ROUNDED_ALPHA, K_ROUNDED_BETA),
                    run_update<float, Kernels::Unconjugated>(
                        mode, a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans,
                        K_ROUNDED_ALPHA, K_ROUNDED_BETA));
            }
        }

        TEST_F(Unit_Syr2kHer2kEdges, ConjTransOnARealTensorIsTrans)
        {
            std::mt19937 generator(K_SEED);
            const auto a = random_matrix<float>(generator, K_INNER, K_ORDER);
            const auto b = random_matrix<float>(generator, K_INNER, K_ORDER);
            const auto c = random_matrix<float>(generator, K_ORDER, K_ORDER);

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                expect_same_bytes(
                    run_update<float, Kernels::Unconjugated>(
                        mode, a, b, c, Core::Uplo::Lower, Core::Transpose::ConjTrans,
                        K_ROUNDED_ALPHA, K_ROUNDED_BETA),
                    run_update<float, Kernels::Unconjugated>(
                        mode, a, b, c, Core::Uplo::Lower, Core::Transpose::Trans,
                        K_ROUNDED_ALPHA, K_ROUNDED_BETA));
            }
        }

        TEST_F(Unit_Syr2kHer2kEdges, EveryRefusalIsAnInvalidArgument)
        {
            const Core::Uplo upper = Core::Uplo::Upper;
            const Core::Transpose no_trans = Core::Transpose::NoTrans;
            const Tensor A = engine->create_tensor({3, 2});
            const Tensor B = engine->create_tensor({3, 2});
            Tensor C = engine->create_tensor({3, 3});
            const Tensor complex_A = engine->create_tensor({3, 2}, Core::DataType::Complex32);
            const Tensor complex_B = engine->create_tensor({3, 2}, Core::DataType::Complex32);
            Tensor complex_C = engine->create_tensor({3, 3}, Core::DataType::Complex32);
            const Tensor tall = engine->create_tensor({4, 2});

            expect_refused("uplo holding 0", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(A, B, C, static_cast<Core::Uplo>(0));
            });
            expect_refused("trans holding 0", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(A, B, C, upper, static_cast<Core::Transpose>(0));
            });
            expect_refused("syr2k with CONJ_TRANS on COMPLEX32", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(complex_A, complex_B, complex_C, upper,
                                 Core::Transpose::ConjTrans);
            });
            expect_refused("her2k with TRANS on COMPLEX32", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.her2k(complex_A, complex_B, complex_C, upper, Core::Transpose::Trans);
            });
            expect_refused("B of another dtype", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other = engine->create_tensor({3, 2}, Core::DataType::Float64);
                return ops.syr2k(A, other, C, upper);
            });
            expect_refused("C of another dtype", [&](SushiBLAS::BLASOps& ops)
            {
                Tensor other = engine->create_tensor({3, 3}, Core::DataType::Float64);
                return ops.syr2k(A, B, other, upper);
            });
            expect_refused("a complex alpha on FLOAT32 through syr2k",
                           [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(A, B, C, upper, no_trans, K_COMPLEX_SCALAR);
            });
            expect_refused("a complex alpha on FLOAT32 through her2k",
                           [&](SushiBLAS::BLASOps& ops)
            {
                return ops.her2k(A, B, C, upper, no_trans, K_COMPLEX_SCALAR);
            });
            expect_refused("a 3 x 4 C", [&](SushiBLAS::BLASOps& ops)
            {
                Tensor other = engine->create_tensor({3, 4});
                return ops.syr2k(A, B, other, upper);
            });
            expect_refused("op(A) with n + 1 rows", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(tall, B, C, upper);
            });
            expect_refused("op(B) with n + 1 rows", [&](SushiBLAS::BLASOps& ops)
            {
                return ops.syr2k(A, tall, C, upper);
            });
            expect_refused("op(B) with k + 1 columns", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other = engine->create_tensor({3, 3});
                return ops.syr2k(A, other, C, upper);
            });
            expect_refused("a column-major A with a row-major C", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor other(A.storage, {3, 2}, Core::DataType::Float32, 0,
                                   Core::Layout::ColumnMajor);
                return ops.syr2k(other, B, C, upper);
            });
            expect_refused("a batch of 3 against a batch of 4", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor batch_A = engine->create_tensor({3, 3, 2});
                const Tensor batch_B = engine->create_tensor({4, 3, 2});
                Tensor batch_C = engine->create_tensor({4, 3, 3});
                return ops.syr2k(batch_A, batch_B, batch_C, upper);
            });
            expect_refused("an INT32 A", [&](SushiBLAS::BLASOps& ops)
            {
                const Tensor int_A = engine->create_tensor({3, 2}, Core::DataType::Int32);
                const Tensor int_B = engine->create_tensor({3, 2}, Core::DataType::Int32);
                Tensor int_C = engine->create_tensor({3, 3}, Core::DataType::Int32);
                return ops.syr2k(int_A, int_B, int_C, upper);
            });
        }

        TEST_F(Unit_Syr2kHer2kEdges, AnEmptyUpdateRecordsAndRuns)
        {
            const HostMatrix<float> c =
                integer_matrix<float>(K_SEED_C, 3, 3, Core::Layout::RowMajor);

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode_name(mode));
                engine->set_mode(mode);

                const Tensor no_rows_A = engine->create_tensor({0, 2});
                const Tensor no_rows_B = engine->create_tensor({0, 2});
                Tensor no_rows_C = engine->create_tensor({0, 0});
                EXPECT_NO_THROW(SushiBLAS::BLASOps(*engine).syr2k(no_rows_A, no_rows_B, no_rows_C,
                                                                  Core::Uplo::Upper));
                EXPECT_NO_THROW(engine->execute());

                const Tensor no_cols_A = engine->create_tensor({3, 0});
                const Tensor no_cols_B = engine->create_tensor({3, 0});
                Tensor C = upload(c);
                EXPECT_NO_THROW(SushiBLAS::BLASOps(*engine).syr2k(
                    no_cols_A, no_cols_B, C, Core::Uplo::Lower, Core::Transpose::NoTrans,
                    K_INTEGER_ALPHA, 1.0));
                EXPECT_NO_THROW(engine->execute());
                expect_same_bytes(read_tensor<float>(C), c.values);
            }
        }

        /**
         * @brief Tests that the tile a record-time refusal measures is the tile of the dtype,
         *        and that an integer dtype demands nothing.
         */
        TEST(Unit_Syr2kTile, TheRecordedDemandIsTheTileOfTheDtype)
        {
            using Kernels::SYCL::Syr2kTile;

            const Device::LocalTileDemand single =
                Internal::local_tile_demand<Syr2kTile>(Core::DataType::Float32);
            EXPECT_EQ(Syr2kTile::demand<float>().local_bytes, single.local_bytes);
            EXPECT_EQ(Syr2kTile::demand<float>().work_group_items, single.work_group_items);

            const Device::LocalTileDemand wide =
                Internal::local_tile_demand<Syr2kTile>(Core::DataType::Complex64);
            EXPECT_EQ(Syr2kTile::demand<std::complex<double>>().local_bytes, wide.local_bytes);
            EXPECT_GT(wide.local_bytes, single.local_bytes);

            const Device::LocalTileDemand none =
                Internal::local_tile_demand<Syr2kTile>(Core::DataType::Int32);
            EXPECT_EQ(std::size_t{0}, none.local_bytes);
        }
    } // namespace Test
} // namespace SushiBLAS
