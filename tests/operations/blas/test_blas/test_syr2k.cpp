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
 * @file test_syr2k.cpp
 * @brief Tests syr2k and her2k against hand-computed updates, a host integer loop and a truth;
 *        the record-time refusals are in test_syr2k_refusals.cpp.
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
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include "blas_rows.hpp"

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

            /// @brief The complex alpha of every her2k case.
            constexpr Wide K_HER2K_ALPHA{1.0, 2.0};

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
            [[nodiscard]] BLASScalar blas_scalar(Wide value) noexcept
            {
                return value.imag() == 0.0 ? BLASScalar(value.real()) : BLASScalar(value);
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
                                     BLASScalar alpha, double beta)
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

            /** @brief Names the form of a rank-2k update by its transpose and its triangle. */
            struct TransposeUplo
            {
                Core::Transpose trans; /**< How A and B are read. */
                Core::Uplo uplo;       /**< The triangle of C the update writes. */
            };

            /** @brief Returns the tail of a row named by a transpose and a triangle. */
            [[nodiscard]] std::string axis_name(const TransposeUplo& form)
            {
                return "_" + word(form.trans) + "_" + word(form.uplo);
            }

            /** @brief Names a batched update by whether A carries the batch axis and the triangle. */
            struct BatchUplo
            {
                bool a_is_batched; /**< True when A has one matrix per update. */
                Core::Uplo uplo;   /**< The triangle of C the update writes. */
            };

            /** @brief Returns the tail of a row named by a batched update. */
            [[nodiscard]] std::string axis_name(const BatchUplo& form)
            {
                return std::string(form.a_is_batched ? "_ABatched_" : "_ARank2_") +
                       word(form.uplo);
            }

            /** @brief The two modes every case runs in. */
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceSYCL,
                                                             ExecutionMode::ForceReference};

            /** @brief The element types syr2k accepts that the helpers of this file build. */
            const std::vector<Core::DataType> K_SYR2K_TYPES = {
                Core::DataType::Float32, Core::DataType::Float64, Core::DataType::Complex32,
                Core::DataType::Complex64};

            /** @brief The two real types the rounding bound is measured on. */
            const std::vector<Core::DataType> K_BOUND_TYPES = {Core::DataType::Float32,
                                                               Core::DataType::Float64};

            /** @brief The two triangles of C an update can write. */
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

            /** @brief Returns every pair of one of the transposes with each triangle. */
            [[nodiscard]] std::vector<TransposeUplo>
            transpose_uplos(const std::vector<Core::Transpose>& transposes)
            {
                std::vector<TransposeUplo> forms;
                for (const Core::Transpose trans : transposes)
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        forms.push_back({trans, uplo});
                return forms;
            }

            /** @brief Returns the batched forms: A batched or of rank 2, each triangle. */
            [[nodiscard]] std::vector<BatchUplo> batch_uplos()
            {
                std::vector<BatchUplo> forms;
                for (const bool a_is_batched : {true, false})
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        forms.push_back({a_is_batched, uplo});
                return forms;
            }

            /** @brief Measures a rank-2k update against a truth, for a real element type. */
            template<typename T>
            struct UpdateBound
            {
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

                /** @brief Expects every element of the updated triangle within its bound. */
                static void expect_within_bound(ExecutionMode mode, const HostMatrix<T>& op_a,
                                                const HostMatrix<T>& op_b, const HostMatrix<T>& c,
                                                Core::Uplo uplo, const std::vector<T>& actual)
                {
                    const double unit = summing_unit(mode);
                    const std::size_t count = roundings(mode, K_INNER);
                    for (int64_t i = 0; i < K_ORDER; ++i)
                    {
                        for (int64_t j = 0; j < K_ORDER; ++j)
                        {
                            if (!is_updated(uplo, i, j))
                                continue;

                            const TruthValue truth = truth_element(op_a, op_b, c, i, j);
                            const double bound = rounding_bound(count, unit, truth.magnitude) +
                                                 unit_roundoff<T>() * std::fabs(truth.value);
                            EXPECT_NEAR(static_cast<double>(actual[c.index(i, j)]), truth.value,
                                        bound)
                                << "(" << i << ", " << j << ")";
                        }
                    }
                }
            };
        } // namespace

        /** @brief Uploads host matrices and runs rank-2k updates on them in the mode of the row. */
        class RankTwoUpdateTest : public BlasEngineStart
        {
            protected:
                /** @brief Returns a tensor of dims that holds values. */
                template<typename T>
                [[nodiscard]] Tensor upload(std::initializer_list<int64_t> dims,
                                            const std::vector<T>& values)
                {
                    Tensor tensor = engine->create_tensor(dims, sushiblas_dtype<T>());
                    EXPECT_EQ(values.size(), static_cast<std::size_t>(tensor.num_elements));
                    if (!values.empty())
                        TestEnvironment::queue()
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
                 * @brief Returns the elements of C after the update of a routine.
                 * @tparam Conj Kernels::Unconjugated for syr2k, Kernels::Conjugated for her2k.
                 */
                template<typename T, typename Conj>
                [[nodiscard]] std::vector<T> run_update(const HostMatrix<T>& a,
                                                        const HostMatrix<T>& b,
                                                        const HostMatrix<T>& c, Core::Uplo uplo,
                                                        Core::Transpose trans, Wide alpha,
                                                        double beta)
                {
                    const Tensor A = upload(a);
                    const Tensor B = upload(b);
                    Tensor C = upload(c);
                    SushiBLAS::BLASOps ops(recorder());
                    record_update<Conj>(ops, A, B, C, uplo, trans, blas_scalar(alpha), beta);
                    recording->run().value();
                    return read_tensor<T>(C);
                }

                /**
                 * @brief Expects the update to equal the host loop exactly on integer operands.
                 * @param layout The layout the engine of the row runs in.
                 */
                template<typename T, typename Conj>
                void expect_exact_on_integers(Core::Layout layout, const TransposeUplo& form,
                                              Wide alpha)
                {
                    const bool transposed = Core::is_transposed(form.trans);
                    const int64_t rows = transposed ? K_INNER : K_ORDER;
                    const int64_t cols = transposed ? K_ORDER : K_INNER;
                    const auto a = integer_matrix<T>(K_SEED_A, rows, cols, layout);
                    const auto b = integer_matrix<T>(K_SEED_B, rows, cols, layout);
                    const auto c = integer_matrix<T>(K_SEED_C, K_ORDER, K_ORDER, layout);

                    const HostMatrix<T> expected = expected_update<T, Conj>(
                        a, b, c, form.uplo, form.trans, alpha, K_INTEGER_BETA);
                    expect_equal(run_update<T, Conj>(a, b, c, form.uplo, form.trans, alpha,
                                                     K_INTEGER_BETA),
                                 expected.values);
                }

                /** @brief Expects a batched syr2k to equal the host loop on each matrix alone. */
                template<typename T>
                void expect_batch_updated_matrix_by_matrix(const BatchUplo& form)
                {
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

                    std::vector<HostMatrix<T>> expected;
                    for (std::size_t index = 0; index < c.size(); ++index)
                        expected.push_back(expected_update<T, Kernels::Unconjugated>(
                            a[form.a_is_batched ? index : 0], b[index], c[index], form.uplo,
                            Core::Transpose::NoTrans, K_INTEGER_ALPHA, K_INTEGER_BETA));

                    const Tensor A = form.a_is_batched
                        ? upload({K_BATCH, n, k}, concatenated(a))
                        : upload(a[0]);
                    const Tensor B = upload({K_BATCH, n, k}, concatenated(b));
                    Tensor C = upload({K_BATCH, n, n}, concatenated(c));

                    SushiBLAS::BLASOps(recorder()).syr2k(A, B, C, form.uplo,
                                                         Core::Transpose::NoTrans,
                                                         K_INTEGER_ALPHA, K_INTEGER_BETA);
                    recording->run().value();

                    expect_equal(read_tensor<T>(C), concatenated(expected));
                }
        };

        /** @brief Runs a case once per row of a suite over the rank-2k fixture. */
        template<typename Axis>
        using UpdateCase = BlasRowCase<RankTwoUpdateTest, Axis>;

        /** @brief Runs the hand-computed syr2k update over the element types and transposes. */
        class Medium_Syr2k : public UpdateCase<Core::Transpose>
        {

        };

        TEST_P(Medium_Syr2k, MatchesAHandComputedUpdate)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Conj = Kernels::Unconjugated;
                const Core::Layout layout = GetParam().layout;
                const Core::Transpose trans = GetParam().axis;

                const auto a = matrix_of<T>(2, 3, layout, {1, 2, 3, 4, 5, 6});
                const auto b = matrix_of<T>(2, 3, layout, {1, 0, -1, 2, 1, 0});
                const auto c = matrix_of<T>(2, 2, layout, {1, 10, 20, 1});

                // A * B^T + B * A^T is {-4 2, 2 26}; twice that minus C is {-9 -6, -16 51}.
                const auto upper = matrix_of<T>(2, 2, layout, {-9, -6, 20, 51});
                const auto lower = matrix_of<T>(2, 2, layout, {-9, 10, -16, 51});

                // Under TRANS the stored operands are the 3 x 2 transposes.
                const auto stored_a = applied(a, trans);
                const auto stored_b = applied(b, trans);

                expect_equal(this->template run_update<T, Conj>(
                                 stored_a, stored_b, c, Core::Uplo::Upper, trans,
                                 K_INTEGER_ALPHA, K_INTEGER_BETA),
                             upper.values);
                expect_equal(this->template run_update<T, Conj>(
                                 stored_a, stored_b, c, Core::Uplo::Lower, trans,
                                 K_INTEGER_ALPHA, K_INTEGER_BETA),
                             lower.values);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2k,
            ::testing::ValuesIn(rows_of(K_SYR2K_TYPES, K_BOTH_MODES,
                                        std::vector<Core::Transpose>{Core::Transpose::NoTrans,
                                                                     Core::Transpose::Trans})),
            row_name<Core::Transpose>);

        /** @brief Runs syr2k against a host integer loop, per transpose and triangle. */
        class Medium_Syr2kExact : public UpdateCase<TransposeUplo>
        {

        };

        TEST_P(Medium_Syr2kExact, IsExactOnSmallIntegers)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                this->template expect_exact_on_integers<T, Kernels::Unconjugated>(
                    GetParam().layout, GetParam().axis, K_INTEGER_ALPHA);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2kExact,
            ::testing::ValuesIn(rows_of(K_SYR2K_TYPES, K_BOTH_MODES,
                                        transpose_uplos({Core::Transpose::NoTrans,
                                                         Core::Transpose::Trans}))),
            row_name<TransposeUplo>);

        /** @brief Runs a batched syr2k against the host loop on each matrix alone. */
        class Medium_Syr2kBatch : public UpdateCase<BatchUplo>
        {

        };

        TEST_P(Medium_Syr2kBatch, ABatchIsUpdatedMatrixByMatrix)
        {
            with_non_half_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                this->template expect_batch_updated_matrix_by_matrix<T>(GetParam().axis);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2kBatch,
            ::testing::ValuesIn(row_major(rows_of(K_SYR2K_TYPES, K_BOTH_MODES, batch_uplos()))),
            row_name<BatchUplo>);

        /** @brief Measures syr2k against a truth, over the real element types that have one. */
        class Medium_Syr2kBound : public UpdateCase<TransposeUplo>
        {

        };

        TEST_P(Medium_Syr2kBound, IsWithinItsBoundOfTheTruth)
        {
            with_single_or_double_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const TransposeUplo& form = GetParam().axis;
                std::mt19937 generator(K_SEED);

                const bool transposed = Core::is_transposed(form.trans);
                const auto a = random_matrix<T>(generator, transposed ? K_INNER : K_ORDER,
                                                transposed ? K_ORDER : K_INNER);
                const auto b = random_matrix<T>(generator, a.rows, a.cols);
                const auto c = random_matrix<T>(generator, K_ORDER, K_ORDER);

                const std::vector<T> actual =
                    this->template run_update<T, Kernels::Unconjugated>(
                        a, b, c, form.uplo, form.trans, K_ROUNDED_ALPHA, K_ROUNDED_BETA);
                ASSERT_EQ(actual.size(), c.values.size());

                UpdateBound<T>::expect_within_bound(GetParam().mode, applied(a, form.trans),
                                                    applied(b, form.trans), c, form.uplo, actual);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2kBound,
            ::testing::ValuesIn(row_major(rows_of(K_BOUND_TYPES, K_BOTH_MODES,
                                                  transpose_uplos({Core::Transpose::NoTrans,
                                                                   Core::Transpose::Trans})))),
            row_name<TransposeUplo>);

        /** @brief Runs the HALF sums of syr2k; the fixture skips a device without fp16. */
        class Medium_Syr2kHalf : public UpdateCase<NoAxis>
        {

        };

        TEST_P(Medium_Syr2kHalf, HalfSumsInFloat)
        {
            const std::vector<sycl::half> a = {sycl::half(2048.0f), sycl::half(1.0f),
                                               sycl::half(-2048.0f)};
            const std::vector<sycl::half> ones(3, sycl::half(1.0f));

            const Tensor A = upload({1, 3}, a);
            const Tensor B = upload({1, 3}, ones);
            Tensor C = upload({1, 1}, std::vector<sycl::half>(1, sycl::half(0.0f)));

            SushiBLAS::BLASOps(recorder()).syr2k(A, B, C, Core::Uplo::Upper,
                                                 Core::Transpose::NoTrans, 1.0, 0.0);
            recording->run().value();

            // Each product sums 2048 + 1 - 2048, which is 1 in FLOAT32 and 0 in HALF.
            const std::vector<sycl::half> result = read_tensor<sycl::half>(C);
            ASSERT_EQ(1u, result.size());
            EXPECT_EQ(static_cast<float>(result[0]), 2.0f);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2kHalf,
            ::testing::ValuesIn(row_major(plain_rows({Core::DataType::Half}, K_BOTH_MODES))),
            row_name<NoAxis>);

        /** @brief Runs the case that hands a kernel an event still pending on a host task. */
        class Medium_Syr2kFixedType : public RankTwoUpdateTest
        {

        };

        TEST_F(Medium_Syr2kFixedType, WaitsForItsDependencies)
        {
            const Core::Layout layout = Core::Layout::RowMajor;
            const Tensor A = upload(matrix_of<float>(2, 3, layout, {1, 2, 3, 4, 5, 6}));
            const Tensor B = upload(matrix_of<float>(2, 3, layout, {1, 0, -1, 2, 1, 0}));
            Tensor C = upload(matrix_of<float>(2, 2, layout, {1, 10, 20, 1}));

            sycl::queue& queue = TestEnvironment::queue();
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

        /** @brief Runs the hand-computed her2k update over the complex element types. */
        class Medium_Her2k : public UpdateCase<NoAxis>
        {

        };

        TEST_P(Medium_Her2k, MatchesAHandComputedUpdate)
        {
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Core::Layout layout = GetParam().layout;
                const Wide untouched(K_UNTOUCHED, K_UNTOUCHED);

                const auto a = matrix_of<T>(2, 1, layout, {Wide(1, 1), Wide(2, 0)});
                const auto b = matrix_of<T>(2, 1, layout, {Wide(0, 1), Wide(1, -1)});
                const auto c = matrix_of<T>(2, 2, layout,
                                            {Wide(1, 5), Wide(3, -1), untouched, Wide(4, 0)});
                const auto expected = matrix_of<T>(
                    2, 2, layout, {Wide(8, 0), Wide(6, 2), untouched, Wide(4, 0)});

                const std::vector<T> actual = this->template run_update<T, Kernels::Conjugated>(
                    a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans, K_HER2K_ALPHA, 2.0);

                // C(0, 0) is 6 + 2 * (1 + 5i) with its imaginary part dropped.
                expect_equal(actual, expected.values);
                EXPECT_EQ(actual[expected.index(0, 0)].imag(), 0);
                EXPECT_EQ(actual[expected.index(1, 1)].imag(), 0);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Her2k, ::testing::ValuesIn(plain_rows(K_COMPLEX_TYPES, K_BOTH_MODES)),
            row_name<NoAxis>);

        /** @brief Runs her2k against a host integer loop, per transpose and triangle. */
        class Medium_Her2kExact : public UpdateCase<TransposeUplo>
        {

        };

        TEST_P(Medium_Her2kExact, IsExactOnGaussianIntegers)
        {
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                this->template expect_exact_on_integers<T, Kernels::Conjugated>(
                    GetParam().layout, GetParam().axis, K_HER2K_ALPHA);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Her2kExact,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES,
                                        transpose_uplos({Core::Transpose::NoTrans,
                                                         Core::Transpose::ConjTrans}))),
            row_name<TransposeUplo>);

        /** @brief Runs her2k with ConjTrans against NoTrans on the conjugate transposes. */
        class Medium_Her2kConjTrans : public UpdateCase<Core::Uplo>
        {

        };

        TEST_P(Medium_Her2kConjTrans, ConjTransEqualsNoTransOnTheConjugateTransposes)
        {
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using Conj = Kernels::Conjugated;
                const Wide alpha = K_HER2K_ALPHA;
                const Core::Layout layout = Core::Layout::RowMajor;
                const Core::Uplo uplo = GetParam().axis;
                const auto a = integer_matrix<T>(K_SEED_A, K_INNER, K_ORDER, layout);
                const auto b = integer_matrix<T>(K_SEED_B, K_INNER, K_ORDER, layout);
                const auto c = integer_matrix<T>(K_SEED_C, K_ORDER, K_ORDER, layout);
                const auto a_h = applied(a, Core::Transpose::ConjTrans);
                const auto b_h = applied(b, Core::Transpose::ConjTrans);

                const std::vector<T> conj_trans = this->template run_update<T, Conj>(
                    a, b, c, uplo, Core::Transpose::ConjTrans, alpha, K_INTEGER_BETA);
                this->start(layout, GetParam().mode);
                const std::vector<T> no_trans = this->template run_update<T, Conj>(
                    a_h, b_h, c, uplo, Core::Transpose::NoTrans, alpha, K_INTEGER_BETA);

                expect_equal(conj_trans, no_trans);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Her2kConjTrans,
            ::testing::ValuesIn(row_major(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_TRIANGLES))),
            row_name<Core::Uplo>);

        /** @brief Runs the FLOAT32 cases of syr2k and her2k that forward or have no work. */
        class Medium_Syr2kHer2kEdges : public UpdateCase<NoAxis>
        {

        };

        TEST_P(Medium_Syr2kHer2kEdges, Her2kOnARealTensorIsSyr2k)
        {
            std::mt19937 generator(K_SEED);
            const auto a = random_matrix<float>(generator, K_ORDER, K_INNER);
            const auto b = random_matrix<float>(generator, K_ORDER, K_INNER);
            const auto c = random_matrix<float>(generator, K_ORDER, K_ORDER);

            const std::vector<float> from_her2k = run_update<float, Kernels::Conjugated>(
                a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans, K_ROUNDED_ALPHA,
                K_ROUNDED_BETA);
            start(GetParam().layout, GetParam().mode);
            const std::vector<float> from_syr2k = run_update<float, Kernels::Unconjugated>(
                a, b, c, Core::Uplo::Upper, Core::Transpose::NoTrans, K_ROUNDED_ALPHA,
                K_ROUNDED_BETA);

            expect_same_bytes(from_her2k, from_syr2k);
        }

        TEST_P(Medium_Syr2kHer2kEdges, ConjTransOnARealTensorIsTrans)
        {
            std::mt19937 generator(K_SEED);
            const auto a = random_matrix<float>(generator, K_INNER, K_ORDER);
            const auto b = random_matrix<float>(generator, K_INNER, K_ORDER);
            const auto c = random_matrix<float>(generator, K_ORDER, K_ORDER);

            const std::vector<float> conj_trans = run_update<float, Kernels::Unconjugated>(
                a, b, c, Core::Uplo::Lower, Core::Transpose::ConjTrans, K_ROUNDED_ALPHA,
                K_ROUNDED_BETA);
            start(GetParam().layout, GetParam().mode);
            const std::vector<float> trans = run_update<float, Kernels::Unconjugated>(
                a, b, c, Core::Uplo::Lower, Core::Transpose::Trans, K_ROUNDED_ALPHA,
                K_ROUNDED_BETA);

            expect_same_bytes(conj_trans, trans);
        }

        TEST_P(Medium_Syr2kHer2kEdges, AnUpdateOfNoRowsRecordsAndRuns)
        {
            const Tensor no_rows_A = engine->create_tensor({0, 2});
            const Tensor no_rows_B = engine->create_tensor({0, 2});
            Tensor no_rows_C = engine->create_tensor({0, 0});

            SushiBLAS::BLASOps(recorder()).syr2k(no_rows_A, no_rows_B, no_rows_C,
                                                 Core::Uplo::Upper);

            EXPECT_TRUE(recording->run().has_value());
        }

        TEST_P(Medium_Syr2kHer2kEdges, AnUpdateOfNoColumnsLeavesCUnchanged)
        {
            const HostMatrix<float> c =
                integer_matrix<float>(K_SEED_C, 3, 3, Core::Layout::RowMajor);
            const Tensor no_cols_A = engine->create_tensor({3, 0});
            const Tensor no_cols_B = engine->create_tensor({3, 0});
            Tensor C = upload(c);

            SushiBLAS::BLASOps(recorder()).syr2k(no_cols_A, no_cols_B, C, Core::Uplo::Lower,
                                                 Core::Transpose::NoTrans, K_INTEGER_ALPHA, 1.0);

            ASSERT_TRUE(recording->run().has_value());
            expect_same_bytes(read_tensor<float>(C), c.values);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Syr2kHer2kEdges,
            ::testing::ValuesIn(row_major(plain_rows({Core::DataType::Float32}, K_BOTH_MODES))),
            row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
