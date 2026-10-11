/**************************************************************************/
/* test_blas_packed.cpp                                                   */
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
 * @file test_blas_packed.cpp
 * @brief Tests the packed routines against hand-computed results, their dense siblings and
 *        each other's kernels.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include "test_utils.hpp"
#include "blas_structured.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The type hand-written and integer test data is stated in.
            using Wide = std::complex<double>;

            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Fills both parts of every slot a call must neither read nor write.
            constexpr double K_POISON = 999.0;

            /// @brief Imaginary part given to a stored Hermitian diagonal, which is not read.
            constexpr double K_DIAGONAL_IMAGINARY = 9.0;

            /// @brief Scales the product or the update in the cases that compare two results.
            constexpr double K_ALPHA = 1.5;

            /// @brief Scales the prior y in the cases that compare two results.
            constexpr double K_BETA = -0.5;

            /// @brief Order of the 3 x 3 hand cases.
            constexpr int64_t K_HAND_ORDER = 3;

            /// @brief Order of the matrices compared against their dense siblings.
            constexpr int64_t K_SIBLING_ORDER = 17;

            /// @brief Order at which a device that plans a work-group per row runs one.
            constexpr int64_t K_LARGE_ORDER = 2048;

            /// @brief Order of the matrices the two kernels are compared on.
            constexpr int64_t K_MODE_ORDER = 64;

            /// @brief Both kernels an operation has.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both orders a packed triangle can be stored in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief Both triangles a packed tensor can hold.
            constexpr Core::Uplo K_TRIANGLES[] = {Core::Uplo::Upper, Core::Uplo::Lower};

            /// @brief Every form op(A) takes.
            constexpr Core::Transpose K_TRANSPOSES[] = {Core::Transpose::NoTrans,
                                                        Core::Transpose::Trans,
                                                        Core::Transpose::ConjTrans};

            /// @brief Both ways a triangular diagonal is read.
            constexpr Core::Diag K_DIAGONALS[] = {Core::Diag::NonUnit, Core::Diag::Unit};

            /**
             * @brief U = [1 2 3; 0 4 5; 0 0 6] or its transpose packed in lines that grow:
             *        UPPER in COLUMN_MAJOR, LOWER in ROW_MAJOR.
             */
            const std::vector<double> K_HAND_GROWING = {1, 2, 4, 3, 5, 6};

            /**
             * @brief The same triangles packed in lines that shrink: UPPER in ROW_MAJOR, LOWER
             *        in COLUMN_MAJOR.
             */
            const std::vector<double> K_HAND_SHRINKING = {1, 2, 3, 4, 5, 6};

            /// @brief The packed routines, in the order a list of their eight results has.
            constexpr const char* K_ROUTINES[] = {"spmv", "hpmv", "tpmv", "tpsv",
                                                  "spr",  "hpr",  "spr2", "hpr2"};

            /// @brief The number of packed routines.
            constexpr std::size_t K_ROUTINE_COUNT = sizeof(K_ROUTINES) / sizeof(K_ROUTINES[0]);

            /// @brief Position of y of spmv in a list of the eight results.
            constexpr std::size_t K_SPMV = 0;

            /// @brief Position of y of hpmv.
            constexpr std::size_t K_HPMV = 1;

            /// @brief Position of x of tpmv.
            constexpr std::size_t K_TPMV = 2;

            /// @brief Position of b of tpsv.
            constexpr std::size_t K_TPSV = 3;

            /// @brief Position of AP of spr; the three other updates follow it.
            constexpr std::size_t K_SPR = 4;

            /// @brief Position of AP of hpr.
            constexpr std::size_t K_HPR = 5;

            /// @brief Position of AP of spr2.
            constexpr std::size_t K_SPR2 = 6;

            /// @brief Position of AP of hpr2.
            constexpr std::size_t K_HPR2 = 7;

            /** @brief Holds the flags of one triangular product of [1, 2, 3] and its result. */
            struct TriangularRow
            {
                Core::Uplo uplo;              /**< UPPER for U, LOWER for its transpose. */
                Core::Transpose trans;        /**< The form of op(A). */
                Core::Diag diag;              /**< How the diagonal is read. */
                double product[K_HAND_ORDER]; /**< op(A) * [1, 2, 3]. */
            };

            /// @brief The triangular products of the design, section 10.3.
            constexpr TriangularRow K_TRIANGULAR_ROWS[] = {
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::NonUnit, {14, 23, 18}},
                {Core::Uplo::Upper, Core::Transpose::Trans, Core::Diag::NonUnit, {1, 10, 31}},
                {Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::Unit, {14, 17, 3}},
                {Core::Uplo::Lower, Core::Transpose::NoTrans, Core::Diag::NonUnit, {1, 10, 31}},
                {Core::Uplo::Lower, Core::Transpose::Trans, Core::Diag::NonUnit, {14, 23, 18}}};

            /** @brief Holds the operands of the eight routines at one small order. */
            struct EdgeRow
            {
                Core::Uplo uplo;        /**< The triangle AP holds. */
                std::vector<double> ap; /**< The packed matrix. */
                std::vector<double> x;  /**< The first vector. */
                std::vector<double> y;  /**< The second vector of spr2 and hpr2. */
                std::vector<double> b;  /**< The right-hand side of tpsv: A * x. */

                /// @brief What each routine leaves, in the order of K_ROUTINES.
                std::vector<double> results[K_ROUTINE_COUNT];
            };

            /**
             * @brief The hand values of orders 1 and 2; [1, 2, 3] packs [1 2; 2 3] in every
             *        layout and triangle.
             */
            const EdgeRow K_EDGE_ROWS[] = {
                {Core::Uplo::Upper, {3}, {2}, {5}, {6},
                 {{6}, {6}, {6}, {2}, {7}, {7}, {23}, {23}}},
                {Core::Uplo::Lower, {3}, {2}, {5}, {6},
                 {{6}, {6}, {6}, {2}, {7}, {7}, {23}, {23}}},
                {Core::Uplo::Upper, {1, 2, 3}, {1, 2}, {3, 5}, {5, 6},
                 {{5, 8}, {5, 8}, {5, 6}, {1, 2}, {2, 4, 7}, {2, 4, 7}, {7, 13, 23},
                  {7, 13, 23}}},
                {Core::Uplo::Lower, {1, 2, 3}, {1, 2}, {3, 5}, {1, 8},
                 {{5, 8}, {5, 8}, {1, 8}, {1, 2}, {2, 4, 7}, {2, 4, 7}, {7, 13, 23},
                  {7, 13, 23}}}};

            /**
             * @brief Holds how far the results of two kernels may differ, as a fraction of the
             *        larger of 1 and the expected magnitude.
             * @tparam R sycl::half, float or double.
             */
            template<typename R>
            struct ModeTolerance;

            /** @brief Allows two units in the last place of HALF. */
            template<>
            struct ModeTolerance<sycl::half>
            {
                static constexpr double K_VALUE = 0.001953125; /**< 2^-9. */
            };

            /** @brief Allows the rounding of a sum of 64 FLOAT32 products of unit size. */
            template<>
            struct ModeTolerance<float>
            {
                static constexpr double K_VALUE = 1e-4; /**< About 800 epsilon. */
            };

            /** @brief Allows the rounding of a sum of 64 FLOAT64 products of unit size. */
            template<>
            struct ModeTolerance<double>
            {
                static constexpr double K_VALUE = 1e-11; /**< About 45000 epsilon. */
            };

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns the names of the flags of a triangular call. */
            [[nodiscard]] std::string flags_name(Core::Uplo uplo, Core::Transpose trans,
                                                 Core::Diag diag)
            {
                return std::string(Core::to_string(uplo)) + ", " + Core::to_string(trans) + ", " +
                       Core::to_string(diag);
            }

            /** @brief Returns whether a triangle is packed in lines that grow under a layout. */
            [[nodiscard]] constexpr bool is_growing(Core::Uplo uplo, Core::Layout layout) noexcept
            {
                return (uplo == Core::Uplo::Upper) == (layout == Core::Layout::ColumnMajor);
            }

            /** @brief Returns whether the test device computes in the component type of T. */
            template<typename T>
            [[nodiscard]] bool device_runs()
            {
                using R = Kernels::RealType<T>;
                const sycl::device device =
                    TestEnvironment::queue().get_device();
                if constexpr (std::is_same_v<R, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<R, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns value in T; a real T drops the imaginary part. */
            template<typename T>
            [[nodiscard]] T narrowed(const Wide& value)
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    using R = typename T::value_type;
                    return T(static_cast<R>(value.real()), static_cast<R>(value.imag()));
                }
                else if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value.real()));
                else
                    return static_cast<T>(value.real());
            }

            /** @brief Returns value as a complex double; a real T has a zero imaginary part. */
            template<typename T>
            [[nodiscard]] Wide widened(const T& value)
            {
                if constexpr (Core::is_complex_v<T>)
                    return Detail::to_complex_double(value);
                else
                    return Wide(Detail::to_double(value), 0.0);
            }

            /** @brief Returns values in T. */
            template<typename T>
            [[nodiscard]] std::vector<T> narrowed_all(const std::vector<Wide>& values)
            {
                std::vector<T> out;
                out.reserve(values.size());
                for (const Wide& value : values)
                    out.push_back(narrowed<T>(value));
                return out;
            }

            /** @brief Returns real values in T. */
            template<typename T>
            [[nodiscard]] std::vector<T> elements(const std::vector<double>& values)
            {
                std::vector<T> out;
                out.reserve(values.size());
                for (const double value : values)
                    out.push_back(narrowed<T>(Wide(value, 0.0)));
                return out;
            }

            /** @brief Returns the value that marks a slot no kernel may read or write. */
            template<typename T>
            [[nodiscard]] T poison()
            {
                return narrowed<T>(Wide(K_POISON, K_POISON));
            }

            /** @brief Returns count elements with every part drawn from [-1, 1]. */
            template<typename T>
            [[nodiscard]] std::vector<T> random_elements(std::mt19937& generator, int64_t count)
            {
                std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                std::vector<T> out(static_cast<std::size_t>(count));
                for (T& element : out)
                {
                    const double real = distribution(generator);
                    const double imaginary = distribution(generator);
                    element = narrowed<T>(Wide(real, imaginary));
                }
                return out;
            }

            /** @brief Returns an integer drawn from low to low + count - 1. */
            [[nodiscard]] double draw_integer(std::mt19937& generator, int low, unsigned count)
            {
                return static_cast<double>(low + static_cast<int>(generator() % count));
            }

            /**
             * @brief Holds a triangular system whose every product and solve is exact in HALF.
             */
            struct ExactTriangle
            {
                std::vector<Wide> stored; /**< The matrix a call receives, row by row. */
                std::vector<Wide> x;      /**< The vector of the product, the solution. */
                std::vector<Wide> b;      /**< op(T) * x, the right-hand side of the solve. */
            };

            /**
             * @brief Returns a triangle of small integers with a diagonal of 1, 2, -2 or 4, an
             *        integer x, and op(T) * x formed in integers.
             * @param complex_parts Whether the off-diagonal elements and x have imaginary parts.
             * @return stored holds K_POISON in the other triangle, and on the diagonal under UNIT.
             */
            [[nodiscard]] ExactTriangle exact_triangle(int64_t n, Core::Uplo uplo,
                                                       Core::Transpose trans, Core::Diag diag,
                                                       bool complex_parts)
            {
                constexpr double diagonals[] = {1.0, 2.0, -2.0, 4.0};
                const bool upper = (uplo == Core::Uplo::Upper);
                const bool unit = (diag == Core::Diag::Unit);
                const auto at = [n](int64_t r, int64_t c)
                {
                    return static_cast<std::size_t>(r * n + c);
                };

                std::mt19937 generator(K_SEED);
                ExactTriangle out;
                out.stored.assign(static_cast<std::size_t>(n * n), Wide(K_POISON, K_POISON));
                std::vector<Wide> model(static_cast<std::size_t>(n * n));

                for (int64_t r = 0; r < n; ++r)
                {
                    for (int64_t c = 0; c < n; ++c)
                    {
                        if (upper ? (r > c) : (r < c))
                            continue;

                        if (r == c)
                        {
                            const Wide value(diagonals[generator() % 4u], 0.0);
                            model[at(r, c)] = unit ? Wide(1.0, 0.0) : value;
                            if (!unit)
                                out.stored[at(r, c)] = value;
                            continue;
                        }

                        const double real = draw_integer(generator, -2, 5u);
                        const double imaginary = draw_integer(generator, -2, 5u);
                        model[at(r, c)] = Wide(real, complex_parts ? imaginary : 0.0);
                        out.stored[at(r, c)] = model[at(r, c)];
                    }
                }

                out.x.resize(static_cast<std::size_t>(n));
                for (Wide& element : out.x)
                {
                    const double real = draw_integer(generator, -3, 7u);
                    const double imaginary = draw_integer(generator, -3, 7u);
                    element = Wide(real, complex_parts ? imaginary : 0.0);
                }

                out.b.assign(static_cast<std::size_t>(n), Wide());
                for (int64_t r = 0; r < n; ++r)
                {
                    for (int64_t c = 0; c < n; ++c)
                    {
                        Wide element = Core::is_transposed(trans) ? model[at(c, r)]
                                                                  : model[at(r, c)];
                        if (Core::is_conjugated(trans))
                            element = std::conj(element);
                        out.b[static_cast<std::size_t>(r)] +=
                            element * out.x[static_cast<std::size_t>(c)];
                    }
                }
                return out;
            }

            /** @brief Expects both parts of every element to equal its expectation exactly. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    EXPECT_EQ(a.real(), e.real()) << "real part of element " << i;
                    EXPECT_EQ(a.imag(), e.imag()) << "imaginary part of element " << i;
                }
            }

            /** @brief Expects both parts of every element within ModeTolerance of expected. */
            template<typename T>
            void expect_near(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                const double tolerance = ModeTolerance<Kernels::RealType<T>>::K_VALUE;

                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    const double bound = tolerance * std::max(1.0, std::abs(e));
                    EXPECT_LE(std::abs(a.real() - e.real()), bound)
                        << "real part of element " << i;
                    EXPECT_LE(std::abs(a.imag() - e.imag()), bound)
                        << "imaginary part of element " << i;
                }
            }
        } // namespace

        /**
         * @brief Builds packed, dense and vector operands of element type T and records the
         *        packed routines on them.
         */
        template<typename T>
        class BLASPackedCases : public SushiBLASTest
        {
            protected:
                /** @brief Holds the operands and the scalars of one call of each routine. */
                struct PackedInputs
                {
                    Core::Uplo uplo;           /**< The triangle every packed operand holds. */
                    std::vector<T> ap;         /**< The matrix of the products and the updates. */
                    std::vector<T> triangular; /**< The matrix of tpmv and tpsv. */
                    std::vector<T> x;          /**< The first vector. */
                    std::vector<T> y;          /**< The second vector of spr2 and hpr2. */
                    std::vector<T> prior;      /**< What y of spmv and hpmv holds before. */
                    std::vector<T> b;          /**< The right-hand side of tpsv. */
                    double alpha;              /**< Scales every product and update. */
                    double beta;               /**< Scales prior. */
                };

                /** @brief Skips the case on a device that lacks the component type of T. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>())
                        GTEST_SKIP() << "the device does not compute in this element type";
                }

                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->engine->set_mode(mode);
                }

                /**
                 * @brief Runs body once in each layout under each kernel.
                 * @param body Callable `(Core::Layout)`, called after the engine is rebuilt.
                 */
                template<typename Body>
                void for_each_engine(Body&& body)
                {
                    for (const Core::Layout layout : K_LAYOUTS)
                    {
                        for (const ExecutionMode mode : K_MODES)
                        {
                            SCOPED_TRACE(engine_name(layout, mode));
                            start(layout, mode);
                            body(layout);
                        }
                    }
                }

                /** @brief Returns a tensor of dims that holds stored in storage order. */
                [[nodiscard]] Tensor tensor_of(const std::vector<int64_t>& dims,
                                               const std::vector<T>& stored)
                {
                    Tensor tensor = this->engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        sushiblas_dtype<T>());
                    EXPECT_EQ(stored.size(), static_cast<std::size_t>(tensor.num_elements));
                    if (!stored.empty())
                        TestEnvironment::queue()
                            .memcpy(tensor.data(), stored.data(), stored.size() * sizeof(T))
                            .wait();
                    return tensor;
                }

                /** @brief Returns a rank-1 tensor that holds values. */
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns an n x n tensor that holds the row-ordered values. */
                [[nodiscard]] Tensor dense_of(int64_t n, const std::vector<T>& row_ordered)
                {
                    return tensor_of({n, n},
                                     in_layout(row_ordered, n, n, this->engine->layout()));
                }

                /** @brief Returns the packed values of one triangle of a dense n x n matrix. */
                [[nodiscard]] std::vector<T> packed(const std::vector<T>& dense, int64_t n,
                                                    Core::Uplo uplo) const
                {
                    return to_packed(dense, n, uplo, this->engine->layout());
                }

                /** @brief Returns the packed tensor of K_HAND_GROWING or K_HAND_SHRINKING. */
                [[nodiscard]] Tensor hand_triangle(Core::Uplo uplo)
                {
                    return vector_of(elements<T>(is_growing(uplo, this->engine->layout())
                                                     ? K_HAND_GROWING
                                                     : K_HAND_SHRINKING));
                }

                /** @brief Returns the elements of a tensor, or none for one without storage. */
                [[nodiscard]] static std::vector<T> read(const Tensor& tensor)
                {
                    return tensor.num_elements == 0 ? std::vector<T>() : read_tensor<T>(tensor);
                }

                /** @brief Returns the elements of an n x n tensor, row by row. */
                [[nodiscard]] static std::vector<T> read_row_ordered(const Tensor& tensor,
                                                                     int64_t n)
                {
                    const std::vector<T> stored = read_tensor<T>(tensor);
                    if (tensor.layout == Core::Layout::RowMajor)
                        return stored;

                    std::vector<T> row_ordered(stored.size());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        for (int64_t j = 0; j < n; ++j)
                            row_ordered[static_cast<std::size_t>(i * n + j)] =
                                stored[static_cast<std::size_t>(i + j * n)];
                    }
                    return row_ordered;
                }

                /**
                 * @brief Records two calls on copies of initial in one graph, runs it and expects
                 *        the same elements from both.
                 * @param first Callable `(Tensor&)` that records on the first copy.
                 * @param second Callable `(Tensor&)` that records on the second copy.
                 * @return What the first call left.
                 */
                template<typename First, typename Second>
                std::vector<T> expect_same_result(const std::vector<T>& initial, First&& first,
                                                  Second&& second)
                {
                    Tensor from_first = vector_of(initial);
                    Tensor from_second = vector_of(initial);
                    first(from_first);
                    second(from_second);
                    this->engine->execute().value();

                    const std::vector<T> result = read_tensor<T>(from_first);
                    expect_equal(result, read_tensor<T>(from_second));
                    return result;
                }

                /**
                 * @brief Records a packed update and its dense sibling in one graph, runs it and
                 *        expects the packed result to be the stored triangle of the dense one.
                 * @param packed_call Callable `(Tensor&)` that records on the packed matrix.
                 * @param dense_call Callable `(Tensor&)` that records on the dense matrix.
                 */
                template<typename Packed, typename Dense>
                void expect_same_update(const std::vector<T>& dense, int64_t n, Core::Uplo uplo,
                                        Packed&& packed_call, Dense&& dense_call)
                {
                    Tensor from_packed = vector_of(packed(dense, n, uplo));
                    Tensor from_dense = dense_of(n, dense);
                    packed_call(from_packed);
                    dense_call(from_dense);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(from_packed),
                                 packed(read_row_ordered(from_dense, n), n, uplo));
                }

                /**
                 * @brief Returns what the eight packed routines leave, in the order of K_ROUTINES,
                 *        all recorded in one graph.
                 */
                [[nodiscard]] std::vector<std::vector<T>> run_eight(const PackedInputs& in)
                {
                    const Tensor ap = vector_of(in.ap);
                    const Tensor triangular = vector_of(in.triangular);
                    const Tensor x = vector_of(in.x);
                    const Tensor y = vector_of(in.y);

                    std::vector<Tensor> results;
                    results.reserve(K_ROUTINE_COUNT);
                    results.push_back(vector_of(in.prior));
                    results.push_back(vector_of(in.prior));
                    results.push_back(vector_of(in.x));
                    results.push_back(vector_of(in.b));
                    for (std::size_t update = K_SPR; update < K_ROUTINE_COUNT; ++update)
                        results.push_back(vector_of(in.ap));

                    SushiBLAS::BLASOps ops(*this->engine);
                    ops.spmv(ap, x, results[K_SPMV], in.uplo, in.alpha, in.beta);
                    ops.hpmv(ap, x, results[K_HPMV], in.uplo, in.alpha, in.beta);
                    ops.tpmv(triangular, results[K_TPMV], in.uplo);
                    ops.tpsv(triangular, results[K_TPSV], in.uplo);
                    ops.spr(x, results[K_SPR], in.uplo, in.alpha);
                    ops.hpr(x, results[K_HPR], in.uplo, in.alpha);
                    ops.spr2(x, y, results[K_SPR2], in.uplo, in.alpha);
                    ops.hpr2(x, y, results[K_HPR2], in.uplo, in.alpha);
                    this->engine->execute().value();

                    std::vector<std::vector<T>> values;
                    values.reserve(results.size());
                    for (const Tensor& result : results)
                        values.push_back(read(result));
                    return values;
                }

                /**
                 * @brief Returns y of spmv on a packed upper triangle of order K_LARGE_ORDER,
                 *        then y of symv on the same matrix.
                 */
                [[nodiscard]] std::array<std::vector<T>, 2> large_symmetric_products()
                {
                    const int64_t n = K_LARGE_ORDER;
                    std::mt19937 generator(K_SEED);
                    const std::vector<T> dense = random_elements<T>(generator, n * n);

                    const Tensor ap = vector_of(packed(dense, n, Core::Uplo::Upper));
                    const Tensor full = dense_of(n, dense);
                    const Tensor x = vector_of(random_elements<T>(generator, n));
                    const std::vector<T> y = random_elements<T>(generator, n);
                    Tensor from_packed = vector_of(y);
                    Tensor from_dense = vector_of(y);

                    SushiBLAS::BLASOps ops(*this->engine);
                    ops.spmv(ap, x, from_packed, Core::Uplo::Upper, K_ALPHA, K_BETA);
                    ops.symv(full, x, from_dense, Core::Uplo::Upper, K_ALPHA, K_BETA);
                    this->engine->execute().value();
                    return {{read_tensor<T>(from_packed), read_tensor<T>(from_dense)}};
                }
        };

        /** @brief Runs the packed routines over the five element types. */
        template<typename T>
        class Unit_BlasPacked : public BLASPackedCases<T>
        {

        };

        /** @brief Runs the conjugating packed cases over the complex element types. */
        template<typename T>
        class Unit_BlasPackedComplex : public BLASPackedCases<T>
        {

        };

        /// @brief Names the suite of the HALF accumulation cases.
        using Unit_BlasPackedHalf = BLASPackedCases<sycl::half>;

        /// @brief Names the suite of the FLOAT32 cases of order K_LARGE_ORDER.
        using Unit_BlasPackedLarge = BLASPackedCases<float>;

        /// @brief Names the five element types the packed routines accept.
        using BLASPackedTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                                 std::complex<double>>;
        /// @brief Names the complex element types, the only ones a conjugation changes.
        using BLASPackedComplexTypes =
            ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_BlasPacked, BLASPackedTypes);
        SB_TYPED_TEST_SUITE(Unit_BlasPackedComplex, BLASPackedComplexTypes);

        /**
         * @brief Tests spmv on the literal packed arrays of [1 2 3; 2 4 5; 3 5 6].
         */
        TYPED_TEST(Unit_BlasPacked, SpmvMatchesByHand)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor AP = this->hand_triangle(uplo);
                    const Tensor x = this->vector_of(elements<T>({1, -1, 2}));
                    Tensor y = this->vector_of(std::vector<T>(3, poison<T>()));

                    SushiBLAS::BLASOps(*this->engine).spmv(AP, x, y, uplo);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(y), elements<T>({5, 8, 10}));
                }
            });
        }

        /**
         * @brief Tests tpmv on the literal packed arrays of U and of its transpose.
         */
        TYPED_TEST(Unit_BlasPacked, TpmvMatchesByHand)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const TriangularRow& row : K_TRIANGULAR_ROWS)
                {
                    SCOPED_TRACE(flags_name(row.uplo, row.trans, row.diag));
                    const Tensor AP = this->hand_triangle(row.uplo);
                    Tensor x = this->vector_of(elements<T>({1, 2, 3}));

                    SushiBLAS::BLASOps(*this->engine).tpmv(AP, x, row.uplo, row.trans, row.diag);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(x),
                                 elements<T>({row.product[0], row.product[1], row.product[2]}));
                }
            });
        }

        /**
         * @brief Tests that tpsv recovers [1, 2, 3] from each product of TpmvMatchesByHand.
         */
        TYPED_TEST(Unit_BlasPacked, TpsvInvertsTpmv)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const TriangularRow& row : K_TRIANGULAR_ROWS)
                {
                    SCOPED_TRACE(flags_name(row.uplo, row.trans, row.diag));
                    const Tensor AP = this->hand_triangle(row.uplo);
                    Tensor b = this->vector_of(
                        elements<T>({row.product[0], row.product[1], row.product[2]}));

                    SushiBLAS::BLASOps(*this->engine).tpsv(AP, b, row.uplo, row.trans, row.diag);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(b), elements<T>({1, 2, 3}));
                }
            });
        }

        /**
         * @brief Tests that spr writes 2 * x * x^T in the order the layout and triangle give.
         */
        TYPED_TEST(Unit_BlasPacked, SprWritesTheStoredTriangle)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor x = this->vector_of(elements<T>({1, 2, 3}));
                    Tensor AP = this->vector_of(elements<T>({0, 0, 0, 0, 0, 0}));

                    SushiBLAS::BLASOps(*this->engine).spr(x, AP, uplo, 2.0);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(AP),
                                 is_growing(uplo, layout) ? elements<T>({2, 4, 8, 6, 12, 18})
                                                          : elements<T>({2, 4, 6, 8, 12, 18}));
                }
            });
        }

        /**
         * @brief Tests spr2 on x = [1, 2], y = [3, 5]: the triangle of [6 11; 11 20].
         */
        TYPED_TEST(Unit_BlasPacked, Spr2MatchesByHand)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor x = this->vector_of(elements<T>({1, 2}));
                    const Tensor y = this->vector_of(elements<T>({3, 5}));
                    Tensor AP = this->vector_of(elements<T>({0, 0, 0}));

                    SushiBLAS::BLASOps(*this->engine).spr2(x, y, AP, uplo, 1.0);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(AP), elements<T>({6, 11, 20}));
                }
            });
        }

        /**
         * @brief Tests that a packed triangle gives the elements of symv and hemv, and that
         *        each packed update leaves the stored triangle of its dense sibling.
         */
        TYPED_TEST(Unit_BlasPacked, APackedTriangleEqualsTheDenseSibling)
        {
            using T = TypeParam;
            const int64_t n = K_SIBLING_ORDER;

            this->for_each_engine([&](Core::Layout)
            {
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(*this->engine);
                const std::vector<T> dense = random_elements<T>(generator, n * n);
                const Tensor full = this->dense_of(n, dense);

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor AP = this->vector_of(this->packed(dense, n, uplo));
                    const Tensor x = this->vector_of(random_elements<T>(generator, n));
                    const Tensor y = this->vector_of(random_elements<T>(generator, n));
                    const std::vector<T> prior = random_elements<T>(generator, n);

                    this->expect_same_result(
                        prior,
                        [&](Tensor& out) { ops.spmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                        [&](Tensor& out) { ops.symv(full, x, out, uplo, K_ALPHA, K_BETA); });
                    this->expect_same_result(
                        prior,
                        [&](Tensor& out) { ops.hpmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                        [&](Tensor& out) { ops.hemv(full, x, out, uplo, K_ALPHA, K_BETA); });

                    this->expect_same_update(
                        dense, n, uplo,
                        [&](Tensor& out) { ops.spr(x, out, uplo, K_ALPHA); },
                        [&](Tensor& out) { ops.syr(x, out, uplo, K_ALPHA); });
                    this->expect_same_update(
                        dense, n, uplo,
                        [&](Tensor& out) { ops.hpr(x, out, uplo, K_ALPHA); },
                        [&](Tensor& out) { ops.her(x, out, uplo, K_ALPHA); });
                    this->expect_same_update(
                        dense, n, uplo,
                        [&](Tensor& out) { ops.spr2(x, y, out, uplo, K_ALPHA); },
                        [&](Tensor& out) { ops.syr2(x, y, out, uplo, K_ALPHA); });
                    this->expect_same_update(
                        dense, n, uplo,
                        [&](Tensor& out) { ops.hpr2(x, y, out, uplo, K_ALPHA); },
                        [&](Tensor& out) { ops.her2(x, y, out, uplo, K_ALPHA); });
                }
            });
        }

        /**
         * @brief Tests that tpmv and tpsv give the elements of trmv and trsv, and the product
         *        and the solution formed in integers on the host.
         */
        TYPED_TEST(Unit_BlasPacked, APackedTriangularMatrixEqualsTheDenseSibling)
        {
            using T = TypeParam;
            const int64_t n = K_SIBLING_ORDER;

            this->for_each_engine([&](Core::Layout)
            {
                SushiBLAS::BLASOps ops(*this->engine);

                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    for (const Core::Transpose trans : K_TRANSPOSES)
                    {
                        for (const Core::Diag diag : K_DIAGONALS)
                        {
                            SCOPED_TRACE(flags_name(uplo, trans, diag));
                            const ExactTriangle exact =
                                exact_triangle(n, uplo, trans, diag, Core::is_complex_v<T>);
                            const std::vector<T> stored = narrowed_all<T>(exact.stored);
                            const std::vector<T> x = narrowed_all<T>(exact.x);
                            const std::vector<T> b = narrowed_all<T>(exact.b);

                            const Tensor AP = this->vector_of(this->packed(stored, n, uplo));
                            const Tensor full = this->dense_of(n, stored);

                            const std::vector<T> product = this->expect_same_result(
                                x,
                                [&](Tensor& out) { ops.tpmv(AP, out, uplo, trans, diag); },
                                [&](Tensor& out) { ops.trmv(full, out, uplo, trans, diag); });
                            expect_equal(product, b);

                            const std::vector<T> solution = this->expect_same_result(
                                b,
                                [&](Tensor& out) { ops.tpsv(AP, out, uplo, trans, diag); },
                                [&](Tensor& out) { ops.trsv(full, out, uplo, trans, diag); });
                            expect_equal(solution, x);
                        }
                    }
                }
            });
        }

        /**
         * @brief Tests that the eight routines record and run on a packed tensor of no
         *        elements, and give the hand values at orders 1 and 2.
         */
        TYPED_TEST(Unit_BlasPacked, EdgeOrders)
        {
            using T = TypeParam;
            using Inputs = typename TestFixture::PackedInputs;

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(std::string("order 0, ") + Core::to_string(uplo));
                    const Inputs empty{uplo, {}, {}, {}, {}, {}, {}, 1.0, 0.0};
                    EXPECT_NO_THROW((void)this->run_eight(empty));
                }

                for (const EdgeRow& row : K_EDGE_ROWS)
                {
                    SCOPED_TRACE(::testing::Message() << "order " << row.x.size() << ", "
                                                      << Core::to_string(row.uplo));
                    const Inputs in{row.uplo,
                                    elements<T>(row.ap),
                                    elements<T>(row.ap),
                                    elements<T>(row.x),
                                    elements<T>(row.y),
                                    std::vector<T>(row.x.size(), poison<T>()),
                                    elements<T>(row.b),
                                    1.0,
                                    0.0};
                    const std::vector<std::vector<T>> results = this->run_eight(in);

                    ASSERT_EQ(results.size(), K_ROUTINE_COUNT);
                    for (std::size_t routine = 0; routine < K_ROUTINE_COUNT; ++routine)
                    {
                        SCOPED_TRACE(K_ROUTINES[routine]);
                        expect_equal(results[routine], elements<T>(row.results[routine]));
                    }
                }
            });
        }

        /**
         * @brief Tests that the two kernels agree within ModeTolerance on the eight routines.
         */
        TYPED_TEST(Unit_BlasPacked, BothModesAgree)
        {
            using T = TypeParam;
            using Inputs = typename TestFixture::PackedInputs;
            const int64_t n = K_MODE_ORDER;

            std::mt19937 generator(K_SEED);
            const std::vector<T> general = random_elements<T>(generator, n * n);
            std::vector<T> dominant = random_elements<T>(generator, n * n);
            for (int64_t i = 0; i < n; ++i)
                dominant[static_cast<std::size_t>(i * n + i)] =
                    narrowed<T>(Wide(static_cast<double>(n + 1), 0.0));
            const std::vector<T> x = random_elements<T>(generator, n);
            const std::vector<T> y = random_elements<T>(generator, n);
            const std::vector<T> prior = random_elements<T>(generator, n);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(engine_name(layout, ExecutionMode::ForceSYCL) + ", " +
                                 Core::to_string(uplo));
                    const Inputs in{uplo,
                                    to_packed(general, n, uplo, layout),
                                    to_packed(dominant, n, uplo, layout),
                                    x,
                                    y,
                                    prior,
                                    x,
                                    K_ALPHA,
                                    K_BETA};

                    this->start(layout, ExecutionMode::ForceReference);
                    const std::vector<std::vector<T>> reference = this->run_eight(in);
                    this->start(layout, ExecutionMode::ForceSYCL);
                    const std::vector<std::vector<T>> device = this->run_eight(in);

                    ASSERT_EQ(device.size(), reference.size());
                    for (std::size_t routine = 0; routine < device.size(); ++routine)
                    {
                        SCOPED_TRACE(K_ROUTINES[routine]);
                        expect_near(device[routine], reference[routine]);
                    }
                }
            }
        }

        /**
         * @brief Tests that hpmv conjugates the mirrored triangle and reads no imaginary
         *        diagonal.
         */
        TYPED_TEST(Unit_BlasPackedComplex, HpmvConjugatesTheMirror)
        {
            using T = TypeParam;
            const int64_t n = K_HAND_ORDER;
            const std::vector<Wide> hermitian = {
                Wide(1, K_DIAGONAL_IMAGINARY), Wide(2, 1),  Wide(0, 0),
                Wide(2, -1), Wide(3, K_DIAGONAL_IMAGINARY), Wide(4, -2),
                Wide(0, 0),  Wide(4, 2), Wide(5, K_DIAGONAL_IMAGINARY)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1), Wide(2, 0)};

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor AP =
                        this->vector_of(this->packed(narrowed_all<T>(hermitian), n, uplo));
                    const Tensor x_tensor = this->vector_of(narrowed_all<T>(x));
                    Tensor y = this->vector_of(std::vector<T>(3, poison<T>()));

                    SushiBLAS::BLASOps(*this->engine).hpmv(AP, x_tensor, y, uplo);
                    this->engine->execute().value();

                    expect_equal(read_tensor<T>(y),
                                 narrowed_all<T>({Wide(0, 2), Wide(10, -2), Wide(8, 4)}));
                }
            });
        }

        /**
         * @brief Tests that hpr adds 2 * x * x^H to the stored triangle and leaves a diagonal
         *        whose imaginary parts are exactly zero.
         */
        TYPED_TEST(Unit_BlasPackedComplex, HprKeepsTheDiagonalReal)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor x = this->vector_of(narrowed_all<T>({Wide(1, 1), Wide(2, 0)}));
                    Tensor AP = this->vector_of(
                        narrowed_all<T>({Wide(0, 7), Wide(0, 0), Wide(0, 7)}));

                    SushiBLAS::BLASOps(*this->engine).hpr(x, AP, uplo, 2.0);
                    this->engine->execute().value();

                    const double sign = (uplo == Core::Uplo::Upper) ? 1.0 : -1.0;
                    expect_equal(read_tensor<T>(AP),
                                 narrowed_all<T>({Wide(4, 0), Wide(4, 4 * sign), Wide(8, 0)}));
                }
            });
        }

        /**
         * @brief Tests that hpr2 multiplies y * x^H by the conjugate of a complex alpha.
         */
        TYPED_TEST(Unit_BlasPackedComplex, Hpr2ConjugatesAlpha)
        {
            using T = TypeParam;

            this->for_each_engine([&](Core::Layout)
            {
                for (const Core::Uplo uplo : K_TRIANGLES)
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    const Tensor x = this->vector_of(narrowed_all<T>({Wide(1, 0), Wide(0, 1)}));
                    const Tensor y = this->vector_of(narrowed_all<T>({Wide(1, 1), Wide(2, 0)}));
                    Tensor AP = this->vector_of(std::vector<T>(3));

                    SushiBLAS::BLASOps(*this->engine).hpr2(x, y, AP, uplo, Wide(0.0, 1.0));
                    this->engine->execute().value();

                    const double sign = (uplo == Core::Uplo::Upper) ? 1.0 : -1.0;
                    expect_equal(read_tensor<T>(AP),
                                 narrowed_all<T>({Wide(2, 0), Wide(-1, sign), Wide(-4, 0)}));
                }
            });
        }

        /**
         * @brief Tests that spmv and tpmv sum a HALF row in FLOAT32: 2048 + 1 + 1 - 2048 is 2
         *        there and 0 in HALF.
         */
        TEST_F(Unit_BlasPackedHalf, HalfSumsInFloat)
        {
            using T = sycl::half;
            constexpr int64_t n = 4;
            const std::vector<double> row = {2048, 1, 1, -2048};
            std::vector<double> square(static_cast<std::size_t>(n * n), 0.0);
            std::copy(row.begin(), row.end(), square.begin());

            for_each_engine([&](Core::Layout)
            {
                SushiBLAS::BLASOps ops(*engine);
                const Tensor AP = vector_of(packed(elements<T>(square), n, Core::Uplo::Upper));
                const Tensor ones = vector_of(elements<T>({1, 1, 1, 1}));

                Tensor from_spmv = vector_of(std::vector<T>(4, poison<T>()));
                ops.spmv(AP, ones, from_spmv, Core::Uplo::Upper);
                Tensor from_tpmv = vector_of(elements<T>({1, 1, 1, 1}));
                ops.tpmv(AP, from_tpmv, Core::Uplo::Upper);
                engine->execute().value();

                expect_equal(read_tensor<T>(from_spmv), elements<T>({2, 1, 1, -2048}));
                expect_equal(read_tensor<T>(from_tpmv), elements<T>({2, 0, 0, 0}));
            });
        }

        /**
         * @brief Tests that tpsv carries a HALF solution in FLOAT32: the middle element is
         *        2049 there, stored as 2048, and the last is 2050 - 2049.
         */
        TEST_F(Unit_BlasPackedHalf, HalfSolveCarriesTheSolutionInFloat)
        {
            using T = sycl::half;
            const std::vector<double> lower = {1, 0, 0, -1, 1, 0, 0, 1, 1};

            for_each_engine([&](Core::Layout)
            {
                const Tensor AP =
                    vector_of(packed(elements<T>(lower), K_HAND_ORDER, Core::Uplo::Lower));
                Tensor b = vector_of(elements<T>({2048, 1, 2050}));

                SushiBLAS::BLASOps(*engine).tpsv(AP, b, Core::Uplo::Lower);
                engine->execute().value();

                expect_equal(read_tensor<T>(b), elements<T>({2048, 2048, 1}));
            });
        }

        /**
         * @brief Tests spmv against symv at an order where a device that plans a work-group
         *        per row runs one.
         */
        TEST_F(Unit_BlasPackedLarge, APackedTriangleEqualsTheDenseSiblingLarge)
        {
            for_each_engine([&](Core::Layout)
            {
                const std::array<std::vector<float>, 2> symmetric = large_symmetric_products();
                expect_equal(symmetric[0], symmetric[1]);
            });
        }

        /**
         * @brief Tests that spmv gives the same bits when the device runs it twice.
         */
        TEST_F(Unit_BlasPackedLarge, RepeatsToTheBit)
        {
            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(engine_name(layout, ExecutionMode::ForceSYCL));
                start(layout, ExecutionMode::ForceSYCL);
                const std::vector<float> first = large_symmetric_products()[0];
                const std::vector<float> second = large_symmetric_products()[0];

                ASSERT_EQ(first.size(), second.size());
                EXPECT_EQ(0, std::memcmp(first.data(), second.data(),
                                         first.size() * sizeof(float)));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
