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
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"

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
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceReference,
                                                             ExecutionMode::ForceSYCL};

            /// @brief The kernel the cases that compare the device with another result run.
            const std::vector<ExecutionMode> K_DEVICE_MODE = {ExecutionMode::ForceSYCL};

            /// @brief The five element types the packed routines accept.
            const std::vector<Core::DataType> K_ALL_TYPES = {
                Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64,
                Core::DataType::Complex32, Core::DataType::Complex64};

            /// @brief The complex element types, the only ones a conjugation changes.
            const std::vector<Core::DataType> K_COMPLEX_TYPES = {Core::DataType::Complex32,
                                                                 Core::DataType::Complex64};

            /// @brief The element type of the HALF accumulation cases.
            const std::vector<Core::DataType> K_HALF_TYPE = {Core::DataType::Half};

            /// @brief The element type of the cases of order K_LARGE_ORDER.
            const std::vector<Core::DataType> K_FLOAT_TYPE = {Core::DataType::Float32};

            /**
             * @brief Magnitude of one product of two entries drawn from [-1, 1] in both parts:
             *        the bound of |a| * |b| for complex entries of modulus at most 2^(1/2).
             */
            constexpr double K_ENTRY_PRODUCT_MAGNITUDE = 2.0;

            /**
             * @brief Ratio of error to rounding bound above which two kernels disagree: the
             *        reference-tester threshold 16 times 4 for the growth of a tpsv solve whose
             *        diagonal n + 1 dominates n off-diagonals of at most 1.
             */
            constexpr double K_AGREEMENT_THRESHOLD = 64.0;

            /// @brief Unit roundoff of HALF, which has no std::numeric_limits entry: 2^-11.
            constexpr double K_HALF_UNIT_ROUNDOFF = 0x1p-11;

            /// @brief Rounded operations on the path to a HALF result: the sum is carried in FLOAT32.
            constexpr std::size_t K_HALF_ROUNDINGS = 2;

            /// @brief Both triangles a packed tensor can hold.
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

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
             * @brief The hand values of orders 0, 1 and 2; [1, 2, 3] packs [1 2; 2 3] in every
             *        layout and triangle.
             */
            const EdgeRow K_EDGE_ROWS[] = {
                {Core::Uplo::Upper, {}, {}, {}, {},
                 {{}, {}, {}, {}, {}, {}, {}, {}}},
                {Core::Uplo::Lower, {}, {}, {}, {},
                 {{}, {}, {}, {}, {}, {}, {}, {}}},
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

            /** @brief Holds the flags of one call on a packed triangle. */
            struct TriangleFlags
            {
                Core::Uplo uplo;       /**< The triangle. */
                Core::Transpose trans; /**< The form of op(A). */
                Core::Diag diag;       /**< How the diagonal is read. */
            };

            /** @brief Holds a routine and the triangle it is called on. */
            struct RoutineTriangle
            {
                std::size_t routine; /**< Position of the routine in K_ROUTINES. */
                Core::Uplo uplo;     /**< The triangle every packed operand holds. */
            };

            /** @brief Holds a routine and the edge row of K_EDGE_ROWS it is called on. */
            struct RoutineEdge
            {
                std::size_t routine; /**< Position of the routine in K_ROUTINES. */
                std::size_t edge;    /**< Position of the row in K_EDGE_ROWS. */
            };

            /** @brief Returns the hand triangular products as rows. */
            [[nodiscard]] std::vector<TriangularRow> triangular_rows()
            {
                return std::vector<TriangularRow>(std::begin(K_TRIANGULAR_ROWS),
                                                  std::end(K_TRIANGULAR_ROWS));
            }

            /** @brief Returns every combination of uplo, trans and diag. */
            [[nodiscard]] std::vector<TriangleFlags> all_triangle_flags()
            {
                std::vector<TriangleFlags> flags;
                for (const Core::Uplo uplo : K_TRIANGLES)
                    for (const Core::Transpose trans : K_TRANSPOSES)
                        for (const Core::Diag diag : K_DIAGONALS)
                            flags.push_back({uplo, trans, diag});
                return flags;
            }

            /** @brief Returns every routine on every triangle. */
            [[nodiscard]] std::vector<RoutineTriangle> all_routines_and_triangles()
            {
                std::vector<RoutineTriangle> rows;
                for (std::size_t routine = 0; routine < K_ROUTINE_COUNT; ++routine)
                    for (const Core::Uplo uplo : K_TRIANGLES)
                        rows.push_back({routine, uplo});
                return rows;
            }

            /** @brief Returns every routine on every edge row. */
            [[nodiscard]] std::vector<RoutineEdge> all_routines_and_edges()
            {
                std::vector<RoutineEdge> rows;
                for (std::size_t edge = 0; edge < std::size(K_EDGE_ROWS); ++edge)
                    for (std::size_t routine = 0; routine < K_ROUTINE_COUNT; ++routine)
                        rows.push_back({routine, edge});
                return rows;
            }

            /** @brief Returns the tail of a row named by the flags of a call. */
            [[nodiscard]] std::string axis_name(const TriangleFlags& flags)
            {
                return "_" + word(flags.uplo) + "_" + word(flags.trans) + "_" + word(flags.diag);
            }

            /** @brief Returns the tail of a row named by the flags of a triangular product. */
            [[nodiscard]] std::string axis_name(const TriangularRow& row)
            {
                return axis_name(TriangleFlags{row.uplo, row.trans, row.diag});
            }

            /** @brief Returns the tail of a row named by a routine and a triangle. */
            [[nodiscard]] std::string axis_name(const RoutineTriangle& row)
            {
                return std::string("_") + K_ROUTINES[row.routine] + Test::axis_name(row.uplo);
            }

            /** @brief Returns the tail of a row named by a routine and an edge row. */
            [[nodiscard]] std::string axis_name(const RoutineEdge& row)
            {
                return std::string("_") + K_ROUTINES[row.routine] + "_order" +
                       std::to_string(K_EDGE_ROWS[row.edge].x.size()) +
                       Test::axis_name(K_EDGE_ROWS[row.edge].uplo);
            }

            /** @brief Returns whether a triangle is packed in lines that grow under a layout. */
            [[nodiscard]] constexpr bool is_growing(Core::Uplo uplo, Core::Layout layout) noexcept
            {
                return (uplo == Core::Uplo::Upper) == (layout == Core::Layout::ColumnMajor);
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

            /** @brief Returns the unit roundoff of the component type R of an element type. */
            template<typename R>
            [[nodiscard]] constexpr double unit_roundoff_of()
            {
                if constexpr (std::is_same_v<R, sycl::half>)
                    return K_HALF_UNIT_ROUNDOFF;
                else
                    return TestSupport::unit_roundoff<R>();
            }

            /**
             * @brief Returns the rounded operations on the path to one result of two kernels
             *        that sum 2 * K_MODE_ORDER terms: the sum, and the carried solve.
             */
            template<typename R>
            [[nodiscard]] constexpr std::size_t agreement_roundings()
            {
                if constexpr (std::is_same_v<R, sycl::half>)
                    return K_HALF_ROUNDINGS;
                else
                    return static_cast<std::size_t>(2 * K_MODE_ORDER);
            }

            /**
             * @brief Expects both parts of every element within the rounding bound two kernels
             *        of the component type of T may differ by: rounded operations * unit
             *        roundoff * magnitude, measured through TestSupport::measure_error, the
             *        magnitude being the largest of 1 and the expected modulus times
             *        K_ENTRY_PRODUCT_MAGNITUDE.
             */
            template<typename T>
            void expect_near(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                using R = Kernels::RealType<T>;

                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    const Wide a = widened(actual[i]);
                    const Wide e = widened(expected[i]);
                    const double magnitude =
                        K_ENTRY_PRODUCT_MAGNITUDE * std::max(1.0, std::abs(e));
                    const double bound = static_cast<double>(agreement_roundings<R>()) *
                                         unit_roundoff_of<R>() * magnitude;

                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.real(), e.real(), bound, K_AGREEMENT_THRESHOLD)))
                        << "real part of element " << i;
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        a.imag(), e.imag(), bound, K_AGREEMENT_THRESHOLD)))
                        << "imaginary part of element " << i;
                }
            }

            /** @brief Holds the operands and the scalars of one call of each routine. */
            template<typename T>
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
        } // namespace

        /** @brief Holds a dense matrix, two vectors and a prior y drawn for a sibling comparison. */
        template<typename T>
        struct SiblingData
        {
            std::vector<T> dense; /**< The matrix, row by row. */
            std::vector<T> x;     /**< The first vector. */
            std::vector<T> y;     /**< The second vector of the rank-2 updates. */
            std::vector<T> prior; /**< What y of the products holds before. */
        };

        /**
         * @brief Builds packed, dense and vector operands of any element type and records the
         *        packed routines on them.
         */
        class BlasPackedCase : public SushiBLASTest
        {
            protected:
                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    layout_ = layout;
                    mode_ = mode;
                    this->reinit_engine(layout);
                    this->set_mode(mode);
                }

                /** @brief Returns a tensor of dims that holds stored in storage order. */
                template<typename T>
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
                template<typename T>
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    return tensor_of({static_cast<int64_t>(values.size())}, values);
                }

                /** @brief Returns an n x n tensor that holds the row-ordered values. */
                template<typename T>
                [[nodiscard]] Tensor dense_of(int64_t n, const std::vector<T>& row_ordered)
                {
                    return tensor_of({n, n},
                                     in_layout(row_ordered, n, n, this->engine->layout()));
                }

                /** @brief Returns the packed values of one triangle of a dense n x n matrix. */
                template<typename T>
                [[nodiscard]] std::vector<T> packed(const std::vector<T>& dense, int64_t n,
                                                    Core::Uplo uplo) const
                {
                    return to_packed(dense, n, uplo, this->engine->layout());
                }

                /** @brief Returns the packed tensor of K_HAND_GROWING or K_HAND_SHRINKING. */
                template<typename T>
                [[nodiscard]] Tensor hand_triangle(Core::Uplo uplo)
                {
                    return vector_of(elements<T>(is_growing(uplo, this->engine->layout())
                                                     ? K_HAND_GROWING
                                                     : K_HAND_SHRINKING));
                }

                /** @brief Returns the elements of a tensor, or none for one without storage. */
                template<typename T>
                [[nodiscard]] static std::vector<T> read(const Tensor& tensor)
                {
                    return tensor.num_elements == 0 ? std::vector<T>() : read_tensor<T>(tensor);
                }

                /** @brief Returns the elements of an n x n tensor, row by row. */
                template<typename T>
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

                /** @brief Returns the matrix and vectors of the comparisons with a dense sibling. */
                template<typename T>
                [[nodiscard]] static SiblingData<T> draw_siblings()
                {
                    const int64_t n = K_SIBLING_ORDER;
                    std::mt19937 generator(K_SEED);
                    SiblingData<T> data;
                    data.dense = random_elements<T>(generator, n * n);
                    data.x = random_elements<T>(generator, n);
                    data.y = random_elements<T>(generator, n);
                    data.prior = random_elements<T>(generator, n);
                    return data;
                }

                /**
                 * @brief Records two calls on copies of initial in one graph, runs it and expects
                 *        the same elements from both.
                 * @param first Callable `(Tensor&)` that records on the first copy.
                 * @param second Callable `(Tensor&)` that records on the second copy.
                 * @return What the first call left.
                 */
                template<typename T, typename First, typename Second>
                std::vector<T> expect_same_result(const std::vector<T>& initial, First&& first,
                                                  Second&& second)
                {
                    Tensor from_first = vector_of(initial);
                    Tensor from_second = vector_of(initial);
                    first(from_first);
                    second(from_second);
                    this->recording->run().value();

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
                template<typename T, typename Packed, typename Dense>
                void expect_same_update(const std::vector<T>& dense, int64_t n, Core::Uplo uplo,
                                        Packed&& packed_call, Dense&& dense_call)
                {
                    Tensor from_packed = vector_of(packed(dense, n, uplo));
                    Tensor from_dense = dense_of(n, dense);
                    packed_call(from_packed);
                    dense_call(from_dense);
                    this->recording->run().value();

                    expect_equal(read_tensor<T>(from_packed),
                                 packed(read_row_ordered<T>(from_dense, n), n, uplo));
                }

                /** @brief Returns what the result tensor of a routine holds before the call. */
                template<typename T>
                [[nodiscard]] static const std::vector<T>& initial_of(std::size_t routine,
                                                                      const PackedInputs<T>& in)
                {
                    if (routine == K_SPMV || routine == K_HPMV)
                        return in.prior;
                    if (routine == K_TPMV)
                        return in.x;
                    return routine == K_TPSV ? in.b : in.ap;
                }

                /**
                 * @brief Records one packed routine on the operands, runs it and returns what it
                 *        leaves in its result tensor.
                 * @param routine Position of the routine in K_ROUTINES.
                 */
                template<typename T>
                [[nodiscard]] std::vector<T> run_routine(std::size_t routine,
                                                         const PackedInputs<T>& in)
                {
                    const Tensor ap = vector_of(in.ap);
                    const Tensor triangular = vector_of(in.triangular);
                    const Tensor x = vector_of(in.x);
                    const Tensor y = vector_of(in.y);
                    Tensor result = vector_of(initial_of(routine, in));

                    SushiBLAS::BLASOps ops(this->recorder());
                    switch (routine)
                    {
                        case K_SPMV:
                            ops.spmv(ap, x, result, in.uplo, in.alpha, in.beta);
                            break;
                        case K_HPMV:
                            ops.hpmv(ap, x, result, in.uplo, in.alpha, in.beta);
                            break;
                        case K_TPMV:
                            ops.tpmv(triangular, result, in.uplo);
                            break;
                        case K_TPSV:
                            ops.tpsv(triangular, result, in.uplo);
                            break;
                        case K_SPR:
                            ops.spr(x, result, in.uplo, in.alpha);
                            break;
                        case K_HPR:
                            ops.hpr(x, result, in.uplo, in.alpha);
                            break;
                        case K_SPR2:
                            ops.spr2(x, y, result, in.uplo, in.alpha);
                            break;
                        case K_HPR2:
                            ops.hpr2(x, y, result, in.uplo, in.alpha);
                            break;
                        default:
                            ADD_FAILURE() << "no such packed routine";
                    }
                    this->recording->run().value();
                    return read<T>(result);
                }

                /**
                 * @brief Expects one routine to leave the same elements, within the rounding of
                 *        the element type, on the reference and on the kernel of the row, in the
                 *        layout of the row.
                 */
                template<typename T>
                void expect_modes_agree(std::size_t routine, const PackedInputs<T>& in)
                {
                    const Core::Layout layout = layout_;
                    const ExecutionMode device_mode = mode_;

                    start(layout, ExecutionMode::ForceReference);
                    const std::vector<T> reference = run_routine(routine, in);
                    start(layout, device_mode);
                    const std::vector<T> device = run_routine(routine, in);

                    expect_near(device, reference);
                }

                /**
                 * @brief Returns y of spmv on a packed upper triangle of order K_LARGE_ORDER,
                 *        then y of symv on the same matrix.
                 */
                template<typename T>
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

                    SushiBLAS::BLASOps ops(this->recorder());
                    ops.spmv(ap, x, from_packed, Core::Uplo::Upper, K_ALPHA, K_BETA);
                    ops.symv(full, x, from_dense, Core::Uplo::Upper, K_ALPHA, K_BETA);
                    this->recording->run().value();
                    return {{read_tensor<T>(from_packed), read_tensor<T>(from_dense)}};
                }

            private:
                /// @brief The layout the engine was last started in.
                Core::Layout layout_ = Core::Layout::RowMajor;

                /// @brief The kernel the engine was last started with.
                ExecutionMode mode_ = ExecutionMode::ForceReference;
        };

        /**
         * @brief Runs a case once per row of the BlasPackedCase suite.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        using PackedRowCase = BlasRowCase<BlasPackedCase, Axis>;

        /** @brief Runs the hand and sibling cases that take one triangle. */
        class Medium_BlasPackedTriangle : public PackedRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the hand products of the triangular routines. */
        class Medium_BlasPackedProduct : public PackedRowCase<TriangularRow>
        {

        };

        /** @brief Runs the triangular comparisons with the dense siblings over every flag combination. */
        class Medium_BlasPackedFlags : public PackedRowCase<TriangleFlags>
        {

        };

        /** @brief Runs each routine on the hand values of the orders 0, 1 and 2. */
        class Medium_BlasPackedEdge : public PackedRowCase<RoutineEdge>
        {

        };

        /** @brief Runs the comparison of the device with the reference for each routine and triangle. */
        class Medium_BlasPackedAgree : public PackedRowCase<RoutineTriangle>
        {

        };

        /** @brief Runs the conjugating packed cases over the complex element types. */
        class Medium_BlasPackedComplex : public PackedRowCase<Core::Uplo>
        {

        };

        /** @brief Runs the HALF accumulation cases. */
        class Medium_BlasPackedHalf : public PackedRowCase<NoAxis>
        {

        };

        /** @brief Runs the FLOAT32 cases of order K_LARGE_ORDER. */
        class Medium_BlasPackedLarge : public PackedRowCase<NoAxis>
        {

        };

        /** @brief Runs the FLOAT32 repeat case of order K_LARGE_ORDER on the device. */
        class Medium_BlasPackedRepeat : public PackedRowCase<NoAxis>
        {

        };

        /**
         * @brief Tests spmv on the literal packed arrays of [1 2 3; 2 4 5; 3 5 6].
         */
        TEST_P(Medium_BlasPackedTriangle, SpmvMatchesByHand)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor AP = hand_triangle<T>(uplo);
                const Tensor x = vector_of(elements<T>({1, -1, 2}));
                Tensor y = vector_of(std::vector<T>(3, poison<T>()));

                SushiBLAS::BLASOps(recorder()).spmv(AP, x, y, uplo);
                recording->run().value();

                expect_equal(read_tensor<T>(y), elements<T>({5, 8, 10}));
            });
        }

        /**
         * @brief Tests that spr writes 2 * x * x^T in the order the layout and triangle give.
         */
        TEST_P(Medium_BlasPackedTriangle, SprWritesTheStoredTriangle)
        {
            const Core::Uplo uplo = GetParam().axis;
            const Core::Layout layout = GetParam().layout;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor x = vector_of(elements<T>({1, 2, 3}));
                Tensor AP = vector_of(elements<T>({0, 0, 0, 0, 0, 0}));

                SushiBLAS::BLASOps(recorder()).spr(x, AP, uplo, 2.0);
                recording->run().value();

                expect_equal(read_tensor<T>(AP),
                             is_growing(uplo, layout) ? elements<T>({2, 4, 8, 6, 12, 18})
                                                      : elements<T>({2, 4, 6, 8, 12, 18}));
            });
        }

        /**
         * @brief Tests spr2 on x = [1, 2], y = [3, 5]: the triangle of [6 11; 11 20].
         */
        TEST_P(Medium_BlasPackedTriangle, Spr2MatchesByHand)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor x = vector_of(elements<T>({1, 2}));
                const Tensor y = vector_of(elements<T>({3, 5}));
                Tensor AP = vector_of(elements<T>({0, 0, 0}));

                SushiBLAS::BLASOps(recorder()).spr2(x, y, AP, uplo, 1.0);
                recording->run().value();

                expect_equal(read_tensor<T>(AP), elements<T>({6, 11, 20}));
            });
        }

        /** @brief Tests that spmv gives the elements of symv on the same matrix. */
        TEST_P(Medium_BlasPackedTriangle, SpmvEqualsSymv)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor AP = vector_of(packed(data.dense, K_SIBLING_ORDER, uplo));
                const Tensor full = dense_of(K_SIBLING_ORDER, data.dense);
                const Tensor x = vector_of(data.x);

                expect_same_result(
                    data.prior,
                    [&](Tensor& out) { ops.spmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                    [&](Tensor& out) { ops.symv(full, x, out, uplo, K_ALPHA, K_BETA); });
            });
        }

        /** @brief Tests that hpmv gives the elements of hemv on the same matrix. */
        TEST_P(Medium_BlasPackedTriangle, HpmvEqualsHemv)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor AP = vector_of(packed(data.dense, K_SIBLING_ORDER, uplo));
                const Tensor full = dense_of(K_SIBLING_ORDER, data.dense);
                const Tensor x = vector_of(data.x);

                expect_same_result(
                    data.prior,
                    [&](Tensor& out) { ops.hpmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                    [&](Tensor& out) { ops.hemv(full, x, out, uplo, K_ALPHA, K_BETA); });
            });
        }

        /** @brief Tests that spr leaves the stored triangle of syr. */
        TEST_P(Medium_BlasPackedTriangle, SprEqualsSyr)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor x = vector_of(data.x);

                expect_same_update(
                    data.dense, K_SIBLING_ORDER, uplo,
                    [&](Tensor& out) { ops.spr(x, out, uplo, K_ALPHA); },
                    [&](Tensor& out) { ops.syr(x, out, uplo, K_ALPHA); });
            });
        }

        /** @brief Tests that hpr leaves the stored triangle of her. */
        TEST_P(Medium_BlasPackedTriangle, HprEqualsHer)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor x = vector_of(data.x);

                expect_same_update(
                    data.dense, K_SIBLING_ORDER, uplo,
                    [&](Tensor& out) { ops.hpr(x, out, uplo, K_ALPHA); },
                    [&](Tensor& out) { ops.her(x, out, uplo, K_ALPHA); });
            });
        }

        /** @brief Tests that spr2 leaves the stored triangle of syr2. */
        TEST_P(Medium_BlasPackedTriangle, Spr2EqualsSyr2)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor x = vector_of(data.x);
                const Tensor y = vector_of(data.y);

                expect_same_update(
                    data.dense, K_SIBLING_ORDER, uplo,
                    [&](Tensor& out) { ops.spr2(x, y, out, uplo, K_ALPHA); },
                    [&](Tensor& out) { ops.syr2(x, y, out, uplo, K_ALPHA); });
            });
        }

        /** @brief Tests that hpr2 leaves the stored triangle of her2. */
        TEST_P(Medium_BlasPackedTriangle, Hpr2EqualsHer2)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const SiblingData<T> data = draw_siblings<T>();
                SushiBLAS::BLASOps ops(recorder());
                const Tensor x = vector_of(data.x);
                const Tensor y = vector_of(data.y);

                expect_same_update(
                    data.dense, K_SIBLING_ORDER, uplo,
                    [&](Tensor& out) { ops.hpr2(x, y, out, uplo, K_ALPHA); },
                    [&](Tensor& out) { ops.her2(x, y, out, uplo, K_ALPHA); });
            });
        }

        /**
         * @brief Tests tpmv on the literal packed arrays of U and of its transpose.
         */
        TEST_P(Medium_BlasPackedProduct, TpmvMatchesByHand)
        {
            const TriangularRow row = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor AP = hand_triangle<T>(row.uplo);
                Tensor x = vector_of(elements<T>({1, 2, 3}));

                SushiBLAS::BLASOps(recorder()).tpmv(AP, x, row.uplo, row.trans, row.diag);
                recording->run().value();

                expect_equal(read_tensor<T>(x),
                             elements<T>({row.product[0], row.product[1], row.product[2]}));
            });
        }

        /**
         * @brief Tests that tpsv recovers [1, 2, 3] from each product of TpmvMatchesByHand.
         */
        TEST_P(Medium_BlasPackedProduct, TpsvInvertsTpmv)
        {
            const TriangularRow row = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor AP = hand_triangle<T>(row.uplo);
                Tensor b = vector_of(
                    elements<T>({row.product[0], row.product[1], row.product[2]}));

                SushiBLAS::BLASOps(recorder()).tpsv(AP, b, row.uplo, row.trans, row.diag);
                recording->run().value();

                expect_equal(read_tensor<T>(b), elements<T>({1, 2, 3}));
            });
        }

        /**
         * @brief Tests that tpmv gives the elements of trmv, and the product formed in integers
         *        on the host.
         */
        TEST_P(Medium_BlasPackedFlags, TpmvEqualsTrmvAndTheHostProduct)
        {
            const TriangleFlags flags = GetParam().axis;
            const int64_t n = K_SIBLING_ORDER;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                SushiBLAS::BLASOps ops(recorder());
                const ExactTriangle exact =
                    exact_triangle(n, flags.uplo, flags.trans, flags.diag, Core::is_complex_v<T>);
                const std::vector<T> stored = narrowed_all<T>(exact.stored);
                const Tensor AP = vector_of(packed(stored, n, flags.uplo));
                const Tensor full = dense_of(n, stored);

                const std::vector<T> product = expect_same_result(
                    narrowed_all<T>(exact.x),
                    [&](Tensor& out) { ops.tpmv(AP, out, flags.uplo, flags.trans, flags.diag); },
                    [&](Tensor& out) { ops.trmv(full, out, flags.uplo, flags.trans, flags.diag); });
                expect_equal(product, narrowed_all<T>(exact.b));
            });
        }

        /**
         * @brief Tests that tpsv gives the elements of trsv, and the solution formed in integers
         *        on the host.
         */
        TEST_P(Medium_BlasPackedFlags, TpsvEqualsTrsvAndTheHostSolution)
        {
            const TriangleFlags flags = GetParam().axis;
            const int64_t n = K_SIBLING_ORDER;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                SushiBLAS::BLASOps ops(recorder());
                const ExactTriangle exact =
                    exact_triangle(n, flags.uplo, flags.trans, flags.diag, Core::is_complex_v<T>);
                const std::vector<T> stored = narrowed_all<T>(exact.stored);
                const Tensor AP = vector_of(packed(stored, n, flags.uplo));
                const Tensor full = dense_of(n, stored);

                const std::vector<T> solution = expect_same_result(
                    narrowed_all<T>(exact.b),
                    [&](Tensor& out) { ops.tpsv(AP, out, flags.uplo, flags.trans, flags.diag); },
                    [&](Tensor& out) { ops.trsv(full, out, flags.uplo, flags.trans, flags.diag); });
                expect_equal(solution, narrowed_all<T>(exact.x));
            });
        }

        /**
         * @brief Tests that one routine records and runs on a packed tensor of no elements, and
         *        gives the hand values at orders 1 and 2.
         */
        TEST_P(Medium_BlasPackedEdge, RoutineGivesTheHandValues)
        {
            const RoutineEdge row = GetParam().axis;
            const EdgeRow& edge = K_EDGE_ROWS[row.edge];
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const PackedInputs<T> in{edge.uplo,
                                         elements<T>(edge.ap),
                                         elements<T>(edge.ap),
                                         elements<T>(edge.x),
                                         elements<T>(edge.y),
                                         std::vector<T>(edge.x.size(), poison<T>()),
                                         elements<T>(edge.b),
                                         1.0,
                                         0.0};

                expect_equal(run_routine(row.routine, in),
                             elements<T>(edge.results[row.routine]));
            });
        }

        /**
         * @brief Tests that the kernel of the row agrees with the reference, within the rounding
         *        of the element type, on one routine and triangle.
         */
        TEST_P(Medium_BlasPackedAgree, RoutineMatchesTheReference)
        {
            const RoutineTriangle row = GetParam().axis;
            const int64_t n = K_MODE_ORDER;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Core::Layout layout = GetParam().layout;
                std::mt19937 generator(K_SEED);
                const std::vector<T> general = random_elements<T>(generator, n * n);
                std::vector<T> dominant = random_elements<T>(generator, n * n);
                for (int64_t i = 0; i < n; ++i)
                    dominant[static_cast<std::size_t>(i * n + i)] =
                        narrowed<T>(Wide(static_cast<double>(n + 1), 0.0));
                const std::vector<T> x = random_elements<T>(generator, n);
                const std::vector<T> y = random_elements<T>(generator, n);
                const std::vector<T> prior = random_elements<T>(generator, n);

                const PackedInputs<T> in{row.uplo,
                                         to_packed(general, n, row.uplo, layout),
                                         to_packed(dominant, n, row.uplo, layout),
                                         x,
                                         y,
                                         prior,
                                         x,
                                         K_ALPHA,
                                         K_BETA};

                expect_modes_agree(row.routine, in);
            });
        }

        /**
         * @brief Tests that hpmv conjugates the mirrored triangle and reads no imaginary
         *        diagonal.
         */
        TEST_P(Medium_BlasPackedComplex, HpmvConjugatesTheMirror)
        {
            const Core::Uplo uplo = GetParam().axis;
            const int64_t n = K_HAND_ORDER;
            const std::vector<Wide> hermitian = {
                Wide(1, K_DIAGONAL_IMAGINARY), Wide(2, 1),  Wide(0, 0),
                Wide(2, -1), Wide(3, K_DIAGONAL_IMAGINARY), Wide(4, -2),
                Wide(0, 0),  Wide(4, 2), Wide(5, K_DIAGONAL_IMAGINARY)};
            const std::vector<Wide> x = {Wide(1, 0), Wide(0, 1), Wide(2, 0)};
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor AP = vector_of(packed(narrowed_all<T>(hermitian), n, uplo));
                const Tensor x_tensor = vector_of(narrowed_all<T>(x));
                Tensor y = vector_of(std::vector<T>(3, poison<T>()));

                SushiBLAS::BLASOps(recorder()).hpmv(AP, x_tensor, y, uplo);
                recording->run().value();

                expect_equal(read_tensor<T>(y),
                             narrowed_all<T>({Wide(0, 2), Wide(10, -2), Wide(8, 4)}));
            });
        }

        /**
         * @brief Tests that hpr adds 2 * x * x^H to the stored triangle and leaves a diagonal
         *        whose imaginary parts are exactly zero.
         */
        TEST_P(Medium_BlasPackedComplex, HprKeepsTheDiagonalReal)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor x = vector_of(narrowed_all<T>({Wide(1, 1), Wide(2, 0)}));
                Tensor AP = vector_of(narrowed_all<T>({Wide(0, 7), Wide(0, 0), Wide(0, 7)}));

                SushiBLAS::BLASOps(recorder()).hpr(x, AP, uplo, 2.0);
                recording->run().value();

                const double sign = (uplo == Core::Uplo::Upper) ? 1.0 : -1.0;
                expect_equal(read_tensor<T>(AP),
                             narrowed_all<T>({Wide(4, 0), Wide(4, 4 * sign), Wide(8, 0)}));
            });
        }

        /**
         * @brief Tests that hpr2 multiplies y * x^H by the conjugate of a complex alpha.
         */
        TEST_P(Medium_BlasPackedComplex, Hpr2ConjugatesAlpha)
        {
            const Core::Uplo uplo = GetParam().axis;
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                const Tensor x = vector_of(narrowed_all<T>({Wide(1, 0), Wide(0, 1)}));
                const Tensor y = vector_of(narrowed_all<T>({Wide(1, 1), Wide(2, 0)}));
                Tensor AP = vector_of(std::vector<T>(3));

                SushiBLAS::BLASOps(recorder()).hpr2(x, y, AP, uplo, Wide(0.0, 1.0));
                recording->run().value();

                const double sign = (uplo == Core::Uplo::Upper) ? 1.0 : -1.0;
                expect_equal(read_tensor<T>(AP),
                             narrowed_all<T>({Wide(2, 0), Wide(-1, sign), Wide(-4, 0)}));
            });
        }

        /**
         * @brief Tests that spmv and tpmv sum a HALF row in FLOAT32: 2048 + 1 + 1 - 2048 is 2
         *        there and 0 in HALF.
         */
        TEST_P(Medium_BlasPackedHalf, HalfSumsInFloat)
        {
            using T = sycl::half;
            constexpr int64_t n = 4;
            const std::vector<double> row = {2048, 1, 1, -2048};
            std::vector<double> square(static_cast<std::size_t>(n * n), 0.0);
            std::copy(row.begin(), row.end(), square.begin());

            SushiBLAS::BLASOps ops(recorder());
            const Tensor AP = vector_of(packed(elements<T>(square), n, Core::Uplo::Upper));
            const Tensor ones = vector_of(elements<T>({1, 1, 1, 1}));

            Tensor from_spmv = vector_of(std::vector<T>(4, poison<T>()));
            ops.spmv(AP, ones, from_spmv, Core::Uplo::Upper);
            Tensor from_tpmv = vector_of(elements<T>({1, 1, 1, 1}));
            ops.tpmv(AP, from_tpmv, Core::Uplo::Upper);
            recording->run().value();

            expect_equal(read_tensor<T>(from_spmv), elements<T>({2, 1, 1, -2048}));
            expect_equal(read_tensor<T>(from_tpmv), elements<T>({2, 0, 0, 0}));
        }

        /**
         * @brief Tests that tpsv carries a HALF solution in FLOAT32: the middle element is
         *        2049 there, stored as 2048, and the last is 2050 - 2049.
         */
        TEST_P(Medium_BlasPackedHalf, HalfSolveCarriesTheSolutionInFloat)
        {
            using T = sycl::half;
            const std::vector<double> lower = {1, 0, 0, -1, 1, 0, 0, 1, 1};

            const Tensor AP =
                vector_of(packed(elements<T>(lower), K_HAND_ORDER, Core::Uplo::Lower));
            Tensor b = vector_of(elements<T>({2048, 1, 2050}));

            SushiBLAS::BLASOps(recorder()).tpsv(AP, b, Core::Uplo::Lower);
            recording->run().value();

            expect_equal(read_tensor<T>(b), elements<T>({2048, 2048, 1}));
        }

        /**
         * @brief Tests spmv against symv at an order where a device that plans a work-group
         *        per row runs one.
         */
        TEST_P(Medium_BlasPackedLarge, APackedTriangleEqualsTheDenseSiblingLarge)
        {
            const std::array<std::vector<float>, 2> symmetric = large_symmetric_products<float>();
            expect_equal(symmetric[0], symmetric[1]);
        }

        /**
         * @brief Tests that spmv gives the same bits when the device runs it twice.
         */
        TEST_P(Medium_BlasPackedRepeat, RepeatsToTheBit)
        {
            const std::vector<float> first = large_symmetric_products<float>()[0];
            const std::vector<float> second = large_symmetric_products<float>()[0];

            ASSERT_EQ(first.size(), second.size());
            EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(float)));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedTriangle,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedProduct,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, triangular_rows())),
            row_name<TriangularRow>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedFlags,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, all_triangle_flags())),
            row_name<TriangleFlags>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedEdge,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_BOTH_MODES, all_routines_and_edges())),
            row_name<RoutineEdge>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedAgree,
            ::testing::ValuesIn(rows_of(K_ALL_TYPES, K_DEVICE_MODE, all_routines_and_triangles())),
            row_name<RoutineTriangle>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedComplex,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedHalf, ::testing::ValuesIn(plain_rows(K_HALF_TYPE, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedLarge, ::testing::ValuesIn(plain_rows(K_FLOAT_TYPE, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasPackedRepeat,
            ::testing::ValuesIn(plain_rows(K_FLOAT_TYPE, K_DEVICE_MODE)), row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
