/**************************************************************************/
/* test_blas_structured_validation.cpp                                    */
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
 * @file test_blas_structured_validation.cpp
 * @brief Tests what the band, packed and rotation routines refuse at record time, and the
 *        forwards of the Hermitian routines on a real tensor.
 * @author Mustafa Garip
 */

#include <array>
#include <random>
#include <iterator>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::DataType;

            /// @brief Seed of the random operands of the forward cases.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Order of every matrix and length of every vector a refusal starts from.
            constexpr int64_t K_ORDER = 4;

            /// @brief Off-diagonals on each side of the diagonal in every band matrix.
            constexpr int64_t K_WIDTH = 1;

            /// @brief Elements per stored line of the general band matrix: kl + ku + 1.
            constexpr int64_t K_GENERAL_LD = 2 * K_WIDTH + 1;

            /// @brief Elements per stored line of the triangle band matrix: k + 1.
            constexpr int64_t K_TRIANGLE_LD = K_WIDTH + 1;

            /// @brief Elements of the packed triangle of order K_ORDER.
            constexpr int64_t K_PACKED_COUNT = K_ORDER * (K_ORDER + 1) / 2;

            /// @brief Elements of a rotm or rotmg parameter tensor.
            constexpr int64_t K_PARAMETERS = 5;

            /// @brief Scales the product or the update in the forward cases.
            constexpr double K_ALPHA = 1.5;

            /// @brief Scales the prior y in the forward cases.
            constexpr double K_BETA = -0.5;

            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BLASScalar K_COMPLEX_SCALAR = std::complex<double>(1.0, 2.0);

            /// @brief Both kernels an operation has.
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceReference,
                                                             ExecutionMode::ForceSYCL};

            /// @brief The one element type every refusal starts from.
            const std::vector<DataType> K_FLOAT32 = {DataType::Float32};

            /// @brief Both triangles a band or a packed matrix can hold.
            const std::vector<Core::Uplo> K_TRIANGLES = {Core::Uplo::Upper, Core::Uplo::Lower};

            /** @brief Names the storage scheme of the matrix a routine takes. */
            enum class Scheme : std::uint8_t
            {
                GeneralBand,  /**< A band with kl sub-diagonals and ku super-diagonals. */
                TriangleBand, /**< One triangle of a band with k off-diagonals. */
                Packed         /**< One packed triangle. */
            };

            /** @brief Holds every argument a band or a packed routine can take. */
            struct Operands
            {
                Tensor general;                                    /**< The general band matrix. */
                Tensor triangle;                                   /**< The triangle band matrix. */
                Tensor packed;                                     /**< The packed matrix. */
                Tensor x;                                          /**< The first vector. */
                Tensor y;                                          /**< The second vector. */
                int64_t kl = K_WIDTH;                              /**< Lower width of general. */
                int64_t ku = K_WIDTH;                              /**< Upper width of general. */
                int64_t k = K_WIDTH;                               /**< Width of triangle. */
                Core::Uplo uplo = Core::Uplo::Upper;               /**< The stored triangle. */
                Core::Transpose trans = Core::Transpose::NoTrans; /**< The form of op(A). */
                Core::Diag diag = Core::Diag::NonUnit;            /**< How the diagonal is read. */
                BLASScalar alpha = 1.0;                            /**< Scales the product. */
                BLASScalar beta = 1.0;                             /**< Scales the prior y. */
            };

            /** @brief Holds one public routine and which of the operands it takes. */
            struct Routine
            {
                const char* name;                          /**< The routine as a call spells it. */
                TaskHandle (*record)(BLASOps&, Operands&); /**< Records it on the operands. */
                Scheme scheme;                             /**< The storage of its matrix. */
                bool represents_its_kernels;               /**< False for a Hermitian sibling. */
                bool takes_y;                              /**< Whether it takes y. */
                bool takes_uplo;                           /**< Whether it takes a Uplo. */
                bool takes_trans;                          /**< Whether it takes a Transpose. */
                bool takes_diag;                           /**< Whether it takes a Diag. */
                bool takes_alpha;                          /**< Whether alpha is a BLASScalar. */
                bool takes_beta;                           /**< Whether beta is a BLASScalar. */
            };

            /// @brief The thirteen band and packed routines.
            const Routine K_ROUTINES[] = {
                {"gbmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.gbmv(o.general, o.x, o.y, o.kl, o.ku, o.trans, o.alpha, o.beta);
                 },
                 Scheme::GeneralBand, true, true, false, true, false, true, true},
                {"sbmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.sbmv(o.triangle, o.x, o.y, o.uplo, o.k, o.alpha, o.beta);
                 },
                 Scheme::TriangleBand, true, true, true, false, false, true, true},
                {"hbmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.hbmv(o.triangle, o.x, o.y, o.uplo, o.k, o.alpha, o.beta);
                 },
                 Scheme::TriangleBand, false, true, true, false, false, true, true},
                {"tbmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.tbmv(o.triangle, o.x, o.uplo, o.k, o.trans, o.diag);
                 },
                 Scheme::TriangleBand, true, false, true, true, true, false, false},
                {"tbsv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.tbsv(o.triangle, o.x, o.uplo, o.k, o.trans, o.diag);
                 },
                 Scheme::TriangleBand, true, false, true, true, true, false, false},
                {"spmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.spmv(o.packed, o.x, o.y, o.uplo, o.alpha, o.beta);
                 },
                 Scheme::Packed, true, true, true, false, false, true, true},
                {"hpmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.hpmv(o.packed, o.x, o.y, o.uplo, o.alpha, o.beta);
                 },
                 Scheme::Packed, false, true, true, false, false, true, true},
                {"tpmv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.tpmv(o.packed, o.x, o.uplo, o.trans, o.diag);
                 },
                 Scheme::Packed, true, false, true, true, true, false, false},
                {"tpsv",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.tpsv(o.packed, o.x, o.uplo, o.trans, o.diag);
                 },
                 Scheme::Packed, true, false, true, true, true, false, false},
                {"spr",
                 [](BLASOps& ops, Operands& o) { return ops.spr(o.x, o.packed, o.uplo, o.alpha); },
                 Scheme::Packed, true, false, true, false, false, true, false},
                {"hpr",
                 [](BLASOps& ops, Operands& o) { return ops.hpr(o.x, o.packed, o.uplo, K_ALPHA); },
                 Scheme::Packed, false, false, true, false, false, false, false},
                {"spr2",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.spr2(o.x, o.y, o.packed, o.uplo, o.alpha);
                 },
                 Scheme::Packed, true, true, true, false, false, true, false},
                {"hpr2",
                 [](BLASOps& ops, Operands& o)
                 {
                     return ops.hpr2(o.x, o.y, o.packed, o.uplo, o.alpha);
                 },
                 Scheme::Packed, false, true, true, false, false, true, false}};

            /** @brief Names a routine of K_ROUTINES as the one axis a row varies. */
            struct RoutineAxis
            {
                std::size_t index; /**< The position of the routine in K_ROUTINES. */
            };

            /** @brief Returns the tail of a row named by a routine. */
            [[nodiscard]] std::string axis_name(const RoutineAxis& axis)
            {
                return std::string("_") + K_ROUTINES[axis.index].name;
            }

            /** @brief Returns whether any routine is selected. */
            [[nodiscard]] bool is_any(const Routine&) noexcept
            {
                return true;
            }

            /** @brief Returns whether a routine is the one checked for the geometry of a pair. */
            [[nodiscard]] bool represents_its_kernels(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels;
            }

            /** @brief Returns whether a routine is checked for geometry and takes y. */
            [[nodiscard]] bool takes_y_of_a_kernel(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels && routine.takes_y;
            }

            /** @brief Returns whether a routine is checked and its vectors have a fixed order. */
            [[nodiscard]] bool has_a_fixed_order(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels && routine.scheme != Scheme::GeneralBand;
            }

            /** @brief Returns whether a routine has a fixed order and takes y. */
            [[nodiscard]] bool has_a_fixed_order_and_y(const Routine& routine) noexcept
            {
                return has_a_fixed_order(routine) && routine.takes_y;
            }

            /** @brief Returns whether a routine takes the general band matrix. */
            [[nodiscard]] bool takes_a_general_band(const Routine& routine) noexcept
            {
                return routine.scheme == Scheme::GeneralBand;
            }

            /** @brief Returns whether a routine is checked for geometry and takes a band. */
            [[nodiscard]] bool takes_a_band(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels && routine.scheme != Scheme::Packed;
            }

            /** @brief Returns whether a routine is checked for geometry and takes a packed AP. */
            [[nodiscard]] bool takes_a_packed_matrix(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels && routine.scheme == Scheme::Packed;
            }

            /** @brief Returns whether a routine takes a triangle. */
            [[nodiscard]] bool takes_a_uplo(const Routine& routine) noexcept
            {
                return routine.takes_uplo;
            }

            /** @brief Returns whether a routine takes a form of op(A). */
            [[nodiscard]] bool takes_a_trans(const Routine& routine) noexcept
            {
                return routine.takes_trans;
            }

            /** @brief Returns whether a routine takes a diagonal flag. */
            [[nodiscard]] bool takes_a_diag(const Routine& routine) noexcept
            {
                return routine.takes_diag;
            }

            /** @brief Returns whether a routine takes a scalar alpha. */
            [[nodiscard]] bool takes_an_alpha(const Routine& routine) noexcept
            {
                return routine.takes_alpha;
            }

            /** @brief Returns whether a routine takes a scalar beta. */
            [[nodiscard]] bool takes_a_beta(const Routine& routine) noexcept
            {
                return routine.takes_beta;
            }

            /** @brief Returns the axis values of every routine that select accepts. */
            [[nodiscard]] std::vector<RoutineAxis> routines_where(bool (*select)(const Routine&))
            {
                std::vector<RoutineAxis> axes;
                for (std::size_t index = 0; index < std::size(K_ROUTINES); ++index)
                    if (select(K_ROUTINES[index]))
                        axes.push_back({index});
                return axes;
            }
        } // namespace

        /**
         * @brief Builds operands every routine accepts, so one case changes one of them and
         *        expects the refusal.
         */
        class BlasStructuredCase : public BlasEngineStart
        {
            protected:
                /** @brief Returns a tensor of dims and dtype whose elements are not set. */
                [[nodiscard]] Tensor raw(const std::vector<int64_t>& dims,
                                         DataType dtype = DataType::Float32)
                {
                    return engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()), dtype);
                }

                /** @brief Returns a FLOAT32 tensor of dims that holds values. */
                [[nodiscard]] Tensor holding(const std::vector<int64_t>& dims,
                                             const std::vector<float>& values)
                {
                    Tensor tensor = raw(dims);
                    EXPECT_EQ(values.size(), static_cast<std::size_t>(tensor.num_elements));
                    TestEnvironment::queue()
                        .memcpy(tensor.data(), values.data(), values.size() * sizeof(float))
                        .wait();
                    return tensor;
                }

                /** @brief Returns a FLOAT32 tensor of dims with every element one. */
                [[nodiscard]] Tensor ones(const std::vector<int64_t>& dims)
                {
                    std::size_t count = 1;
                    for (const int64_t extent : dims)
                        count *= static_cast<std::size_t>(extent);
                    return holding(dims, std::vector<float>(count, 1.0f));
                }

                /** @brief Returns the shape of a band of lines of ld elements in the layout. */
                [[nodiscard]] std::vector<int64_t> band_dims(int64_t lines, int64_t ld) const
                {
                    if (engine->layout() == Core::Layout::ColumnMajor)
                        return {ld, lines};
                    return {lines, ld};
                }

                /** @brief Returns a tensor of dims and of the engine's layout with no storage. */
                [[nodiscard]] Tensor without_storage(const std::vector<int64_t>& dims) const
                {
                    return Tensor(SushiRuntime::sushi_ptr<Storage>(),
                                  SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                                  DataType::Float32, 0, engine->layout());
                }

                /**
                 * @brief Returns a rank-1 view of count elements of a new tensor, increment
                 *        elements apart.
                 */
                [[nodiscard]] Tensor strided(int64_t count, int64_t increment,
                                             DataType dtype = DataType::Float32)
                {
                    const Tensor parent = raw({count * (increment == 0 ? 1 : increment)}, dtype);
                    const int64_t dims[] = {count};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = increment;
                    return Tensor(parent.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, dtype, 0, parent.layout);
                }

                /** @brief Returns a view of a square tensor with its two axes exchanged. */
                [[nodiscard]] Tensor transposed_square(int64_t edge)
                {
                    return raw({edge, edge}).transpose(0, 1);
                }

                /** @brief Returns operands of ones that all thirteen routines accept. */
                [[nodiscard]] Operands valid_operands()
                {
                    Operands operands;
                    operands.general = ones(band_dims(K_ORDER, K_GENERAL_LD));
                    operands.triangle = ones(band_dims(K_ORDER, K_TRIANGLE_LD));
                    operands.packed = ones({K_PACKED_COUNT});
                    operands.x = ones({K_ORDER});
                    operands.y = ones({K_ORDER});
                    return operands;
                }

                /**
                 * @brief Expects record to throw std::invalid_argument and the engine to run the
                 *        graph the refusal left empty.
                 * @param record Callable `(SushiBLAS::BLASOps&) -> TaskHandle`.
                 */
                template<typename Record>
                void expect_refused(const std::string& what, Record&& record)
                {
                    SCOPED_TRACE(what);
                    SushiBLAS::BLASOps ops(recorder());
                    EXPECT_THROW((void)record(ops), std::invalid_argument);
                    EXPECT_TRUE(recording->run().has_value());
                }

                /** @brief Returns count FLOAT32 values drawn from [-1, 1]. */
                [[nodiscard]] static std::vector<float> random_values(std::mt19937& generator,
                                                                      int64_t count)
                {
                    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                    std::vector<float> values(static_cast<std::size_t>(count));
                    for (float& value : values)
                        value = distribution(generator);
                    return values;
                }

                /**
                 * @brief Records two calls on copies of initial in one graph, runs it and expects
                 *        the same bits from both.
                 * @param first Callable `(Tensor&)` that records on the first copy.
                 * @param second Callable `(Tensor&)` that records on the second copy.
                 */
                template<typename First, typename Second>
                void expect_same_bits(const std::vector<float>& initial, First&& first,
                                      Second&& second)
                {
                    const std::vector<int64_t> dims = {static_cast<int64_t>(initial.size())};
                    Tensor from_first = holding(dims, initial);
                    Tensor from_second = holding(dims, initial);
                    first(from_first);
                    second(from_second);
                    recording->run().value();

                    const std::vector<float> a = read_tensor<float>(from_first);
                    const std::vector<float> b = read_tensor<float>(from_second);
                    ASSERT_EQ(a.size(), b.size());
                    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
                }
        };

        /**
         * @brief Runs a case once per row of the structured suite.
         * @tparam Axis NoAxis, or the type of the value that completes each row.
         */
        template<typename Axis>
        using BlasStructuredRowCase = BlasRowCase<BlasStructuredCase, Axis>;

        /**
         * @brief Runs a case on one routine of K_ROUTINES.
         */
        class BlasStructuredRoutineCase : public BlasStructuredRowCase<RoutineAxis>
        {
            protected:
                /** @brief Returns the routine of the row. */
                [[nodiscard]] const Routine& routine() const
                {
                    return K_ROUTINES[GetParam().axis.index];
                }

                /**
                 * @brief Expects the routine of the row to refuse valid operands after one change.
                 * @param what Names the change for a failure message.
                 * @param change Callable `(Operands&)` that makes the operands unacceptable.
                 */
                template<typename Change>
                void expect_routine_refused(const char* what, Change&& change)
                {
                    Operands operands = valid_operands();
                    change(operands);
                    expect_refused(what, [&](SushiBLAS::BLASOps& ops)
                    {
                        return routine().record(ops, operands);
                    });
                }
        };

        /** @brief Runs the cases that take every one of the thirteen routines. */
        class Medium_BlasStructuredAnyRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the nine routines that stand for their kernels. */
        class Medium_BlasStructuredKernelRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the checked routines that take y. */
        class Medium_BlasStructuredYRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the checked routines whose vectors have a fixed order. */
        class Medium_BlasStructuredFixedOrderRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the fixed-order routines that take y. */
        class Medium_BlasStructuredFixedOrderYRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routine that takes the general band matrix. */
        class Medium_BlasStructuredGeneralBand : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the checked routines that take a band. */
        class Medium_BlasStructuredBandRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the checked routines that take a packed matrix. */
        class Medium_BlasStructuredPackedRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routines that take a triangle. */
        class Medium_BlasStructuredUploRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routines that take a form of op(A). */
        class Medium_BlasStructuredTransRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routines that take a diagonal flag. */
        class Medium_BlasStructuredDiagRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routines that take a scalar alpha. */
        class Medium_BlasStructuredAlphaRoutine : public BlasStructuredRoutineCase
        {

        };

        /** @brief Runs the cases of the routines that take a scalar beta. */
        class Medium_BlasStructuredBetaRoutine : public BlasStructuredRoutineCase
        {

        };

        /**
         * @brief Tests that every routine records and runs on the operands the refusals change.
         */
        TEST_P(Medium_BlasStructuredAnyRoutine, AcceptsTheOperandsARefusalStartsFrom)
        {
            Operands operands = valid_operands();
            SushiBLAS::BLASOps ops(recorder());
            EXPECT_NO_THROW((void)routine().record(ops, operands));
            EXPECT_TRUE(recording->run().has_value());
        }

        /**
         * @brief Tests that a matrix with elements and no storage is refused.
         */
        TEST_P(Medium_BlasStructuredKernelRoutine, RefusesAMatrixWithoutStorage)
        {
            expect_routine_refused("a matrix without storage", [&](Operands& o)
            {
                o.general = without_storage(band_dims(K_ORDER, K_GENERAL_LD));
                o.triangle = without_storage(band_dims(K_ORDER, K_TRIANGLE_LD));
                o.packed = without_storage({K_PACKED_COUNT});
            });
        }

        /**
         * @brief Tests that operands of an integer dtype are refused.
         */
        TEST_P(Medium_BlasStructuredKernelRoutine, RefusesInt32Operands)
        {
            expect_routine_refused("INT32 operands", [&](Operands& o)
            {
                o.general = raw(band_dims(K_ORDER, K_GENERAL_LD), DataType::Int32);
                o.triangle = raw(band_dims(K_ORDER, K_TRIANGLE_LD), DataType::Int32);
                o.packed = raw({K_PACKED_COUNT}, DataType::Int32);
                o.x = raw({K_ORDER}, DataType::Int32);
                o.y = raw({K_ORDER}, DataType::Int32);
            });
        }

        /**
         * @brief Tests that a vector of rank 2 that is not contiguous is refused; a 2 x 2 view
         *        holds K_ORDER elements, so only its strides are at fault.
         */
        TEST_P(Medium_BlasStructuredKernelRoutine, RefusesANonContiguousVectorOfRankTwo)
        {
            expect_routine_refused("a transposed 2 x 2 view as x",
                                   [&](Operands& o) { o.x = transposed_square(2); });
        }

        /**
         * @brief Tests that a vector whose increment is zero is refused.
         */
        TEST_P(Medium_BlasStructuredKernelRoutine, RefusesAVectorIncrementOfZero)
        {
            expect_routine_refused("x of increment 0",
                                   [&](Operands& o) { o.x = strided(K_ORDER, 0); });
        }

        /**
         * @brief Tests that an x whose dtype is not the matrix's is refused.
         */
        TEST_P(Medium_BlasStructuredKernelRoutine, RefusesAFloat64X)
        {
            expect_routine_refused("a FLOAT64 x",
                                   [&](Operands& o) { o.x = raw({K_ORDER}, DataType::Float64); });
        }

        /**
         * @brief Tests that a y whose dtype is not the matrix's is refused.
         */
        TEST_P(Medium_BlasStructuredYRoutine, RefusesAFloat64Y)
        {
            expect_routine_refused("a FLOAT64 y",
                                   [&](Operands& o) { o.y = raw({K_ORDER}, DataType::Float64); });
        }

        /**
         * @brief Tests that an x one element longer than the matrix order is refused.
         */
        TEST_P(Medium_BlasStructuredFixedOrderRoutine, RefusesAnXOneElementTooLong)
        {
            expect_routine_refused("x one element too long",
                                   [&](Operands& o) { o.x = ones({K_ORDER + 1}); });
        }

        /**
         * @brief Tests that a y one element longer than the matrix order is refused.
         */
        TEST_P(Medium_BlasStructuredFixedOrderYRoutine, RefusesAYOneElementTooLong)
        {
            expect_routine_refused("y one element too long",
                                   [&](Operands& o) { o.y = ones({K_ORDER + 1}); });
        }

        /**
         * @brief Tests that gbmv refuses the vector that runs along the stored extent when it is
         *        one element too long: x under COLUMN_MAJOR, y under ROW_MAJOR.
         */
        TEST_P(Medium_BlasStructuredGeneralBand, RefusesAVectorLongerThanTheStoredExtent)
        {
            const bool storage_gives_columns = (GetParam().layout == Core::Layout::ColumnMajor);
            expect_routine_refused(storage_gives_columns ? "x longer than the stored columns"
                                                         : "y longer than the stored rows",
                                   [&](Operands& o)
            {
                (storage_gives_columns ? o.x : o.y) = ones({K_ORDER + 1});
            });
        }

        /**
         * @brief Tests that gbmv refuses a super-diagonal count below zero.
         */
        TEST_P(Medium_BlasStructuredGeneralBand, RefusesANegativeUpperWidth)
        {
            expect_routine_refused("ku of -1", [](Operands& o) { o.ku = -1; });
        }

        /**
         * @brief Tests that a band width below zero is refused.
         */
        TEST_P(Medium_BlasStructuredBandRoutine, RefusesANegativeWidth)
        {
            expect_routine_refused("kl or k of -1", [](Operands& o)
            {
                o.kl = -1;
                o.k = -1;
            });
        }

        /**
         * @brief Tests that a band matrix of rank 1 is refused.
         */
        TEST_P(Medium_BlasStructuredBandRoutine, RefusesABandMatrixOfRankOne)
        {
            expect_routine_refused("a band matrix of rank 1", [&](Operands& o)
            {
                o.general = ones({K_ORDER * K_GENERAL_LD});
                o.triangle = ones({K_ORDER * K_TRIANGLE_LD});
            });
        }

        /**
         * @brief Tests that a band matrix of rank 3 is refused.
         */
        TEST_P(Medium_BlasStructuredBandRoutine, RefusesABandMatrixOfRankThree)
        {
            expect_routine_refused("a band matrix of rank 3", [&](Operands& o)
            {
                std::vector<int64_t> general = band_dims(K_ORDER, K_GENERAL_LD);
                std::vector<int64_t> triangle = band_dims(K_ORDER, K_TRIANGLE_LD);
                general.insert(general.begin(), 1);
                triangle.insert(triangle.begin(), 1);
                o.general = ones(general);
                o.triangle = ones(triangle);
            });
        }

        /**
         * @brief Tests that a band matrix whose strides are not those of its shape is refused;
         *        a square view keeps the lines and the line length the widths need.
         */
        TEST_P(Medium_BlasStructuredBandRoutine, RefusesAStridedBandMatrix)
        {
            expect_routine_refused("a transposed view as the band matrix", [&](Operands& o)
            {
                o.general = transposed_square(K_ORDER);
                o.triangle = transposed_square(K_ORDER);
            });
        }

        /**
         * @brief Tests that a band matrix with fewer elements per line than its widths need is
         *        refused.
         */
        TEST_P(Medium_BlasStructuredBandRoutine, RefusesABandNarrowerThanItsWidths)
        {
            expect_routine_refused("one off-diagonal more than a line holds", [](Operands& o)
            {
                o.ku = K_WIDTH + 1;
                o.k = K_WIDTH + 1;
            });
        }

        /**
         * @brief Tests that a packed matrix whose rank is not 1 is refused.
         */
        TEST_P(Medium_BlasStructuredPackedRoutine, RefusesAPackedMatrixOfRankTwo)
        {
            expect_routine_refused("a packed matrix of rank 2",
                                   [&](Operands& o) { o.packed = ones({K_PACKED_COUNT / 2, 2}); });
        }

        /**
         * @brief Tests that a packed matrix viewed with an increment is refused.
         */
        TEST_P(Medium_BlasStructuredPackedRoutine, RefusesAStridedPackedMatrix)
        {
            expect_routine_refused("every second element as the packed matrix",
                                   [&](Operands& o) { o.packed = strided(K_PACKED_COUNT, 2); });
        }

        /**
         * @brief Tests that a packed matrix whose element count is no n * (n + 1) / 2 is refused.
         */
        TEST_P(Medium_BlasStructuredPackedRoutine, RefusesAPackedCountThatIsNoTriangle)
        {
            expect_routine_refused("a packed matrix one element short",
                                   [&](Operands& o) { o.packed = ones({K_PACKED_COUNT - 1}); });
        }

        /**
         * @brief Tests that a triangle flag holding no enumerator is refused.
         */
        TEST_P(Medium_BlasStructuredUploRoutine, RefusesAUploThatIsNoEnumerator)
        {
            expect_routine_refused("uplo holding 0",
                                   [](Operands& o) { o.uplo = static_cast<Core::Uplo>(0); });
        }

        /**
         * @brief Tests that a form of op(A) holding no enumerator is refused.
         */
        TEST_P(Medium_BlasStructuredTransRoutine, RefusesATransposeThatIsNoEnumerator)
        {
            expect_routine_refused("trans holding 0",
                                   [](Operands& o) { o.trans = static_cast<Core::Transpose>(0); });
        }

        /**
         * @brief Tests that a diagonal flag holding no enumerator is refused.
         */
        TEST_P(Medium_BlasStructuredDiagRoutine, RefusesADiagThatIsNoEnumerator)
        {
            expect_routine_refused("diag holding 0",
                                   [](Operands& o) { o.diag = static_cast<Core::Diag>(0); });
        }

        /**
         * @brief Tests that a real tensor refuses an alpha with an imaginary part.
         */
        TEST_P(Medium_BlasStructuredAlphaRoutine, RefusesAComplexAlphaOnFloat32)
        {
            expect_routine_refused("a complex alpha on FLOAT32",
                                   [](Operands& o) { o.alpha = K_COMPLEX_SCALAR; });
        }

        /**
         * @brief Tests that a real tensor refuses a beta with an imaginary part.
         */
        TEST_P(Medium_BlasStructuredBetaRoutine, RefusesAComplexBetaOnFloat32)
        {
            expect_routine_refused("a complex beta on FLOAT32",
                                   [](Operands& o) { o.beta = K_COMPLEX_SCALAR; });
        }

        /**
         * @brief Runs the refusals of rotg, rotm and rotmg, which have no axis of their own.
         */
        class Medium_BlasStructuredRotation : public BlasStructuredRowCase<NoAxis>
        {
            protected:
                /** @brief Returns a FLOAT32 tensor of one element. */
                [[nodiscard]] Tensor one()
                {
                    return ones({1});
                }

                /** @brief Returns a FLOAT32 tensor of two elements. */
                [[nodiscard]] Tensor pair()
                {
                    return ones({2});
                }

                /** @brief Returns a FLOAT32 parameter tensor of a rotm or rotmg. */
                [[nodiscard]] Tensor parameters()
                {
                    return ones({K_PARAMETERS});
                }

                /** @brief Expects rotg to refuse the four operands. */
                void expect_rotg_refused(const std::string& what, Tensor a, Tensor b, Tensor c, Tensor s)
                {
                    expect_refused(std::string("rotg: ") + what, [&](SushiBLAS::BLASOps& ops)
                    {
                        return ops.rotg(a, b, c, s);
                    });
                }

                /** @brief Expects rotm to refuse the three operands. */
                void expect_rotm_refused(const std::string& what, Tensor x, Tensor y, const Tensor param)
                {
                    expect_refused(std::string("rotm: ") + what, [&](SushiBLAS::BLASOps& ops)
                    {
                        return ops.rotm(x, y, param);
                    });
                }

                /** @brief Expects rotmg to refuse the five operands. */
                void expect_rotmg_refused(const std::string& what, Tensor d1, Tensor d2, Tensor x1,
                                          const Tensor y1, Tensor param)
                {
                    expect_refused(std::string("rotmg: ") + what, [&](SushiBLAS::BLASOps& ops)
                    {
                        return ops.rotmg(d1, d2, x1, y1, param);
                    });
                }

                /** @brief Expects rotmg to refuse operands of one dtype. */
                void expect_rotmg_refuses_dtype(DataType dtype)
                {
                    expect_rotmg_refused(std::string(Core::to_string(dtype)) + " operands", raw({1}, dtype), raw({1}, dtype),
                                         raw({1}, dtype), raw({1}, dtype),
                                         raw({K_PARAMETERS}, dtype));
                }
        };

        /** @brief Runs the forwards of the Hermitian routines, one row per triangle. */
        class Medium_BlasStructuredHermitian : public BlasStructuredRowCase<Core::Uplo>
        {

        };

        /**
         * @brief Tests that rotg, rotmg and rotm record and run on the operands the refusals
         *        start from.
         */
        TEST_P(Medium_BlasStructuredRotation, AcceptsTheOperandsARefusalStartsFrom)
        {
            Tensor a = one();
            Tensor b = one();
            Tensor c = one();
            Tensor s = one();
            Tensor x = ones({K_ORDER});
            Tensor y = ones({K_ORDER});
            Tensor param = parameters();

            SushiBLAS::BLASOps ops(recorder());
            EXPECT_NO_THROW((void)ops.rotg(a, b, c, s));
            EXPECT_NO_THROW((void)ops.rotmg(a, b, c, s, param));
            EXPECT_NO_THROW((void)ops.rotm(x, y, param));
            EXPECT_TRUE(recording->run().has_value());
        }

        /** @brief Tests that rotg refuses INT32 operands. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesInt32Operands)
        {
            expect_rotg_refused("INT32 operands", raw({1}, DataType::Int32),
                                raw({1}, DataType::Int32), raw({1}, DataType::Int32),
                                raw({1}, DataType::Int32));
        }

        /** @brief Tests that rotg refuses an a of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesAOfTwoElements)
        {
            expect_rotg_refused("a of two elements", pair(), one(), one(), one());
        }

        /** @brief Tests that rotg refuses a b of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesBOfTwoElements)
        {
            expect_rotg_refused("b of two elements", one(), pair(), one(), one());
        }

        /** @brief Tests that rotg refuses a c of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesCOfTwoElements)
        {
            expect_rotg_refused("c of two elements", one(), one(), pair(), one());
        }

        /** @brief Tests that rotg refuses an s of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesSOfTwoElements)
        {
            expect_rotg_refused("s of two elements", one(), one(), one(), pair());
        }

        /** @brief Tests that rotg refuses an a of no elements. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesAOfNoElements)
        {
            expect_rotg_refused("a of no elements", raw({0}), one(), one(), one());
        }

        /** @brief Tests that rotg refuses a FLOAT64 b. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesAFloat64B)
        {
            expect_rotg_refused("a FLOAT64 b", one(), raw({1}, DataType::Float64), one(), one());
        }

        /** @brief Tests that rotg refuses a FLOAT64 c. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesAFloat64C)
        {
            expect_rotg_refused("a FLOAT64 c", one(), one(), raw({1}, DataType::Float64), one());
        }

        /** @brief Tests that rotg refuses a FLOAT64 s. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesAFloat64S)
        {
            expect_rotg_refused("a FLOAT64 s", one(), one(), one(), raw({1}, DataType::Float64));
        }

        /** @brief Tests that rotg refuses a complex c beside a complex a. */
        TEST_P(Medium_BlasStructuredRotation, RotgRefusesComplexOperands)
        {
            expect_rotg_refused("complex operands", raw({1}, DataType::Complex32),
                                raw({1}, DataType::Complex32), raw({1}, DataType::Complex32),
                                raw({1}, DataType::Complex32));
        }

        /** @brief Tests that rotm refuses INT32 operands. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesInt32Operands)
        {
            expect_rotm_refused("INT32 operands", raw({K_ORDER}, DataType::Int32),
                                raw({K_ORDER}, DataType::Int32),
                                raw({K_PARAMETERS}, DataType::Int32));
        }

        /** @brief Tests that rotm refuses a vector of rank 2 that is not contiguous. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesANonContiguousX)
        {
            expect_rotm_refused("a transposed 2 x 2 view as x", transposed_square(2),
                                ones({K_ORDER}), parameters());
        }

        /** @brief Tests that rotm refuses an x whose increment is zero. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAnXOfIncrementZero)
        {
            expect_rotm_refused("x of increment 0", strided(K_ORDER, 0), ones({K_ORDER}),
                                parameters());
        }

        /** @brief Tests that rotm refuses complex operands. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesComplexOperands)
        {
            expect_rotm_refused("complex operands", raw({K_ORDER}, DataType::Complex32),
                                raw({K_ORDER}, DataType::Complex32),
                                raw({K_PARAMETERS}, DataType::Complex32));
        }

        /** @brief Tests that rotm refuses a FLOAT64 y. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAFloat64Y)
        {
            expect_rotm_refused("a FLOAT64 y", ones({K_ORDER}), raw({K_ORDER}, DataType::Float64),
                                parameters());
        }

        /** @brief Tests that rotm refuses a FLOAT64 param. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAFloat64Param)
        {
            expect_rotm_refused("a FLOAT64 param", ones({K_ORDER}), ones({K_ORDER}),
                                raw({K_PARAMETERS}, DataType::Float64));
        }

        /** @brief Tests that rotm refuses a y one element longer than x. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAYOneElementTooLong)
        {
            expect_rotm_refused("y one element too long", ones({K_ORDER}), ones({K_ORDER + 1}),
                                parameters());
        }

        /** @brief Tests that rotm refuses a param of four elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAParamOfFourElements)
        {
            expect_rotm_refused("param of four elements", ones({K_ORDER}), ones({K_ORDER}),
                                ones({K_PARAMETERS - 1}));
        }

        /** @brief Tests that rotm refuses a param of six elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAParamOfSixElements)
        {
            expect_rotm_refused("param of six elements", ones({K_ORDER}), ones({K_ORDER}),
                                ones({K_PARAMETERS + 1}));
        }

        /** @brief Tests that rotm refuses a param of increment 2. */
        TEST_P(Medium_BlasStructuredRotation, RotmRefusesAParamOfIncrementTwo)
        {
            expect_rotm_refused("param of increment 2", ones({K_ORDER}), ones({K_ORDER}),
                                strided(K_PARAMETERS, 2));
        }

        /** @brief Tests that rotmg refuses INT32 operands. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesInt32Operands)
        {
            expect_rotmg_refuses_dtype(DataType::Int32);
        }

        /** @brief Tests that rotmg refuses COMPLEX32 operands. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesComplexOperands)
        {
            expect_rotmg_refuses_dtype(DataType::Complex32);
        }

        /** @brief Tests that rotmg refuses HALF operands. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesHalfOperands)
        {
            expect_rotmg_refuses_dtype(DataType::Half);
        }

        /** @brief Tests that rotmg refuses a d1 of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAD1OfTwoElements)
        {
            expect_rotmg_refused("d1 of two elements", pair(), one(), one(), one(), parameters());
        }

        /** @brief Tests that rotmg refuses a d2 of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAD2OfTwoElements)
        {
            expect_rotmg_refused("d2 of two elements", one(), pair(), one(), one(), parameters());
        }

        /** @brief Tests that rotmg refuses an x1 of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAnX1OfTwoElements)
        {
            expect_rotmg_refused("x1 of two elements", one(), one(), pair(), one(), parameters());
        }

        /** @brief Tests that rotmg refuses a y1 of two elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAY1OfTwoElements)
        {
            expect_rotmg_refused("y1 of two elements", one(), one(), one(), pair(), parameters());
        }

        /** @brief Tests that rotmg refuses a FLOAT64 d1. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAFloat64D1)
        {
            expect_rotmg_refused("a FLOAT64 d1", raw({1}, DataType::Float64), one(), one(), one(),
                                 parameters());
        }

        /** @brief Tests that rotmg refuses a FLOAT64 d2. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAFloat64D2)
        {
            expect_rotmg_refused("a FLOAT64 d2", one(), raw({1}, DataType::Float64), one(), one(),
                                 parameters());
        }

        /** @brief Tests that rotmg refuses a FLOAT64 x1. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAFloat64X1)
        {
            expect_rotmg_refused("a FLOAT64 x1", one(), one(), raw({1}, DataType::Float64), one(),
                                 parameters());
        }

        /** @brief Tests that rotmg refuses a FLOAT64 y1. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAFloat64Y1)
        {
            expect_rotmg_refused("a FLOAT64 y1", one(), one(), one(),
                                 raw({1}, DataType::Float64), parameters());
        }

        /** @brief Tests that rotmg refuses a FLOAT64 param. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAFloat64Param)
        {
            expect_rotmg_refused("a FLOAT64 param", one(), one(), one(), one(),
                                 raw({K_PARAMETERS}, DataType::Float64));
        }

        /** @brief Tests that rotmg refuses a param of four elements. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAParamOfFourElements)
        {
            expect_rotmg_refused("param of four elements", one(), one(), one(), one(),
                                 ones({K_PARAMETERS - 1}));
        }

        /** @brief Tests that rotmg refuses a param of increment 2. */
        TEST_P(Medium_BlasStructuredRotation, RotmgRefusesAParamOfIncrementTwo)
        {
            expect_rotmg_refused("param of increment 2", one(), one(), one(), one(),
                                 strided(K_PARAMETERS, 2));
        }

        /**
         * @brief Tests that hbmv on a FLOAT32 tensor gives the bits of sbmv.
         */
        TEST_P(Medium_BlasStructuredHermitian, HbmvGivesTheBitsOfSbmv)
        {
            const Core::Uplo uplo = GetParam().axis;
            std::mt19937 generator(K_SEED);
            SushiBLAS::BLASOps ops(recorder());
            const Tensor A = holding(band_dims(K_ORDER, K_TRIANGLE_LD),
                                     random_values(generator, K_ORDER * K_TRIANGLE_LD));
            const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
            const std::vector<float> y = random_values(generator, K_ORDER);

            expect_same_bits(
                y, [&](Tensor& out) { ops.hbmv(A, x, out, uplo, K_WIDTH, K_ALPHA, K_BETA); },
                [&](Tensor& out) { ops.sbmv(A, x, out, uplo, K_WIDTH, K_ALPHA, K_BETA); });
        }

        /**
         * @brief Tests that hpmv on a FLOAT32 tensor gives the bits of spmv.
         */
        TEST_P(Medium_BlasStructuredHermitian, HpmvGivesTheBitsOfSpmv)
        {
            const Core::Uplo uplo = GetParam().axis;
            std::mt19937 generator(K_SEED);
            SushiBLAS::BLASOps ops(recorder());
            const Tensor AP = holding({K_PACKED_COUNT}, random_values(generator, K_PACKED_COUNT));
            const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
            const std::vector<float> y = random_values(generator, K_ORDER);

            expect_same_bits(
                y, [&](Tensor& out) { ops.hpmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                [&](Tensor& out) { ops.spmv(AP, x, out, uplo, K_ALPHA, K_BETA); });
        }

        /**
         * @brief Tests that hpr on a FLOAT32 tensor gives the bits of spr.
         */
        TEST_P(Medium_BlasStructuredHermitian, HprGivesTheBitsOfSpr)
        {
            const Core::Uplo uplo = GetParam().axis;
            std::mt19937 generator(K_SEED);
            SushiBLAS::BLASOps ops(recorder());
            const std::vector<float> ap = random_values(generator, K_PACKED_COUNT);
            const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));

            expect_same_bits(ap, [&](Tensor& out) { ops.hpr(x, out, uplo, K_ALPHA); },
                             [&](Tensor& out) { ops.spr(x, out, uplo, K_ALPHA); });
        }

        /**
         * @brief Tests that hpr2 on a FLOAT32 tensor gives the bits of spr2.
         */
        TEST_P(Medium_BlasStructuredHermitian, Hpr2GivesTheBitsOfSpr2)
        {
            const Core::Uplo uplo = GetParam().axis;
            std::mt19937 generator(K_SEED);
            SushiBLAS::BLASOps ops(recorder());
            const std::vector<float> ap = random_values(generator, K_PACKED_COUNT);
            const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
            const Tensor y = holding({K_ORDER}, random_values(generator, K_ORDER));

            expect_same_bits(ap, [&](Tensor& out) { ops.hpr2(x, y, out, uplo, K_ALPHA); },
                             [&](Tensor& out) { ops.spr2(x, y, out, uplo, K_ALPHA); });
        }

        /// @brief The rows of every routine, every checked routine and the other subsets.
        const std::vector<RoutineAxis> K_ANY = routines_where(is_any);
        const std::vector<RoutineAxis> K_KERNELS = routines_where(represents_its_kernels);
        const std::vector<RoutineAxis> K_KERNELS_WITH_Y = routines_where(takes_y_of_a_kernel);
        const std::vector<RoutineAxis> K_FIXED_ORDER = routines_where(has_a_fixed_order);
        const std::vector<RoutineAxis> K_FIXED_ORDER_WITH_Y = routines_where(has_a_fixed_order_and_y);
        const std::vector<RoutineAxis> K_GENERAL_BAND = routines_where(takes_a_general_band);
        const std::vector<RoutineAxis> K_BANDS = routines_where(takes_a_band);
        const std::vector<RoutineAxis> K_PACKED = routines_where(takes_a_packed_matrix);
        const std::vector<RoutineAxis> K_WITH_UPLO = routines_where(takes_a_uplo);
        const std::vector<RoutineAxis> K_WITH_TRANS = routines_where(takes_a_trans);
        const std::vector<RoutineAxis> K_WITH_DIAG = routines_where(takes_a_diag);
        const std::vector<RoutineAxis> K_WITH_ALPHA = routines_where(takes_an_alpha);
        const std::vector<RoutineAxis> K_WITH_BETA = routines_where(takes_a_beta);

        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredAnyRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_ANY)), row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredKernelRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_KERNELS)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredYRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_KERNELS_WITH_Y)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredFixedOrderRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_FIXED_ORDER)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredFixedOrderYRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_FIXED_ORDER_WITH_Y)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredGeneralBand,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_GENERAL_BAND)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredBandRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_BANDS)), row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredPackedRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_PACKED)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredUploRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_WITH_UPLO)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredTransRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_WITH_TRANS)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredDiagRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_WITH_DIAG)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredAlphaRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_WITH_ALPHA)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredBetaRoutine,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_WITH_BETA)),
            row_name<RoutineAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredRotation, ::testing::ValuesIn(plain_rows(K_FLOAT32, K_BOTH_MODES)),
            row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_BlasStructuredHermitian,
            ::testing::ValuesIn(rows_of(K_FLOAT32, K_BOTH_MODES, K_TRIANGLES)),
            row_name<Core::Uplo>);
    } // namespace Test
} // namespace SushiBLAS
