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
 * @file test_blas_structured_validation.cpp
 * @brief Tests what the band, packed and rotation routines refuse at record time, and the
 *        forwards of the Hermitian routines on a real tensor.
 * @author Mustafa Garip
 */

#include <array>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/blas_types.hpp>
#include "test_utils.hpp"

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
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both element orders a band or a packed triangle can be stored in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

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

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns whether a routine is the one checked for the geometry of a pair. */
            [[nodiscard]] bool represents_its_kernels(const Routine& routine) noexcept
            {
                return routine.represents_its_kernels;
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
        } // namespace

        /**
         * @brief Builds operands every routine accepts, changes one and expects the refusal.
         */
        class Unit_BlasStructuredValidation : public SushiBLASTest
        {
            protected:
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
                            reinit_engine(layout);
                            engine->set_mode(mode);
                            body(layout);
                        }
                    }
                }

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
                    SushiBLAS::BLASOps ops(*engine);
                    EXPECT_THROW((void)record(ops), std::invalid_argument);
                    EXPECT_TRUE(engine->execute().has_value());
                }

                /**
                 * @brief Expects every selected routine to refuse valid operands after one change.
                 * @param select Callable `(const Routine&) -> bool`.
                 * @param change Callable `(Operands&)` that makes the operands unacceptable.
                 */
                template<typename Select, typename Change>
                void expect_refusals(const char* what, Select&& select, Change&& change)
                {
                    for (const Routine& routine : K_ROUTINES)
                    {
                        if (!select(routine))
                            continue;

                        Operands operands = valid_operands();
                        change(operands);
                        expect_refused(std::string(routine.name) + ": " + what,
                                       [&](SushiBLAS::BLASOps& ops)
                                       {
                                           return routine.record(ops, operands);
                                       });
                    }
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
                    engine->execute().value();

                    const std::vector<float> a = read_tensor<float>(from_first);
                    const std::vector<float> b = read_tensor<float>(from_second);
                    ASSERT_EQ(a.size(), b.size());
                    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
                }
        };

        /**
         * @brief Tests that every routine records and runs on the operands the refusals change.
         */
        TEST_F(Unit_BlasStructuredValidation, EveryRoutineAcceptsTheOperandsARefusalStartsFrom)
        {
            for_each_engine([&](Core::Layout)
            {
                for (const Routine& routine : K_ROUTINES)
                {
                    SCOPED_TRACE(routine.name);
                    Operands operands = valid_operands();
                    SushiBLAS::BLASOps ops(*engine);
                    EXPECT_NO_THROW((void)routine.record(ops, operands));
                    EXPECT_TRUE(engine->execute().has_value());
                }

                Tensor a = ones({1});
                Tensor b = ones({1});
                Tensor c = ones({1});
                Tensor s = ones({1});
                Tensor x = ones({K_ORDER});
                Tensor y = ones({K_ORDER});
                Tensor param = ones({K_PARAMETERS});

                SushiBLAS::BLASOps ops(*engine);
                EXPECT_NO_THROW((void)ops.rotg(a, b, c, s));
                EXPECT_NO_THROW((void)ops.rotmg(a, b, c, s, param));
                EXPECT_NO_THROW((void)ops.rotm(x, y, param));
                EXPECT_TRUE(engine->execute().has_value());
            });
        }

        /**
         * @brief Tests that a matrix with elements and no storage is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AMatrixWithoutStorageIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a matrix without storage", represents_its_kernels,
                                [&](Operands& o)
                {
                    o.general = without_storage(band_dims(K_ORDER, K_GENERAL_LD));
                    o.triangle = without_storage(band_dims(K_ORDER, K_TRIANGLE_LD));
                    o.packed = without_storage({K_PACKED_COUNT});
                });
            });
        }

        /**
         * @brief Tests that operands of an integer dtype are refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AnIntegerDtypeIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("INT32 operands", represents_its_kernels, [&](Operands& o)
                {
                    o.general = raw(band_dims(K_ORDER, K_GENERAL_LD), DataType::Int32);
                    o.triangle = raw(band_dims(K_ORDER, K_TRIANGLE_LD), DataType::Int32);
                    o.packed = raw({K_PACKED_COUNT}, DataType::Int32);
                    o.x = raw({K_ORDER}, DataType::Int32);
                    o.y = raw({K_ORDER}, DataType::Int32);
                });

                Tensor scalar = raw({1}, DataType::Int32);
                Tensor other = raw({1}, DataType::Int32);
                Tensor third = raw({1}, DataType::Int32);
                Tensor fourth = raw({1}, DataType::Int32);
                Tensor x = raw({K_ORDER}, DataType::Int32);
                Tensor y = raw({K_ORDER}, DataType::Int32);
                Tensor param = raw({K_PARAMETERS}, DataType::Int32);

                expect_refused("rotg: INT32 operands", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(scalar, other, third, fourth);
                });
                expect_refused("rotm: INT32 operands", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotm(x, y, param);
                });
                expect_refused("rotmg: INT32 operands", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(scalar, other, third, fourth, param);
                });
            });
        }

        /**
         * @brief Tests that a vector of rank 2 that is not contiguous is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, ANonContiguousVectorOfRankTwoIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                // A 2 x 2 view holds K_ORDER elements, so only its strides are at fault.
                expect_refusals("a transposed 2 x 2 view as x", represents_its_kernels,
                                [&](Operands& o) { o.x = transposed_square(2); });

                Tensor x = transposed_square(2);
                Tensor y = ones({K_ORDER});
                const Tensor param = ones({K_PARAMETERS});
                expect_refused("rotm: a transposed 2 x 2 view as x", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotm(x, y, param);
                });
            });
        }

        /**
         * @brief Tests that a vector whose increment is zero is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AVectorIncrementOfZeroIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("x of increment 0", represents_its_kernels,
                                [&](Operands& o) { o.x = strided(K_ORDER, 0); });

                Tensor x = strided(K_ORDER, 0);
                Tensor y = ones({K_ORDER});
                const Tensor param = ones({K_PARAMETERS});
                expect_refused("rotm: x of increment 0", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotm(x, y, param);
                });
            });
        }

        /**
         * @brief Tests that each flag is refused when it holds no enumerator of its type.
         */
        TEST_F(Unit_BlasStructuredValidation, AFlagThatIsNoEnumeratorIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("uplo holding 0",
                                [](const Routine& routine) { return routine.takes_uplo; },
                                [](Operands& o) { o.uplo = static_cast<Core::Uplo>(0); });
                expect_refusals("trans holding 0",
                                [](const Routine& routine) { return routine.takes_trans; },
                                [](Operands& o) { o.trans = static_cast<Core::Transpose>(0); });
                expect_refusals("diag holding 0",
                                [](const Routine& routine) { return routine.takes_diag; },
                                [](Operands& o) { o.diag = static_cast<Core::Diag>(0); });
            });
        }

        /**
         * @brief Tests that a real tensor refuses a scalar with an imaginary part.
         */
        TEST_F(Unit_BlasStructuredValidation, AnImaginaryScalarOnARealTensorIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a complex alpha on FLOAT32",
                                [](const Routine& routine) { return routine.takes_alpha; },
                                [](Operands& o) { o.alpha = K_COMPLEX_SCALAR; });
                expect_refusals("a complex beta on FLOAT32",
                                [](const Routine& routine) { return routine.takes_beta; },
                                [](Operands& o) { o.beta = K_COMPLEX_SCALAR; });
            });
        }

        /**
         * @brief Tests that a vector whose dtype is not the matrix's is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AVectorOfAnotherDtypeIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a FLOAT64 x", represents_its_kernels,
                                [&](Operands& o) { o.x = raw({K_ORDER}, DataType::Float64); });
                expect_refusals("a FLOAT64 y",
                                [](const Routine& routine)
                                {
                                    return routine.represents_its_kernels && routine.takes_y;
                                },
                                [&](Operands& o) { o.y = raw({K_ORDER}, DataType::Float64); });
            });
        }

        /**
         * @brief Tests that a vector whose length is not the matrix extent it runs along is
         *        refused; gbmv takes one extent from a vector, so only its other one can err.
         */
        TEST_F(Unit_BlasStructuredValidation, AVectorOfTheWrongLengthIsRefused)
        {
            for_each_engine([&](Core::Layout layout)
            {
                const auto has_a_fixed_order = [](const Routine& routine)
                {
                    return routine.represents_its_kernels &&
                           routine.scheme != Scheme::GeneralBand;
                };

                expect_refusals("x one element too long", has_a_fixed_order,
                                [&](Operands& o) { o.x = ones({K_ORDER + 1}); });
                expect_refusals("y one element too long",
                                [&](const Routine& routine)
                                {
                                    return has_a_fixed_order(routine) && routine.takes_y;
                                },
                                [&](Operands& o) { o.y = ones({K_ORDER + 1}); });

                // The storage gives n under COLUMN_MAJOR and m under ROW_MAJOR.
                const bool storage_gives_columns = (layout == Core::Layout::ColumnMajor);
                expect_refusals(storage_gives_columns ? "x longer than the stored columns"
                                                      : "y longer than the stored rows",
                                [](const Routine& routine)
                                {
                                    return routine.scheme == Scheme::GeneralBand;
                                },
                                [&](Operands& o)
                                {
                                    (storage_gives_columns ? o.x : o.y) = ones({K_ORDER + 1});
                                });
            });
        }

        /**
         * @brief Tests that a band width below zero is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, ANegativeBandWidthIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("kl or k of -1", takes_a_band, [](Operands& o)
                {
                    o.kl = -1;
                    o.k = -1;
                });
                expect_refusals("ku of -1",
                                [](const Routine& routine)
                                {
                                    return routine.scheme == Scheme::GeneralBand;
                                },
                                [](Operands& o) { o.ku = -1; });
            });
        }

        /**
         * @brief Tests that a band matrix of rank 1 or of rank 3 is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, ABandMatrixOfAnotherRankIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a band matrix of rank 1", takes_a_band, [&](Operands& o)
                {
                    o.general = ones({K_ORDER * K_GENERAL_LD});
                    o.triangle = ones({K_ORDER * K_TRIANGLE_LD});
                });
                expect_refusals("a band matrix of rank 3", takes_a_band, [&](Operands& o)
                {
                    std::vector<int64_t> general = band_dims(K_ORDER, K_GENERAL_LD);
                    std::vector<int64_t> triangle = band_dims(K_ORDER, K_TRIANGLE_LD);
                    general.insert(general.begin(), 1);
                    triangle.insert(triangle.begin(), 1);
                    o.general = ones(general);
                    o.triangle = ones(triangle);
                });
            });
        }

        /**
         * @brief Tests that a band matrix whose strides are not those of its shape is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AStridedBandMatrixIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                // A square view keeps the lines and the line length the widths need.
                expect_refusals("a transposed view as the band matrix", takes_a_band,
                                [&](Operands& o)
                {
                    o.general = transposed_square(K_ORDER);
                    o.triangle = transposed_square(K_ORDER);
                });
            });
        }

        /**
         * @brief Tests that a band matrix with fewer elements per line than its widths need is
         *        refused.
         */
        TEST_F(Unit_BlasStructuredValidation, ABandNarrowerThanItsWidthsIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("one off-diagonal more than a line holds", takes_a_band,
                                [](Operands& o)
                {
                    o.ku = K_WIDTH + 1;
                    o.k = K_WIDTH + 1;
                });
            });
        }

        /**
         * @brief Tests that a packed matrix whose rank is not 1 is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, APackedMatrixOfAnotherRankIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a packed matrix of rank 2", takes_a_packed_matrix,
                                [&](Operands& o) { o.packed = ones({K_PACKED_COUNT / 2, 2}); });
            });
        }

        /**
         * @brief Tests that a packed matrix viewed with an increment is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, AStridedPackedMatrixIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("every second element as the packed matrix",
                                takes_a_packed_matrix,
                                [&](Operands& o) { o.packed = strided(K_PACKED_COUNT, 2); });
            });
        }

        /**
         * @brief Tests that a packed matrix whose element count is no n * (n + 1) / 2 is refused.
         */
        TEST_F(Unit_BlasStructuredValidation, APackedCountThatIsNoTriangleIsRefused)
        {
            for_each_engine([&](Core::Layout)
            {
                expect_refusals("a packed matrix one element short", takes_a_packed_matrix,
                                [&](Operands& o) { o.packed = ones({K_PACKED_COUNT - 1}); });
            });
        }

        /**
         * @brief Tests the element counts and the dtypes rotg refuses.
         */
        TEST_F(Unit_BlasStructuredValidation, RotgRefusesItsOperands)
        {
            for_each_engine([&](Core::Layout)
            {
                Tensor a = ones({1});
                Tensor b = ones({1});
                Tensor c = ones({1});
                Tensor s = ones({1});
                Tensor pair = ones({2});
                Tensor none = raw({0});
                Tensor wide = raw({1}, DataType::Float64);

                expect_refused("a of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(pair, b, c, s);
                });
                expect_refused("b of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, pair, c, s);
                });
                expect_refused("c of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, b, pair, s);
                });
                expect_refused("s of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, b, c, pair);
                });
                expect_refused("a of no elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(none, b, c, s);
                });
                expect_refused("a FLOAT64 b", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, wide, c, s);
                });
                expect_refused("a FLOAT64 s", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, b, c, wide);
                });
                expect_refused("a FLOAT64 c", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotg(a, b, wide, s);
                });
                expect_refused("a complex c beside a complex a", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor complex_a = raw({1}, DataType::Complex32);
                    Tensor complex_b = raw({1}, DataType::Complex32);
                    Tensor complex_c = raw({1}, DataType::Complex32);
                    Tensor complex_s = raw({1}, DataType::Complex32);
                    return ops.rotg(complex_a, complex_b, complex_c, complex_s);
                });
            });
        }

        /**
         * @brief Tests the dtypes, the lengths and the parameter tensors rotm refuses.
         */
        TEST_F(Unit_BlasStructuredValidation, RotmRefusesItsOperands)
        {
            for_each_engine([&](Core::Layout)
            {
                Tensor x = ones({K_ORDER});
                Tensor y = ones({K_ORDER});
                const Tensor param = ones({K_PARAMETERS});

                expect_refused("complex operands", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor complex_x = raw({K_ORDER}, DataType::Complex32);
                    Tensor complex_y = raw({K_ORDER}, DataType::Complex32);
                    const Tensor complex_param = raw({K_PARAMETERS}, DataType::Complex32);
                    return ops.rotm(complex_x, complex_y, complex_param);
                });
                expect_refused("a FLOAT64 y", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor other = raw({K_ORDER}, DataType::Float64);
                    return ops.rotm(x, other, param);
                });
                expect_refused("a FLOAT64 param", [&](SushiBLAS::BLASOps& ops)
                {
                    const Tensor other = raw({K_PARAMETERS}, DataType::Float64);
                    return ops.rotm(x, y, other);
                });
                expect_refused("y one element too long", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor other = ones({K_ORDER + 1});
                    return ops.rotm(x, other, param);
                });
                expect_refused("param of four elements", [&](SushiBLAS::BLASOps& ops)
                {
                    const Tensor other = ones({K_PARAMETERS - 1});
                    return ops.rotm(x, y, other);
                });
                expect_refused("param of six elements", [&](SushiBLAS::BLASOps& ops)
                {
                    const Tensor other = ones({K_PARAMETERS + 1});
                    return ops.rotm(x, y, other);
                });
                expect_refused("param of increment 2", [&](SushiBLAS::BLASOps& ops)
                {
                    const Tensor other = strided(K_PARAMETERS, 2);
                    return ops.rotm(x, y, other);
                });
            });
        }

        /**
         * @brief Tests the dtypes, the element counts and the parameter tensors rotmg refuses.
         */
        TEST_F(Unit_BlasStructuredValidation, RotmgRefusesItsOperands)
        {
            for_each_engine([&](Core::Layout)
            {
                Tensor d1 = ones({1});
                Tensor d2 = ones({1});
                Tensor x1 = ones({1});
                Tensor y1 = ones({1});
                Tensor param = ones({K_PARAMETERS});
                Tensor pair = ones({2});
                Tensor wide = raw({1}, DataType::Float64);

                for (const DataType dtype : {DataType::Complex32, DataType::Half})
                {
                    expect_refused(std::string(Core::to_string(dtype)) + " operands",
                                   [&](SushiBLAS::BLASOps& ops)
                    {
                        Tensor other_d1 = raw({1}, dtype);
                        Tensor other_d2 = raw({1}, dtype);
                        Tensor other_x1 = raw({1}, dtype);
                        const Tensor other_y1 = raw({1}, dtype);
                        Tensor other_param = raw({K_PARAMETERS}, dtype);
                        return ops.rotmg(other_d1, other_d2, other_x1, other_y1, other_param);
                    });
                }

                expect_refused("d1 of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(pair, d2, x1, y1, param);
                });
                expect_refused("d2 of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, pair, x1, y1, param);
                });
                expect_refused("x1 of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, d2, pair, y1, param);
                });
                expect_refused("y1 of two elements", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, d2, x1, pair, param);
                });
                expect_refused("a FLOAT64 d1", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(wide, d2, x1, y1, param);
                });
                expect_refused("a FLOAT64 d2", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, wide, x1, y1, param);
                });
                expect_refused("a FLOAT64 x1", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, d2, wide, y1, param);
                });
                expect_refused("a FLOAT64 y1", [&](SushiBLAS::BLASOps& ops)
                {
                    return ops.rotmg(d1, d2, x1, wide, param);
                });
                expect_refused("a FLOAT64 param", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor other = raw({K_PARAMETERS}, DataType::Float64);
                    return ops.rotmg(d1, d2, x1, y1, other);
                });
                expect_refused("param of four elements", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor other = ones({K_PARAMETERS - 1});
                    return ops.rotmg(d1, d2, x1, y1, other);
                });
                expect_refused("param of increment 2", [&](SushiBLAS::BLASOps& ops)
                {
                    Tensor other = strided(K_PARAMETERS, 2);
                    return ops.rotmg(d1, d2, x1, y1, other);
                });
            });
        }

        /**
         * @brief Tests that hbmv on a FLOAT32 tensor gives the bits of sbmv.
         */
        TEST_F(Unit_BlasStructuredValidation, HbmvOnARealTensorGivesTheBitsOfSbmv)
        {
            for_each_engine([&](Core::Layout)
            {
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(*engine);
                const Tensor A = holding(band_dims(K_ORDER, K_TRIANGLE_LD),
                                         random_values(generator, K_ORDER * K_TRIANGLE_LD));
                const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
                const std::vector<float> y = random_values(generator, K_ORDER);

                for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    expect_same_bits(
                        y,
                        [&](Tensor& out) { ops.hbmv(A, x, out, uplo, K_WIDTH, K_ALPHA, K_BETA); },
                        [&](Tensor& out) { ops.sbmv(A, x, out, uplo, K_WIDTH, K_ALPHA, K_BETA); });
                }
            });
        }

        /**
         * @brief Tests that hpmv on a FLOAT32 tensor gives the bits of spmv.
         */
        TEST_F(Unit_BlasStructuredValidation, HpmvOnARealTensorGivesTheBitsOfSpmv)
        {
            for_each_engine([&](Core::Layout)
            {
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(*engine);
                const Tensor AP =
                    holding({K_PACKED_COUNT}, random_values(generator, K_PACKED_COUNT));
                const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
                const std::vector<float> y = random_values(generator, K_ORDER);

                for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    expect_same_bits(
                        y,
                        [&](Tensor& out) { ops.hpmv(AP, x, out, uplo, K_ALPHA, K_BETA); },
                        [&](Tensor& out) { ops.spmv(AP, x, out, uplo, K_ALPHA, K_BETA); });
                }
            });
        }

        /**
         * @brief Tests that hpr and hpr2 on a FLOAT32 tensor give the bits of spr and spr2.
         */
        TEST_F(Unit_BlasStructuredValidation, HprAndHpr2OnARealTensorGiveTheBitsOfSprAndSpr2)
        {
            for_each_engine([&](Core::Layout)
            {
                std::mt19937 generator(K_SEED);
                SushiBLAS::BLASOps ops(*engine);
                const std::vector<float> ap = random_values(generator, K_PACKED_COUNT);
                const Tensor x = holding({K_ORDER}, random_values(generator, K_ORDER));
                const Tensor y = holding({K_ORDER}, random_values(generator, K_ORDER));

                for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                {
                    SCOPED_TRACE(Core::to_string(uplo));
                    expect_same_bits(ap,
                                     [&](Tensor& out) { ops.hpr(x, out, uplo, K_ALPHA); },
                                     [&](Tensor& out) { ops.spr(x, out, uplo, K_ALPHA); });
                    expect_same_bits(ap,
                                     [&](Tensor& out) { ops.hpr2(x, y, out, uplo, K_ALPHA); },
                                     [&](Tensor& out) { ops.spr2(x, y, out, uplo, K_ALPHA); });
                }
            });
        }
    } // namespace Test
} // namespace SushiBLAS
