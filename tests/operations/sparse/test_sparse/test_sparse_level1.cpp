/**************************************************************************/
/* test_sparse_level1.cpp                                                 */
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
 * @file test_sparse_level1.cpp
 * @brief Tests the sparse-dense level 1 routines of SparseOps against values worked out by hand.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief A host vector of expectations or of operand values. */
            using Wide = std::vector<std::complex<double>>;

            /// @brief Length of the dense vector of every hand-worked case.
            constexpr int64_t K_SIZE = 8;

            /// @brief The value the elements between those of a strided view hold.
            constexpr double K_FILLER = -7.0;

            /// @brief Length of the dense vector of the long dot product.
            constexpr int64_t K_LONG_SIZE = 100000;

            /// @brief One stored position per this many dense positions in the long dot product.
            constexpr int64_t K_LONG_SPACING = 3;

            /// @brief Stored entries of the long dot product.
            constexpr int64_t K_LONG_ENTRIES = (K_LONG_SIZE + K_LONG_SPACING - 1) / K_LONG_SPACING;

            static_assert(K_LONG_ENTRIES == 33334 &&
                              static_cast<std::size_t>(K_LONG_ENTRIES) >
                                  Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS,
                          "The long dot product has to take the two-stage shape on a CPU device.");

            /** @brief Returns the stored positions of the hand-worked cases, zero-based. */
            [[nodiscard]] std::vector<int64_t> sample_positions()
            {
                return {6, 1, 4};
            }

            /** @brief Returns the stored values 10, 20, 30 of the hand-worked cases. */
            [[nodiscard]] Wide sample_entries()
            {
                return {{10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}};
            }

            /** @brief Returns the complex doubles of a list of real values. */
            [[nodiscard]] Wide reals(const std::vector<double>& values)
            {
                Wide wide(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    wide[i] = std::complex<double>(values[i], 0.0);
                return wide;
            }
        } // namespace

        /**
         * @brief Builds a sparse vector and its dense partner and reads back what a call left.
         */
        template<typename TypeParam>
        class Medium_SparseLevel1Test : public SushiBLASTest
        {
            protected:
                /** @brief The element type under test. */
                using T = typename TypeParam::Value;

                /** @brief The index type under test. */
                using Index = typename TypeParam::IndexType;

                /** @brief Holds the operands of one level 1 call. */
                struct Operands
                {
                    SparseVector x; /**< The sparse vector. */
                    Tensor y;       /**< The dense vector, of the length x.size. */
                    Tensor result;  /**< The scalar a dot product writes; it starts at 9. */
                };

                /** @brief Skips the HALF suite on a device without the fp16 aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (std::is_same_v<T, sycl::half> &&
                        !TestEnvironment::queue().get_device().has(
                            sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Returns the sparse vector of zero-based positions, stored with base. */
                [[nodiscard]] SparseVector build(int64_t size,
                                                 const std::vector<int64_t>& positions,
                                                 const Wide& values, Core::IndexBase base)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);

                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.vector_from_entries(size, stored_indices<Index>(positions, base),
                                                       host, base);
                }

                /** @brief Returns a rank-1 tensor holding values narrowed to T. */
                [[nodiscard]] Tensor vector_of(const Wide& values)
                {
                    std::vector<T> host(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        host[i] = narrowed<T>(values[i]);

                    const int64_t count = static_cast<int64_t>(values.size());
                    return create_initialized_tensor<T>({count}, host);
                }

                /** @brief Sets the mode and returns operands built from host data. */
                [[nodiscard]] Operands operands(ExecutionMode mode, Core::IndexBase base,
                                                const std::vector<int64_t>& positions,
                                                const Wide& entries, const Wide& dense)
                {
                    engine->set_mode(mode);
                    return Operands{
                        build(static_cast<int64_t>(dense.size()), positions, entries, base),
                        vector_of(dense), vector_of(Wide(1, std::complex<double>(9.0, 0.0)))};
                }

                /** @brief Sets the mode and returns x = (10, 20, 30) at 6, 1, 4 and y = 1 to 8. */
                [[nodiscard]] Operands operands(ExecutionMode mode, Core::IndexBase base)
                {
                    return operands(mode, base, sample_positions(), sample_entries(),
                                    counting(static_cast<std::size_t>(K_SIZE), 1.0));
                }

                /** @brief Returns the stride-2 view of the even elements of a tensor. */
                [[nodiscard]] static Tensor every_second(const Tensor& wide)
                {
                    const int64_t dims[1] = {wide.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(wide.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, wide.dtype, 0, wide.layout);
                }

                /** @brief Expects every element within tolerance of its host expectation. */
                static void expect_near(const std::vector<T>& actual, const Wide& expected,
                                        double tolerance)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> got = widened(actual[i]);
                        EXPECT_NEAR(got.real(), expected[i].real(), tolerance) << "element " << i;
                        EXPECT_NEAR(got.imag(), expected[i].imag(), tolerance) << "element " << i;
                    }
                }

                /** @brief Returns whether two values of T hold the same bits. */
                [[nodiscard]] static bool same_bits(const T& a, const T& b)
                {
                    return std::memcmp(&a, &b, sizeof(T)) == 0;
                }
        };

        SB_TYPED_TEST_SUITE(Medium_SparseLevel1Test, SparseCases);

        TYPED_TEST(Medium_SparseLevel1Test, AxpyiAddsTheScaledEntriesAtTheirPositions)
        {
            using T = typename TypeParam::Value;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    SparseOps(*this->engine).axpyi(2.0, on.x, on.y);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.y), reals({1, 42, 3, 4, 65, 6, 27, 8}),
                                      0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, DotiIsDefinedForRealTensorsAndNamesItsSiblings)
        {
            using T = typename TypeParam::Value;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    if (!Core::is_complex_v<T>)
                    {
                        // 10 * 7 + 20 * 2 + 30 * 5.
                        SparseOps(*this->engine).doti(on.x, on.y, on.result);
                        this->engine->execute().value();
                        this->expect_near(read_tensor<T>(on.result), reals({260}), 0.0);
                        continue;
                    }

                    std::string refusal;
                    try
                    {
                        SparseOps(*this->engine).doti(on.x, on.y, on.result);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        refusal = error.what();
                    }
                    EXPECT_NE(refusal.find("dotui"), std::string::npos)
                        << "doti on a complex tensor threw \"" << refusal << "\"";
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, DotuiMultipliesAsStoredAndDotciConjugatesX)
        {
            using T = typename TypeParam::Value;

            // One entry, 1 + 2i at position 3, against y[3] = 3 + 4i.
            const Wide entry = {{1.0, Core::is_complex_v<T> ? 2.0 : 0.0}};
            Wide dense = counting(static_cast<std::size_t>(K_SIZE), 1.0);
            dense[3] = {3.0, Core::is_complex_v<T> ? 4.0 : 0.0};

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base, {3}, entry, dense);
                    Tensor conjugated = this->vector_of(reals({9}));

                    SparseOps ops(*this->engine);
                    ops.dotui(on.x, on.y, on.result);
                    ops.dotci(on.x, on.y, conjugated);
                    this->engine->execute().value();

                    if (Core::is_complex_v<T>)
                    {
                        this->expect_near(read_tensor<T>(on.result), {{-5.0, 10.0}}, 0.0);
                        this->expect_near(read_tensor<T>(conjugated), {{11.0, -2.0}}, 0.0);
                    }
                    else
                    {
                        this->expect_near(read_tensor<T>(on.result), reals({3}), 0.0);
                        this->expect_near(read_tensor<T>(conjugated), reals({3}), 0.0);
                    }
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, OnARealTensorTheThreeDotProductsHoldTheSameBits)
        {
            using T = typename TypeParam::Value;

            if (Core::is_complex_v<T>)
                GTEST_SKIP() << "doti is not defined for a complex tensor";

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                auto on = this->operands(mode, Core::IndexBase::Zero);
                Tensor unconjugated = this->vector_of(reals({9}));
                Tensor conjugated = this->vector_of(reals({9}));

                SparseOps ops(*this->engine);
                ops.doti(on.x, on.y, on.result);
                ops.dotui(on.x, on.y, unconjugated);
                ops.dotci(on.x, on.y, conjugated);
                this->engine->execute().value();

                const T plain = read_tensor<T>(on.result)[0];
                EXPECT_EQ(widened(plain), std::complex<double>(260.0, 0.0));
                EXPECT_TRUE(this->same_bits(plain, read_tensor<T>(unconjugated)[0]));
                EXPECT_TRUE(this->same_bits(plain, read_tensor<T>(conjugated)[0]));
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, GthrCopiesTheElementsAtThePositionsAndLeavesY)
        {
            using T = typename TypeParam::Value;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    SparseOps(*this->engine).gthr(on.y, on.x);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.x.values), reals({7, 2, 5}), 0.0);
                    this->expect_near(read_tensor<T>(on.y), reals({1, 2, 3, 4, 5, 6, 7, 8}), 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, GthrzAlsoZeroesTheGatheredElements)
        {
            using T = typename TypeParam::Value;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    SparseOps(*this->engine).gthrz(on.y, on.x);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.x.values), reals({7, 2, 5}), 0.0);
                    this->expect_near(read_tensor<T>(on.y), reals({1, 0, 3, 4, 0, 6, 0, 8}), 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, SctrWritesTheEntriesAtTheirPositions)
        {
            using T = typename TypeParam::Value;

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    SparseOps(*this->engine).sctr(on.x, on.y);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.y), reals({1, 20, 3, 4, 30, 6, 10, 8}),
                                      0.0);
                    this->expect_near(read_tensor<T>(on.x.values), sample_entries(), 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, RotiRotatesTheEntriesAgainstTheirElements)
        {
            using T = typename TypeParam::Value;

            // With c = 0.25 and s = 0.5 a swap of the two gives other values at every position.
            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base);
                    SparseOps(*this->engine).roti(on.x, on.y, 0.25, 0.5);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.x.values), reals({6, 6, 10}), 0.0);
                    this->expect_near(read_tensor<T>(on.y),
                                      reals({1, -9.5, 3, 4, -13.75, 6, -3.25, 8}), 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, AStridedYIsAddressedByItsIncrement)
        {
            using T = typename TypeParam::Value;

            // The view holds 1 to 8 at the even elements of sixteen; the odd ones hold K_FILLER.
            Wide spread(static_cast<std::size_t>(2 * K_SIZE), std::complex<double>(K_FILLER, 0.0));
            Wide expected = spread;
            const Wide updated = reals({1, 42, 3, 4, 65, 6, 27, 8});
            for (std::size_t i = 0; i < updated.size(); ++i)
            {
                spread[2 * i] = std::complex<double>(static_cast<double>(i + 1), 0.0);
                expected[2 * i] = updated[i];
            }

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    this->engine->set_mode(mode);
                    const SparseVector x =
                        this->build(K_SIZE, sample_positions(), sample_entries(), base);
                    const Tensor wide = this->vector_of(spread);
                    Tensor y = this->every_second(wide);

                    SparseOps(*this->engine).axpyi(2.0, x, y);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(wide), expected, 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, AVectorWithoutEntriesLeavesYAndGivesAZeroDotProduct)
        {
            using T = typename TypeParam::Value;

            const Wide dense = counting(static_cast<std::size_t>(K_SIZE), 1.0);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                for (const Core::IndexBase base : K_BOTH_BASES)
                {
                    auto on = this->operands(mode, base, {}, Wide(), dense);
                    ASSERT_EQ(on.x.nnz(), 0);

                    // dotui serves every element type; on a real tensor it records doti.
                    SparseOps ops(*this->engine);
                    ops.axpyi(2.0, on.x, on.y);
                    ops.dotui(on.x, on.y, on.result);
                    this->engine->execute().value();

                    this->expect_near(read_tensor<T>(on.y), dense, 0.0);
                    this->expect_near(read_tensor<T>(on.result), reals({0}), 0.0);
                }
            }
        }

        TYPED_TEST(Medium_SparseLevel1Test, ALongDotProductOfIntegersIsExactInBothModes)
        {
            using T = typename TypeParam::Value;

            // Entry k is (k mod 7) - 3 at position 3 * k; element p of y is (p mod 5) - 2.
            std::vector<int64_t> positions(static_cast<std::size_t>(K_LONG_ENTRIES));
            Wide entries(positions.size());
            Wide dense(static_cast<std::size_t>(K_LONG_SIZE));
            for (std::size_t p = 0; p < dense.size(); ++p)
                dense[p] = std::complex<double>(static_cast<double>(p % 5) - 2.0, 0.0);

            double sum = 0.0;
            for (std::size_t k = 0; k < positions.size(); ++k)
            {
                positions[k] = static_cast<int64_t>(k) * K_LONG_SPACING;
                entries[k] = std::complex<double>(static_cast<double>(k % 7) - 3.0, 0.0);
                sum += entries[k].real() * dense[static_cast<std::size_t>(positions[k])].real();
            }

            // dotui serves every element type; on a real tensor it records doti.
            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                auto on = this->operands(mode, Core::IndexBase::One, positions, entries, dense);
                SparseOps(*this->engine).dotui(on.x, on.y, on.result);
                this->engine->execute().value();

                this->expect_near(read_tensor<T>(on.result), reals({sum}), 0.0);
            }
        }

        /**
         * @brief Runs the HALF update whose product a HALF accumulator would round away.
         */
        class Medium_SparseLevel1Half : public Medium_SparseLevel1Test<SparseCase<sycl::half, int64_t>>
        {

        };

        TEST_F(Medium_SparseLevel1Half, AxpyiFormsItsProductInFloat32)
        {
            // 3 * 683 - 1 is 2048; a product rounded to HALF is 2048 and leaves 2047.
            const Wide dense(static_cast<std::size_t>(K_SIZE), std::complex<double>(-1.0, 0.0));
            Wide expected = dense;
            expected[2] = std::complex<double>(2048.0, 0.0);

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                auto on = operands(mode, Core::IndexBase::Zero, {2}, reals({683}), dense);
                SparseOps(*engine).axpyi(3.0, on.x, on.y);
                engine->execute().value();

                expect_near(read_tensor<sycl::half>(on.y), expected, 0.0);
            }
        }

        /**
         * @brief Runs the level 1 routines on sparse vectors whose values are integers.
         */
        class Medium_SparseLevel1Integers : public SushiBLASTest
        {
            protected:
                /** @brief Holds the operands of one level 1 call on values of type V. */
                struct Operands
                {
                    SparseVector x; /**< Entries 10, 20, 30 at the positions 6, 1, 4. */
                    Tensor y;       /**< The dense vector 1 to 8. */
                    Tensor result;  /**< A scalar of the value dtype. */
                };

                /** @brief Sets the mode and returns operands whose values have type V. */
                template<typename V>
                [[nodiscard]] Operands operands(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    const SparseBuilder builder(engine->tensor_factory());
                    return Operands{
                        builder.vector_from_entries(K_SIZE, std::vector<int32_t>{6, 1, 4},
                                                    std::vector<V>{10, 20, 30}),
                        create_initialized_tensor<V>({K_SIZE}, {1, 2, 3, 4, 5, 6, 7, 8}),
                        engine->create_tensor({1}, sushiblas_dtype<V>())};
                }

                /** @brief Expects gthr, gthrz and sctr to move values of type V as they stand. */
                template<typename V>
                void expect_moves()
                {
                    for (const ExecutionMode mode : K_BOTH_MODES)
                    {
                        Operands gathered = operands<V>(mode);
                        SparseOps(*engine).gthr(gathered.y, gathered.x);
                        engine->execute().value();
                        EXPECT_EQ(read_tensor<V>(gathered.x.values), (std::vector<V>{7, 2, 5}));
                        EXPECT_EQ(read_tensor<V>(gathered.y),
                                  (std::vector<V>{1, 2, 3, 4, 5, 6, 7, 8}));

                        Operands zeroed = operands<V>(mode);
                        SparseOps(*engine).gthrz(zeroed.y, zeroed.x);
                        engine->execute().value();
                        EXPECT_EQ(read_tensor<V>(zeroed.x.values), (std::vector<V>{7, 2, 5}));
                        EXPECT_EQ(read_tensor<V>(zeroed.y),
                                  (std::vector<V>{1, 0, 3, 4, 0, 6, 0, 8}));

                        Operands scattered = operands<V>(mode);
                        SparseOps(*engine).sctr(scattered.x, scattered.y);
                        engine->execute().value();
                        EXPECT_EQ(read_tensor<V>(scattered.y),
                                  (std::vector<V>{1, 20, 3, 4, 30, 6, 10, 8}));
                    }
                }

                /** @brief Expects axpyi, doti and roti to refuse values of type V. */
                template<typename V>
                void expect_arithmetic_refused()
                {
                    for (const ExecutionMode mode : K_BOTH_MODES)
                    {
                        Operands on = operands<V>(mode);
                        SparseOps ops(*engine);
                        EXPECT_THROW(ops.axpyi(2.0, on.x, on.y), std::invalid_argument);
                        EXPECT_THROW(ops.doti(on.x, on.y, on.result), std::invalid_argument);
                        EXPECT_THROW(ops.roti(on.x, on.y, 0.25, 0.5), std::invalid_argument);
                    }
                }
        };

        TEST_F(Medium_SparseLevel1Integers, GatherAndScatterMoveIntegerValues)
        {
            expect_moves<int32_t>();
            expect_moves<int64_t>();
        }

        TEST_F(Medium_SparseLevel1Integers, TheArithmeticRoutinesRefuseIntegerValues)
        {
            expect_arithmetic_refused<int32_t>();
            expect_arithmetic_refused<int64_t>();
        }
    } // namespace Test
} // namespace SushiBLAS
