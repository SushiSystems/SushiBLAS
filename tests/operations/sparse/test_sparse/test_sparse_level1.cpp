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
 * @file test_sparse_level1.cpp
 * @brief Tests the sparse-dense level 1 routines of SparseOps against values worked out by hand.
 * @author Mustafa Garip
 */

#include <array>
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
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>
#include "support/forced_mode_checks.hpp"
#include "support/refusal_naming.hpp"
#include "support/sparse_test_sweeps.hpp"

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

            /** @brief Returns the one entry 1 + 2i the complex dot products multiply. */
            [[nodiscard]] Wide complex_entry()
            {
                return {{1.0, 2.0}};
            }

            /** @brief Returns the dense vector 1 to 8 with 3 + 4i at position 3. */
            [[nodiscard]] Wide dense_with_a_complex_element()
            {
                Wide dense = counting(static_cast<std::size_t>(K_SIZE), 1.0);
                dense[3] = {3.0, 4.0};
                return dense;
            }

            /** @brief Holds a view of the even elements of sixteen and what axpyi leaves there. */
            struct StridedCase
            {
                Wide spread;   /**< Sixteen elements: 1 to 8 at the even ones, K_FILLER between. */
                Wide expected; /**< The same after y += 2 * x for the sample x. */
            };

            /** @brief Returns the sixteen elements of a strided y and what axpyi leaves in them. */
            [[nodiscard]] StridedCase strided_case()
            {
                StridedCase built;
                built.spread.assign(static_cast<std::size_t>(2 * K_SIZE),
                                    std::complex<double>(K_FILLER, 0.0));
                built.expected = built.spread;
                const Wide updated = reals({1, 42, 3, 4, 65, 6, 27, 8});
                for (std::size_t i = 0; i < updated.size(); ++i)
                {
                    built.spread[2 * i] = std::complex<double>(static_cast<double>(i + 1), 0.0);
                    built.expected[2 * i] = updated[i];
                }
                return built;
            }

            /** @brief Holds the operands and the sum of the long dot product. */
            struct LongDotCase
            {
                std::vector<int64_t> positions; /**< Zero-based, one every K_LONG_SPACING. */
                Wide entries;                   /**< Entry k is (k mod 7) - 3. */
                Wide dense;                     /**< Element p is (p mod 5) - 2. */
                double sum;                     /**< The exact dot product. */
            };

            /** @brief Returns the long dot product: entry k at position 3 * k against element p. */
            [[nodiscard]] LongDotCase long_dot_case()
            {
                LongDotCase built;
                built.positions.resize(static_cast<std::size_t>(K_LONG_ENTRIES));
                built.entries.resize(built.positions.size());
                built.dense.resize(static_cast<std::size_t>(K_LONG_SIZE));
                for (std::size_t p = 0; p < built.dense.size(); ++p)
                    built.dense[p] = std::complex<double>(static_cast<double>(p % 5) - 2.0, 0.0);

                built.sum = 0.0;
                for (std::size_t k = 0; k < built.positions.size(); ++k)
                {
                    built.positions[k] = static_cast<int64_t>(k) * K_LONG_SPACING;
                    built.entries[k] = std::complex<double>(static_cast<double>(k % 7) - 3.0, 0.0);
                    built.sum += built.entries[k].real() *
                                 built.dense[static_cast<std::size_t>(built.positions[k])].real();
                }
                return built;
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
                    set_mode(mode);
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

                /** @brief Expects every element to equal its host expectation exactly. */
                static void expect_equal(const std::vector<T>& actual, const Wide& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        const std::complex<double> got = widened(actual[i]);
                        EXPECT_EQ(got.real(), expected[i].real()) << "element " << i;
                        EXPECT_EQ(got.imag(), expected[i].imag()) << "element " << i;
                    }
                }

                /** @brief Returns whether two values of T hold the same bits. */
                [[nodiscard]] static bool same_bits(const T& a, const T& b)
                {
                    return std::memcmp(&a, &b, sizeof(T)) == 0;
                }
        };

        /**
         * @brief The level 1 fixture for the routines defined on a real tensor; a complex element
         *        type skips every test of it.
         */
        template<typename TypeParam>
        class Medium_SparseLevel1RealTest : public Medium_SparseLevel1Test<TypeParam>
        {
            protected:
                /** @brief Skips the test when the element type is complex. */
                void SetUp() override
                {
                    Medium_SparseLevel1Test<TypeParam>::SetUp();
                    if (Core::is_complex_v<typename TypeParam::Value>)
                        GTEST_SKIP() << "the routine under test is defined for a real tensor";
                }
        };

        /**
         * @brief The level 1 fixture for the routines that read the imaginary part; a real element
         *        type skips every test of it.
         */
        template<typename TypeParam>
        class Medium_SparseLevel1ComplexTest : public Medium_SparseLevel1Test<TypeParam>
        {
            protected:
                /** @brief Skips the test when the element type is real. */
                void SetUp() override
                {
                    Medium_SparseLevel1Test<TypeParam>::SetUp();
                    if (!Core::is_complex_v<typename TypeParam::Value>)
                        GTEST_SKIP() << "the routine under test reads an imaginary part";
                }
        };

        SB_TYPED_TEST_SUITE(Medium_SparseLevel1Test, SparseCases);
        SB_TYPED_TEST_SUITE(Medium_SparseLevel1RealTest, SparseCases);
        SB_TYPED_TEST_SUITE(Medium_SparseLevel1ComplexTest, SparseCases);

        TYPED_TEST(Medium_SparseLevel1Test, AxpyiAddsTheScaledEntriesAtTheirPositions)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                SparseOps(this->recorder()).axpyi(2.0, on.x, on.y);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.y), reals({1, 42, 3, 4, 65, 6, 27, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1RealTest, DotiSumsTheProductsOfARealTensor)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                // 10 * 7 + 20 * 2 + 30 * 5.
                SparseOps(this->recorder()).doti(on.x, on.y, on.result);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.result), reals({260}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1ComplexTest, DotiRefusesAComplexTensorAndNamesDotui)
        {
            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);

                expect_refusal_naming(
                    [&] { SparseOps(this->recorder()).doti(on.x, on.y, on.result); }, "dotui");
            });
        }

        TYPED_TEST(Medium_SparseLevel1ComplexTest, DotuiMultipliesAsStored)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base, {3}, complex_entry(),
                                         dense_with_a_complex_element());

                SparseOps(this->recorder()).dotui(on.x, on.y, on.result);
                this->recording->run().value();

                // (1 + 2i) * (3 + 4i).
                this->expect_equal(read_tensor<T>(on.result), {{-5.0, 10.0}});
            });
        }

        TYPED_TEST(Medium_SparseLevel1ComplexTest, DotciConjugatesX)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base, {3}, complex_entry(),
                                         dense_with_a_complex_element());

                SparseOps(this->recorder()).dotci(on.x, on.y, on.result);
                this->recording->run().value();

                // (1 - 2i) * (3 + 4i).
                this->expect_equal(read_tensor<T>(on.result), {{11.0, -2.0}});
            });
        }

        TYPED_TEST(Medium_SparseLevel1RealTest, OnARealTensorTheThreeDotProductsHoldTheSameBits)
        {
            using T = typename TypeParam::Value;

            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode, Core::IndexBase::Zero);
                Tensor unconjugated = this->vector_of(reals({9}));
                Tensor conjugated = this->vector_of(reals({9}));

                SparseOps ops(this->recorder());
                ops.doti(on.x, on.y, on.result);
                ops.dotui(on.x, on.y, unconjugated);
                ops.dotci(on.x, on.y, conjugated);
                this->recording->run().value();

                const T plain = read_tensor<T>(on.result)[0];
                EXPECT_EQ(widened(plain), std::complex<double>(260.0, 0.0));
                EXPECT_TRUE(this->same_bits(plain, read_tensor<T>(unconjugated)[0]));
                EXPECT_TRUE(this->same_bits(plain, read_tensor<T>(conjugated)[0]));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, GthrCopiesTheElementsAtThePositionsAndLeavesY)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                SparseOps(this->recorder()).gthr(on.y, on.x);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.x.values), reals({7, 2, 5}));
                this->expect_equal(read_tensor<T>(on.y), reals({1, 2, 3, 4, 5, 6, 7, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, GthrzAlsoZeroesTheGatheredElements)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                SparseOps(this->recorder()).gthrz(on.y, on.x);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.x.values), reals({7, 2, 5}));
                this->expect_equal(read_tensor<T>(on.y), reals({1, 0, 3, 4, 0, 6, 0, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, SctrWritesTheEntriesAtTheirPositions)
        {
            using T = typename TypeParam::Value;

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                SparseOps(this->recorder()).sctr(on.x, on.y);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.y), reals({1, 20, 3, 4, 30, 6, 10, 8}));
                this->expect_equal(read_tensor<T>(on.x.values), sample_entries());
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, RotiRotatesTheEntriesAgainstTheirElements)
        {
            using T = typename TypeParam::Value;

            // With c = 0.25 and s = 0.5 a swap of the two gives other values at every position.
            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base);
                SparseOps(this->recorder()).roti(on.x, on.y, 0.25, 0.5);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.x.values), reals({6, 6, 10}));
                this->expect_equal(read_tensor<T>(on.y),
                                   reals({1, -9.5, 3, 4, -13.75, 6, -3.25, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, AStridedYIsAddressedByItsIncrement)
        {
            using T = typename TypeParam::Value;

            // The view holds 1 to 8 at the even elements of sixteen; the odd ones hold K_FILLER.
            const StridedCase strided = strided_case();

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                this->set_mode(mode);
                const SparseVector x =
                    this->build(K_SIZE, sample_positions(), sample_entries(), base);
                const Tensor wide = this->vector_of(strided.spread);
                Tensor y = this->every_second(wide);

                SparseOps(this->recorder()).axpyi(2.0, x, y);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(wide), strided.expected);
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, AxpyiOnAVectorWithoutEntriesLeavesY)
        {
            using T = typename TypeParam::Value;

            const Wide dense = counting(static_cast<std::size_t>(K_SIZE), 1.0);

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base, {}, Wide(), dense);
                ASSERT_EQ(on.x.nnz(), 0);

                SparseOps(this->recorder()).axpyi(2.0, on.x, on.y);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.y), dense);
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, DotuiOfAVectorWithoutEntriesIsZero)
        {
            using T = typename TypeParam::Value;

            const Wide dense = counting(static_cast<std::size_t>(K_SIZE), 1.0);

            in_each_mode_and_base([&](ExecutionMode mode, Core::IndexBase base) {
                auto on = this->operands(mode, base, {}, Wide(), dense);
                ASSERT_EQ(on.x.nnz(), 0);

                // dotui serves every element type; on a real tensor it records doti.
                SparseOps(this->recorder()).dotui(on.x, on.y, on.result);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.result), reals({0}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Test, ALongDotProductOfIntegersIsExactInBothModes)
        {
            using T = typename TypeParam::Value;

            // Entry k is (k mod 7) - 3 at position 3 * k; element p of y is (p mod 5) - 2.
            const LongDotCase built = long_dot_case();

            // dotui serves every element type; on a real tensor it records doti.
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode, Core::IndexBase::One, built.positions, built.entries,
                                         built.dense);
                SparseOps(this->recorder()).dotui(on.x, on.y, on.result);
                this->recording->run().value();

                this->expect_equal(read_tensor<T>(on.result), reals({built.sum}));
            });
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

            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = operands(mode, Core::IndexBase::Zero, {2}, reals({683}), dense);
                SparseOps(recorder()).axpyi(3.0, on.x, on.y);
                recording->run().value();

                expect_equal(read_tensor<sycl::half>(on.y), expected);
            });
        }

        /**
         * @brief Runs the level 1 routines on sparse vectors whose values are integers of type V.
         */
        template<typename V>
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
                [[nodiscard]] Operands operands(ExecutionMode mode)
                {
                    set_mode(mode);
                    const SparseBuilder builder(engine->tensor_factory());
                    return Operands{
                        builder.vector_from_entries(K_SIZE, std::vector<int32_t>{6, 1, 4},
                                                    std::vector<V>{10, 20, 30}),
                        create_initialized_tensor<V>({K_SIZE}, {1, 2, 3, 4, 5, 6, 7, 8}),
                        engine->create_tensor({1}, sushiblas_dtype<V>())};
                }
        };

        /// @brief The two integer element types of a sparse vector.
        using IntegerValues = ::testing::Types<int32_t, int64_t>;

        SB_TYPED_TEST_SUITE(Medium_SparseLevel1Integers, IntegerValues);

        TYPED_TEST(Medium_SparseLevel1Integers, GthrMovesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);
                SparseOps(this->recorder()).gthr(on.y, on.x);
                this->recording->run().value();

                EXPECT_EQ(read_tensor<TypeParam>(on.x.values), (std::vector<TypeParam>{7, 2, 5}));
                EXPECT_EQ(read_tensor<TypeParam>(on.y),
                          (std::vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Integers, GthrzMovesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);
                SparseOps(this->recorder()).gthrz(on.y, on.x);
                this->recording->run().value();

                EXPECT_EQ(read_tensor<TypeParam>(on.x.values), (std::vector<TypeParam>{7, 2, 5}));
                EXPECT_EQ(read_tensor<TypeParam>(on.y),
                          (std::vector<TypeParam>{1, 0, 3, 4, 0, 6, 0, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Integers, SctrMovesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);
                SparseOps(this->recorder()).sctr(on.x, on.y);
                this->recording->run().value();

                EXPECT_EQ(read_tensor<TypeParam>(on.y),
                          (std::vector<TypeParam>{1, 20, 3, 4, 30, 6, 10, 8}));
            });
        }

        TYPED_TEST(Medium_SparseLevel1Integers, AxpyiRefusesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);

                EXPECT_THROW(SparseOps(this->recorder()).axpyi(2.0, on.x, on.y),
                             std::invalid_argument);
            });
        }

        TYPED_TEST(Medium_SparseLevel1Integers, DotiRefusesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);

                EXPECT_THROW(SparseOps(this->recorder()).doti(on.x, on.y, on.result),
                             std::invalid_argument);
            });
        }

        TYPED_TEST(Medium_SparseLevel1Integers, RotiRefusesIntegerValues)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                auto on = this->operands(mode);

                EXPECT_THROW(SparseOps(this->recorder()).roti(on.x, on.y, 0.25, 0.5),
                             std::invalid_argument);
            });
        }
    } // namespace Test
} // namespace SushiBLAS
