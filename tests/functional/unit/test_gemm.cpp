/**************************************************************************/
/* test_gemm.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <limits>
#include <vector>
#include <complex>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>
#include <SushiBLAS/kernels/reference/blas/level3/gemm.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Graph::Literals;

        template <typename T>
        class Unit_GEMMTest : public SushiBLASTest
        {
            public:
                /**
                * @brief Differential check of the SYCL kernel against the reference one.
                *
                * The reference kernel is this operation's specification, so every shape
                * below is judged against it rather than against a stored expectation.
                * What that cannot catch is a defect the two implementations share - see
                * Unit_GEMMKernelTest for the absolute, hand-computed pins that can.
                */
                void test_accuracy(int64_t m, int64_t n, int64_t k,
                                   Core::Layout layout = Core::Layout::ROW_MAJOR,
                                   bool transA = false, bool transB = false,
                                   double alpha = 1.0, double beta = 0.0)
                {
                    reinit_engine(layout);

                    const auto dtype = get_sushiblas_dtype<T>();

                    // op(A) is m x k and op(B) is k x n, so the stored operands are the
                    // transposes of those when the corresponding flag is set.
                    const int64_t a_rows = transA ? k : m;
                    const int64_t a_cols = transA ? m : k;
                    const int64_t b_rows = transB ? n : k;
                    const int64_t b_cols = transB ? k : n;

                    auto A = engine->create_tensor({a_rows, a_cols}, dtype);
                    auto B = engine->create_tensor({b_rows, b_cols}, dtype);
                    auto C_actual = engine->create_tensor({m, n}, dtype);
                    auto C_expected = engine->create_tensor({m, n}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(B);

                    // Both C buffers must start identical, or a non-zero beta compares
                    // two different problems. They are poisoned rather than zeroed when
                    // beta is zero: BLAS says that case must not read C at all, and a
                    // NaN that leaks through fails the comparison instead of hiding in
                    // a plausible number.
                    seed_output(C_actual, C_expected, beta == 0.0);

                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    engine->blas().gemm(A, B, C_actual, transA, transB, alpha, beta);
                    engine->execute().wait();

                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    engine->blas().gemm(A, B, C_expected, transA, transB, alpha, beta);
                    engine->execute().wait();

                    SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, tolerance_for(dtype, k), T);
                }

                /**
                * @brief The same differential check over a batch of matrices.
                *
                * Distinct m, n and k give the three operands three distinct batch
                * strides, which is what makes this exercise the stride arithmetic rather
                * than a single shared step.
                */
                void test_batched(int64_t batch, int64_t m, int64_t n, int64_t k,
                                  double alpha = 1.0, double beta = 0.0)
                {
                    reinit_engine(Core::Layout::ROW_MAJOR);

                    const auto dtype = get_sushiblas_dtype<T>();

                    auto A = engine->create_tensor({batch, m, k}, dtype);
                    auto B = engine->create_tensor({batch, k, n}, dtype);
                    auto C_actual = engine->create_tensor({batch, m, n}, dtype);
                    auto C_expected = engine->create_tensor({batch, m, n}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(B);
                    seed_output(C_actual, C_expected, beta == 0.0);

                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    engine->blas().gemm(A, B, C_actual, false, false, alpha, beta);
                    engine->execute().wait();

                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    engine->blas().gemm(A, B, C_expected, false, false, alpha, beta);
                    engine->execute().wait();

                    SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, tolerance_for(dtype, k), T);
                }

            private:
                /**
                * @brief Gives both output tensors the same starting contents.
                *
                * @param actual The buffer the SYCL kernel writes.
                * @param expected The buffer the reference kernel writes.
                * @param poison Whether to fill with a value that must never survive -
                *        used when beta is zero, where reading C at all is a defect.
                */
                void seed_output(Tensor& actual, Tensor& expected, bool poison)
                {
                    if (poison)
                    {
                        // A NaN is the strongest available witness: 0*NaN is NaN, so any
                        // read of C at beta == 0 propagates and EXPECT_NEAR rejects it.
                        // half and the complex types all carry a quiet NaN.
                        const std::vector<float> nans(
                            static_cast<std::size_t>(actual.num_elements),
                            std::numeric_limits<float>::quiet_NaN());
                        fill_tensor(actual, nans);
                        fill_tensor(expected, nans);
                        return;
                    }

                    fill_random<T>(actual);

                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memcpy(expected.data(), actual.data(),
                             static_cast<std::size_t>(actual.num_elements) * sizeof(T)).wait();
                }

                /**
                * @brief A type- and reduction-length-appropriate comparison tolerance.
                *
                * Both implementations accumulate in T and differ only in summation
                * order, so the gap they can open grows with the length of the reduction
                * instead of staying fixed. That only matters for half, which has about
                * three decimal digits to spend.
                */
                [[nodiscard]] static double tolerance_for(Core::DataType dtype, int64_t k)
                {
                    if (dtype == Core::DataType::HALF)
                        return 1e-2 * static_cast<double>(1 + k / 8);
                    return 1e-4;
                }
            };

            using DataTypes = ::testing::Types<float, double, sycl::half, std::complex<float>, std::complex<double>>;
            SB_TYPED_TEST_SUITE(Unit_GEMMTest, DataTypes);

            TYPED_TEST(Unit_GEMMTest, SmallMatrix)
            {
                this->test_accuracy(4, 4, 4);
            }

            TYPED_TEST(Unit_GEMMTest, SmallMatrixColMajor)
            {
                this->test_accuracy(4, 4, 4, Core::Layout::COLUMN_MAJOR);
            }

            TYPED_TEST(Unit_GEMMTest, Rectangular)
            {
                this->test_accuracy(8, 16, 4);
            }

            TYPED_TEST(Unit_GEMMTest, RectangularColMajor)
            {
                this->test_accuracy(8, 16, 4, Core::Layout::COLUMN_MAJOR);
            }

            // Every extent is prime and none divides any work-group block, so each of
            // the row, column and K tails is ragged at the same time.
            TYPED_TEST(Unit_GEMMTest, RaggedOnEveryAxis)
            {
                this->test_accuracy(37, 23, 13);
            }

            TYPED_TEST(Unit_GEMMTest, RaggedOnEveryAxisColMajor)
            {
                this->test_accuracy(37, 23, 13, Core::Layout::COLUMN_MAJOR);
            }

            // Spans several work-groups in both output dimensions while K is shorter
            // than one local-memory K-tile, so the staged tile is mostly zero padding.
            TYPED_TEST(Unit_GEMMTest, KShorterThanOneTile)
            {
                this->test_accuracy(40, 40, 3);
            }

            TYPED_TEST(Unit_GEMMTest, SingleElement)
            {
                this->test_accuracy(1, 1, 1);
            }

            TYPED_TEST(Unit_GEMMTest, TallSkinny)
            {
                this->test_accuracy(65, 2, 33);
            }

            TYPED_TEST(Unit_GEMMTest, ShortWide)
            {
                this->test_accuracy(2, 65, 33);
            }

            TYPED_TEST(Unit_GEMMTest, NoTransposeScaled)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ROW_MAJOR, false, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeA)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ROW_MAJOR, true, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeB)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ROW_MAJOR, false, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBoth)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ROW_MAJOR, true, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBothColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::COLUMN_MAJOR, true, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeAColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::COLUMN_MAJOR, true, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::COLUMN_MAJOR, false, true, 1.5, 0.5);
            }

            // beta == 0 against a C full of NaN: standard BLAS forbids reading C at all
            // in that case, and a read would leave NaN behind for the comparison to
            // catch. Ragged on purpose so the tail path is covered too.
            TYPED_TEST(Unit_GEMMTest, BetaZeroNeverReadsC)
            {
                this->test_accuracy(37, 23, 13, Core::Layout::ROW_MAJOR, false, false, 1.0, 0.0);
            }

            TYPED_TEST(Unit_GEMMTest, Batched)
            {
                this->test_batched(4, 13, 7, 5, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, BatchedRagged)
            {
                this->test_batched(3, 33, 17, 23);
            }

            /**
            * @brief Absolute correctness pins and direct kernel-variant coverage.
            *
            * Two things the typed differential suite above structurally cannot do.
            *
            * First, hand-computed expectations. A differential test passes whenever the
            * SYCL and reference kernels agree, including when they are wrong in the same
            * way - which is exactly how an inverted COLUMN_MAJOR addressing arm survived
            * in both for as long as it did. The layout and transpose cases below are
            * therefore checked against numbers worked out by hand.
            *
            * Second, variant coverage. GEMM picks between its local-tiled and no-local
            * shapes from the device profile, so on any one machine the public API only
            * ever reaches one of them. These call both directly.
            */
            class Unit_GEMMKernelTest : public SushiBLASTest
            {
                protected:
                    /** @brief Row-major strides for an operand of leading dimension @p ld. */
                    [[nodiscard]] static Kernels::SYCL::Detail::OperandStrides row_major(int64_t ld)
                    {
                        return Kernels::SYCL::Detail::strides_for(Core::Layout::ROW_MAJOR, false, ld);
                    }

                    /** @brief Copies a tensor's contents out to a host vector. */
                    [[nodiscard]] static std::vector<float> download(const Tensor& t)
                    {
                        std::vector<float> host(static_cast<std::size_t>(t.num_elements));
                        auto& q = TestEnvironment::get_context().get_queue();
                        q.memcpy(host.data(), t.data(), host.size() * sizeof(float)).wait();
                        return host;
                    }
            };

            TEST_F(Unit_GEMMKernelTest, RowMajorMatchesHandComputedResult)
            {
                // A (2x3) = [[1,2,3],[4,5,6]], B (3x2) = [[1,0],[0,1],[1,1]].
                auto A = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 0, 1, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::FLOAT32);

                engine->set_mode(ExecutionMode::FORCE_SYCL);
                engine->blas().gemm(A, B, C);
                engine->execute().wait();

                // C = [[1+3, 2+3], [4+6, 5+6]] = [[4,5],[10,11]], row-major.
                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 5, 10, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, ColumnMajorMatchesHandComputedResult)
            {
                reinit_engine(Core::Layout::COLUMN_MAJOR);

                // The same problem as above, stored column-major: A (2x3) is
                // [1,4,2,5,3,6] with lda = 2, B (3x2) is [1,0,1,0,1,1] with ldb = 3.
                //
                // This is the case that used to be wrong. The old kernels read a
                // column-major A as if it were row-major with a column-major leading
                // dimension, which both produced the wrong element and - because that
                // leading dimension is m, not k - ran off the end of the buffer whenever
                // m > k. It survived because the reference kernel had the identical
                // defect, so the differential tests agreed.
                auto A = create_initialized_tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 1, 0, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::FLOAT32);

                engine->set_mode(ExecutionMode::FORCE_SYCL);
                engine->blas().gemm(A, B, C);
                engine->execute().wait();

                // C = [[4,5],[10,11]] again, stored column-major with ldc = 2.
                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 10, 5, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, ColumnMajorReferenceMatchesHandComputedResult)
            {
                // The oracle itself, pinned to the same numbers. Without this the SYCL
                // kernel could only ever be as right as whatever the reference does.
                reinit_engine(Core::Layout::COLUMN_MAJOR);

                auto A = create_initialized_tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 1, 0, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::FLOAT32);

                engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                engine->blas().gemm(A, B, C);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 10, 5, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, TransposedAMatchesHandComputedResult)
            {
                // op(A) is the same [[1,2,3],[4,5,6]] as above, so A is stored 3x2 as
                // its transpose: [[1,4],[2,5],[3,6]]. Pins the ROW_MAJOR+transposed arm,
                // which must address identically to COLUMN_MAJOR+untransposed.
                auto A = create_initialized_tensor<float>({3, 2}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 0, 1, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::FLOAT32);

                engine->set_mode(ExecutionMode::FORCE_SYCL);
                engine->blas().gemm(A, B, C, true, false);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 5, 10, 11}), 1e-5);
            }

            /**
            * @struct BiasReluEpilogue
            * @brief A deliberately non-trivial epilogue: two broadcasts and a clamp.
            *
            * The per-row and per-column terms are what prove both coordinates reach the
            * functor - a column bias alone would leave a wrong `row` undetected. The
            * clamp makes it non-linear, so it cannot be absorbed into alpha or beta.
            */
            struct BiasReluEpilogue
            {
                /** @brief One value per row of C, broadcast across its columns. */
                const float* row_bias;

                /** @brief One value per column of C, broadcast down its rows. */
                const float* col_bias;

                /**
                * @brief Adds both biases to @p value and clamps the result at zero.
                *
                * @tparam T The element type of C.
                * @param value The fully formed alpha*(A*B) + beta*C element.
                * @param row The element's row in C.
                * @param col The element's column in C.
                * @return max(value + row_bias[row] + col_bias[col], 0).
                */
                template<typename T>
                [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
                {
                    const T biased = value + static_cast<T>(row_bias[row])
                                           + static_cast<T>(col_bias[col]);
                    return biased > T(0) ? biased : T(0);
                }
            };

            TEST_F(Unit_GEMMKernelTest, EpilogueFusesBiasAndReluIntoTheStore)
            {
                // A (2x2) = [[1,2],[3,4]], B (2x3) = [[1,0,-1],[0,1,-1]], alpha = 2.
                //   A*B          = [[1, 2, -3], [3, 4, -7]]
                //   2*(A*B)      = [[2, 4, -6], [6, 8, -14]]
                // row_bias = [0, 1], col_bias = [0.5, -1, 5]:
                //   + biases    = [[2.5, 3, -1], [7.5, 8, -8]]
                //   relu        = [[2.5, 3,  0], [7.5, 8,  0]]
                auto A = create_initialized_tensor<float>({2, 2}, {1, 2, 3, 4});
                auto B = create_initialized_tensor<float>({2, 3}, {1, 0, -1, 0, 1, -1});
                auto C = engine->create_tensor({2, 3}, Core::DataType::FLOAT32);
                auto row_bias = create_initialized_tensor<float>({2}, {0.0f, 1.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {0.5f, -1.0f, 5.0f});

                auto& q = TestEnvironment::get_context().get_queue();
                const BiasReluEpilogue epilogue{static_cast<const float*>(row_bias.data()),
                                                static_cast<const float*>(col_bias.data())};

                Kernels::SYCL::gemm<float>(
                    q, Core::Layout::ROW_MAJOR, false, false, 2, 3, 2,
                    2.0f, static_cast<const float*>(A.data()), 2, 0,
                    static_cast<const float*>(B.data()), 3, 0,
                    0.0f, static_cast<float*>(C.data()), 3, 0,
                    1, {}, epilogue).wait();

                EXPECT_TENSOR_NEAR(C, (std::vector<float>{2.5f, 3.0f, 0.0f, 7.5f, 8.0f, 0.0f}), 1e-5);
            }

            /**
            * @brief The epilogue through the public seam, not through the kernel.
            *
            * The test above calls `Kernels::SYCL::gemm` directly, which no consumer can
            * do through the Engine API. These go through
            * `Dispatch::execute_gemm` - recorded as a task on the graph, run by
            * `Engine::execute()`, dispatched on the runtime dtype - because that is the
            * seam that is actually supported, and a hook that only works when called
            * from inside this repository is the thing this test exists to rule out.
            *
            * C is square (3x3) on purpose, and both biases are distinct. A square C is
            * what makes a transposed `row`/`col` a *wrong answer* rather than an
            * out-of-range read: with a 2x3 C, swapping the coordinates indexes a 2-entry
            * row bias with a column index and the failure is undefined behaviour instead
            * of a comparison this can report. The expected values below differ under the
            * swap (row 0 would end [2, 0, 9] rather than [2, 0, 6]), so the coordinates
            * are pinned individually and not merely jointly.
            */
            class Unit_GEMMSeamTest : public SushiBLASTest
            {
                protected:
                    /** @brief A = 3x2, chosen so A*B is square and not symmetric. */
                    static std::vector<float> a_values() { return {1, 0, 0, 1, 1, 1}; }

                    /** @brief B = 2x3. */
                    static std::vector<float> b_values() { return {1, 2, 3, 4, 5, 6}; }

                    /**
                    * @brief The hand-computed result of the fused call.
                    *
                    * A*B = [[1,2,3], [4,5,6], [5,7,9]]. With row_bias = [0, -6, 5] and
                    * col_bias = [1, -8, 3], `value + row_bias[row] + col_bias[col]` is
                    *
                    *     [[  2,  -6,   6],
                    *      [ -1,  -9,   3],
                    *      [ 11,   4,  17]]
                    *
                    * and the ReLU clamps the three negatives to zero.
                    */
                    static std::vector<float> expected()
                    {
                        return {2, 0, 6,
                                0, 0, 3,
                                11, 4, 17};
                    }
            };

            TEST_F(Unit_GEMMSeamTest, ExecuteGemmFusesBiasBroadcastAndReluOnTheSyclPath)
            {
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto row_bias = create_initialized_tensor<float>({3}, {0.0f, -6.0f, 5.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

                const BiasReluEpilogue epilogue{row_bias.data_as<float>(),
                                                col_bias.data_as<float>()};

                engine->set_mode(ExecutionMode::FORCE_SYCL);
                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_bias_relu", "test.gemm_bias_relu"_op,
                                       epilogue);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, expected(), 1e-5);
            }

            TEST_F(Unit_GEMMSeamTest, ExecuteGemmAppliesTheSameEpilogueOnTheReferencePath)
            {
                // The reference kernel is this operation's specification, so it has to
                // apply the epilogue too - otherwise VERIFY mode would declare every
                // fused GEMM wrong, and FORCE_REFERENCE would quietly compute a different
                // function from FORCE_SYCL.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto row_bias = create_initialized_tensor<float>({3}, {0.0f, -6.0f, 5.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

                const BiasReluEpilogue epilogue{row_bias.data_as<float>(),
                                                col_bias.data_as<float>()};

                engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_bias_relu_ref", "test.gemm_bias_relu_ref"_op,
                                       epilogue);
                engine->execute().wait();

                EXPECT_TENSOR_NEAR(C, expected(), 1e-5);
            }

            TEST_F(Unit_GEMMSeamTest, ExecuteGemmWithTheIdentityEpilogueIsPlainGemm)
            {
                // What `Level3::gemm` now is, asserted directly: the unfused entry point
                // and the seam with the identity must agree element for element, or the
                // delegation in src/engine/blas/level3/gemm.cpp changed behaviour.
                auto A = engine->create_tensor({13, 7}, Core::DataType::FLOAT32);
                auto B = engine->create_tensor({7, 11}, Core::DataType::FLOAT32);
                auto via_api = engine->create_tensor({13, 11}, Core::DataType::FLOAT32);
                auto via_seam = engine->create_tensor({13, 11}, Core::DataType::FLOAT32);
                fill_random<float>(A);
                fill_random<float>(B);

                engine->blas().gemm(A, B, via_api, false, false, 1.5, 0.0);
                Dispatch::execute_gemm(*engine, A, B, via_seam, false, false, 1.5, 0.0,
                                       "test.gemm_identity", "test.gemm_identity"_op,
                                       Kernels::GemmIdentityEpilogue{});
                engine->execute().wait();

                SB_ASSERT_TENSOR_NEAR(via_seam, via_api, 1e-6, float);
            }

            // ===============================================================
            // The epilogue's own operands
            // ===============================================================
            //
            // `execute_gemm` declares A, B and C. Whatever the epilogue itself
            // touches is invisible to it - the functor is an opaque callable, and
            // that opacity is the feature - so an undeclared bias is not a wrong
            // dependency edge, it is NO edge, and the graph is free to run this
            // GEMM at the same time as whatever writes the bias. Nothing throws
            // and nothing is NaN; the run simply computes with a value that had
            // not arrived yet.
            //
            // `Dispatch::gemm_epilogue(...).reads(t)/.writes(t)` is how the caller
            // hands that back, and every assertion below is on `in_degree`, never
            // on an output value - for the reason
            // Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation gives
            // at length: with no edge the two tasks MAY still come out in the right
            // order by luck, so a correct answer proves nothing about ordering. The
            // in-degree is the fact that decides it.

            /**
            * @struct BiasColumnEpilogue
            * @brief A per-column bias add - the shape a fused Linear layer uses.
            *
            * Deliberately simpler than @ref BiasReluEpilogue: these tests are about
            * what the task *declares*, not about what it computes, and a bias-only
            * epilogue is the one the ordering hazard was actually found on.
            */
            struct BiasColumnEpilogue
            {
                /** @brief One value per column of C, broadcast down its rows. */
                const float* bias;

                /**
                * @brief Adds the column's bias to @p value.
                * @tparam T The element type of C.
                * @param value The fully formed element.
                * @param col The element's column in C.
                * @return value + bias[col].
                */
                template<typename T>
                [[nodiscard]] T operator()(T value, int64_t, int64_t col) const noexcept
                {
                    return value + static_cast<T>(bias[col]);
                }
            };

            /**
            * @struct BiasSpillEpilogue
            * @brief A bias add that also spills its result to an auxiliary buffer.
            *
            * The shape cuBLASLt calls `GELU_AUX_BIAS`: the pre-activation the
            * backward pass has to differentiate at, written out on the way past so
            * it still exists after the fused store. No activation is applied - what
            * is under test is whether the seam can express an auxiliary *write* at
            * all, which is the thing that decides whether such an epilogue can ever
            * be safe, not any particular activation's arithmetic.
            */
            struct BiasSpillEpilogue
            {
                /** @brief One value per column of C. */
                const float* bias;

                /** @brief Row-major spill buffer, the same extents as C. */
                float* aux;

                /** @brief C's column count, so the epilogue can address @ref aux. */
                int64_t columns;

                /**
                * @brief Adds the bias, spills the result and returns it.
                * @tparam T The element type of C.
                * @param value The fully formed element.
                * @param row The element's row in C.
                * @param col The element's column in C.
                * @return value + bias[col].
                */
                template<typename T>
                T operator()(T value, int64_t row, int64_t col) const
                {
                    const T pre = value + static_cast<T>(bias[col]);
                    aux[row * columns + col] = static_cast<float>(pre);
                    return pre;
                }
            };

            TEST_F(Unit_GEMMSeamTest, ADeclaredEpilogueReadIsOrderedBehindItsProducer)
            {
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
                auto bias = engine->create_tensor({3}, Core::DataType::FLOAT32);

                // The bias is written inside the same graph, which is the case that
                // matters: a parameter an optimizer rewrites within one compiled
                // plan looks exactly like this. A and B have no producer here, so
                // the GEMM's in-degree counts the bias edge and nothing else.
                engine->elementwise().mul(seed, seed, bias);

                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_declared_bias", "test.gemm_declared_bias"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The GEMM whose epilogue dereferences `bias` must depend on the op "
                       "that wrote it. An in-degree of 0 means the declaration made through "
                       "Dispatch::gemm_epilogue never reached the task's read set.";
            }

            TEST_F(Unit_GEMMSeamTest, AnUndeclaredEpilogueOperandGetsNoEdgeAtAll)
            {
                // The control, and the whole reason the test above is worth writing:
                // the same graph with the same functor, passed bare. This is what
                // every fused GEMM in this library did before gemm_epilogue existed,
                // and it is asserted rather than merely described because a fix whose
                // test would also pass without the fix is not a test.
                //
                // It is also the compatibility pin: a bare epilogue must go on
                // producing exactly this access set, so nothing that never declared
                // anything changes behaviour.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
                auto bias = engine->create_tensor({3}, Core::DataType::FLOAT32);

                engine->elementwise().mul(seed, seed, bias);

                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_bare_bias", "test.gemm_bare_bias"_op,
                                       BiasColumnEpilogue{bias.data_as<float>()});

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(0, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "A bare epilogue declares nothing, so the GEMM must be a root - "
                       "unordered against the op writing the bias it reads. If this is 1, "
                       "something is inferring the epilogue's operands, which this layer "
                       "cannot do correctly and must not appear to do.";
            }

            TEST_F(Unit_GEMMSeamTest, DeclaredOperandsOrderAcrossTwoViewsOfOneAllocation)
            {
                // The bare-pointer hazard, on this seam. A region built from `data()`
                // widens to the whole allocation of whatever it points at, so a view
                // at a non-zero offset and its own arena look like two unrelated
                // resources, overlaps() says false, and the tracker emits no edge at
                // all - not a wrong edge, none. That is what cost SushiAI's first
                // arena-packed training run, and it is why
                // GemmEpilogue::reads takes a Tensor and calls Tensor::region()
                // itself rather than accepting a region a caller built.
                constexpr int64_t kN = 3;

                auto storage = SushiRuntime::make_sushi<Storage>(
                    TestEnvironment::get_context().get_allocator(),
                    static_cast<size_t>(2 * kN) * sizeof(float),
                    SushiRuntime::Memory::Strategies::Shared::instance());

                Tensor seed(storage, {kN}, Core::DataType::FLOAT32, 0);
                Tensor bias(storage, {kN}, Core::DataType::FLOAT32, kN);

                fill_tensor(seed, {1.0f, 2.0f, 3.0f});

                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);

                engine->elementwise().mul(seed, seed, bias);

                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_arena_bias", "test.gemm_arena_bias"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The declared read must be described by Tensor::region() - keyed on "
                       "the allocation base and narrowed to the view's byte interval. An "
                       "in-degree of 0 means it was described by a bare pointer, which keys "
                       "on a base nothing else in this graph shares.";
            }

            TEST_F(Unit_GEMMSeamTest, ADeclaredEpilogueWriteOrdersItsReaderBehindTheGemm)
            {
                // The direction that decides whether an activation can ever be fused
                // into a *training* forward pass: the epilogue spills its
                // pre-activation, and whatever differentiates against that spill has
                // to be ordered behind the GEMM that produced it. Nothing else in the
                // task's access set mentions the aux buffer, so this edge exists only
                // if the write was declared.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto aux = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto consumer = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);

                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_spill", "test.gemm_spill"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasSpillEpilogue{bias.data_as<float>(),
                                                             aux.data_as<float>(), 3})
                                           .reads(bias)
                                           .writes(aux));

                engine->elementwise().mul(aux, aux, consumer);

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The op reading the epilogue's spill buffer must depend on the GEMM "
                       "that filled it. An in-degree of 0 means the declared write never "
                       "reached the task's write set, and the spill buys nothing.";
            }

            TEST_F(Unit_GEMMSeamTest, AnUndeclaredEpilogueWriteGetsNoEdgeAtAll)
            {
                // The control for the test above, and the shape of the bug an aux
                // buffer would otherwise have: the epilogue really does write `aux`,
                // the reader really does read it, and without the declaration the two
                // are unordered.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto aux = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto consumer = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);

                Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                       "test.gemm_spill_bare", "test.gemm_spill_bare"_op,
                                       BiasSpillEpilogue{bias.data_as<float>(),
                                                         aux.data_as<float>(), 3});

                engine->elementwise().mul(aux, aux, consumer);

                auto nodes = engine->get_graph().take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(0, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "Nothing can see an undeclared write, so the reader must be a root.";
            }

            TEST_F(Unit_GEMMSeamTest, DeclaringOperandsChangesNoArithmetic)
            {
                // The declaration is bookkeeping on the host: the bundle is unwrapped
                // before the closure is built, so the functor that reaches the kernel
                // is the same object either way and the two spellings must agree to
                // the bit. If they do not, the unwrapping copied something it should
                // not have - or failed to.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto bare = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
                auto declared = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);

                engine->set_mode(ExecutionMode::FORCE_SYCL);
                Dispatch::execute_gemm(*engine, A, B, bare, false, false, 1.0, 0.0,
                                       "test.gemm_bias_bare", "test.gemm_bias_bare"_op,
                                       BiasColumnEpilogue{bias.data_as<float>()});
                Dispatch::execute_gemm(*engine, A, B, declared, false, false, 1.0, 0.0,
                                       "test.gemm_bias_declared", "test.gemm_bias_declared"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));
                engine->execute().wait();

                // A*B = [[1,2,3], [4,5,6], [5,7,9]] plus the column bias [1, -8, 3].
                EXPECT_TENSOR_NEAR(declared, (std::vector<float>{2, -6, 6,
                                                                 5, -3, 9,
                                                                 6, -1, 12}), 1e-5);
                SB_ASSERT_TENSOR_NEAR(declared, bare, 0.0, float);
            }

            TEST_F(Unit_GEMMSeamTest, DeclaringAnOperandWithNoStorageIsRefused)
            {
                // A storage-less tensor yields a base-less region, which the tracker
                // orders against nothing. Accepting it would let a declaration compile,
                // read like a fix in review, and do nothing at all - which is the exact
                // failure mode declaring operands exists to remove.
                Tensor nothing;

                EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).reads(nothing),
                             std::runtime_error);
                EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).writes(nothing),
                             std::runtime_error);
            }

            TEST_F(Unit_GEMMKernelTest, NoLocalVariantMatchesReferenceOnRaggedShape)
            {
                constexpr int64_t m = 37, n = 45, k = 23;
                constexpr float alpha = 1.5f;

                auto A = engine->create_tensor({m, k}, Core::DataType::FLOAT32);
                auto B = engine->create_tensor({k, n}, Core::DataType::FLOAT32);
                auto C = engine->create_tensor({m, n}, Core::DataType::FLOAT32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::ROW_MAJOR, false, false, m, n, k,
                                                alpha, host_a.data(), k, host_b.data(), n,
                                                0.0f, expected.data(), n);

                auto& q = TestEnvironment::get_context().get_queue();
                Kernels::SYCL::Detail::gemm_no_local<float>(
                    q, Device::profile_for(q), m, n, k,
                    alpha, static_cast<const float*>(A.data()), row_major(k), 0,
                    static_cast<const float*>(B.data()), row_major(n), 0,
                    0.0f, static_cast<float*>(C.data()), row_major(n), 0,
                    1, {}, Kernels::GemmIdentityEpilogue{}).wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-4);
            }

            TEST_F(Unit_GEMMKernelTest, LocalTiledVariantMatchesReferenceOnRaggedShape)
            {
                // The variant a GPU would take. This machine has no GPU - the only
                // reachable device is an OpenCL CPU - so what this proves is that the
                // kernel compiles, launches, synchronises and computes the right answer,
                // on a device whose "local memory" is really just cache. It is not
                // evidence about performance on real scratchpad hardware, and it is not
                // a substitute for running it on one.
                constexpr int64_t m = 37, n = 45, k = 23;
                constexpr float alpha = 1.5f;

                auto& q = TestEnvironment::get_context().get_queue();
                const Device::Profile& profile = Device::profile_for(q);
                using Cfg = Kernels::SYCL::Detail::GemmLocalTile<float>;

                if (profile.local_mem_bytes < Cfg::local_mem_bytes() ||
                    profile.max_work_group_size < static_cast<std::size_t>(Cfg::wg_items))
                    GTEST_SKIP() << "device cannot host the local-tiled work-group";

                auto A = engine->create_tensor({m, k}, Core::DataType::FLOAT32);
                auto B = engine->create_tensor({k, n}, Core::DataType::FLOAT32);
                auto C = engine->create_tensor({m, n}, Core::DataType::FLOAT32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::ROW_MAJOR, false, false, m, n, k,
                                                alpha, host_a.data(), k, host_b.data(), n,
                                                0.0f, expected.data(), n);

                Kernels::SYCL::Detail::gemm_local_tiled<float>(
                    q, m, n, k,
                    alpha, static_cast<const float*>(A.data()), row_major(k), 0,
                    static_cast<const float*>(B.data()), row_major(n), 0,
                    0.0f, static_cast<float*>(C.data()), row_major(n), 0,
                    1, {}, Kernels::GemmIdentityEpilogue{}).wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-4);
            }

            TEST_F(Unit_GEMMKernelTest, LocalTiledVariantHandlesColumnMajorAndTransposes)
            {
                // Same variant, driven through the stride pair rather than the identity
                // case, so the layout/transpose hoisting is exercised on the path the
                // profile will not select here.
                constexpr int64_t m = 19, n = 21, k = 17;

                auto& q = TestEnvironment::get_context().get_queue();
                const Device::Profile& profile = Device::profile_for(q);
                using Cfg = Kernels::SYCL::Detail::GemmLocalTile<float>;

                if (profile.local_mem_bytes < Cfg::local_mem_bytes() ||
                    profile.max_work_group_size < static_cast<std::size_t>(Cfg::wg_items))
                    GTEST_SKIP() << "device cannot host the local-tiled work-group";

                // A is stored k x m and used transposed; B is stored n x k likewise.
                auto A = engine->create_tensor({k, m}, Core::DataType::FLOAT32);
                auto B = engine->create_tensor({n, k}, Core::DataType::FLOAT32);
                auto C = engine->create_tensor({m, n}, Core::DataType::FLOAT32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::COLUMN_MAJOR, true, true, m, n, k,
                                                1.0f, host_a.data(), k, host_b.data(), n,
                                                0.0f, expected.data(), m);

                namespace Kd = Kernels::SYCL::Detail;
                Kernels::SYCL::Detail::gemm_local_tiled<float>(
                    q, m, n, k,
                    1.0f, static_cast<const float*>(A.data()),
                    Kd::strides_for(Core::Layout::COLUMN_MAJOR, true, k), 0,
                    static_cast<const float*>(B.data()),
                    Kd::strides_for(Core::Layout::COLUMN_MAJOR, true, n), 0,
                    0.0f, static_cast<float*>(C.data()),
                    Kd::strides_for(Core::Layout::COLUMN_MAJOR, false, m), 0,
                    1, {}, Kernels::GemmIdentityEpilogue{}).wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-4);
            }

            TEST_F(Unit_GEMMKernelTest, StrideTableMatchesTheStandardIdentity)
            {
                // Transposing an operand and flipping its layout are the same operation.
                // Cheap to state, and it is the check that would have caught the
                // inverted COLUMN_MAJOR arm the moment it was written.
                namespace Kd = Kernels::SYCL::Detail;
                constexpr int64_t ld = 7;

                const auto row_plain = Kd::strides_for(Core::Layout::ROW_MAJOR, false, ld);
                const auto row_trans = Kd::strides_for(Core::Layout::ROW_MAJOR, true, ld);
                const auto col_plain = Kd::strides_for(Core::Layout::COLUMN_MAJOR, false, ld);
                const auto col_trans = Kd::strides_for(Core::Layout::COLUMN_MAJOR, true, ld);

                EXPECT_EQ(row_plain.row, ld);
                EXPECT_EQ(row_plain.col, 1);
                EXPECT_EQ(row_trans.row, 1);
                EXPECT_EQ(row_trans.col, ld);

                EXPECT_EQ(col_plain.row, row_trans.row);
                EXPECT_EQ(col_plain.col, row_trans.col);
                EXPECT_EQ(col_trans.row, row_plain.row);
                EXPECT_EQ(col_trans.col, row_plain.col);
            }

    } // namespace Test
} // namespace SushiBLAS::Test
