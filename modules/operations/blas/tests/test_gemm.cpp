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
 * @file test_gemm.cpp
 * @brief Tests gemm against the reference kernel, hand-computed results and its epilogue seam.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <complex>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include "test_utils.hpp"
#include "blas_flags.hpp"
#include "graph_probe.hpp"

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
                                   Core::Layout layout = Core::Layout::RowMajor,
                                   bool transA = false, bool transB = false,
                                   double alpha = 1.0, double beta = 0.0)
                {
                    reinit_engine(layout);

                    const auto dtype = get_sushiblas_dtype<T>();

                    // op(A) is m x k and op(B) is k x n; a set flag stores the transpose.
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

                    // Both C buffers must start identical. See tests/unit/README.md.
                    seed_output(C_actual, C_expected, beta == 0.0);

                    engine->set_mode(ExecutionMode::ForceSycl);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C_actual, transpose_of(transA),
                                                     transpose_of(transB), alpha, beta);
                    engine->execute();

                    engine->set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C_expected, transpose_of(transA),
                                                     transpose_of(transB), alpha, beta);
                    engine->execute();

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
                    reinit_engine(Core::Layout::RowMajor);

                    const auto dtype = get_sushiblas_dtype<T>();

                    auto A = engine->create_tensor({batch, m, k}, dtype);
                    auto B = engine->create_tensor({batch, k, n}, dtype);
                    auto C_actual = engine->create_tensor({batch, m, n}, dtype);
                    auto C_expected = engine->create_tensor({batch, m, n}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(B);
                    seed_output(C_actual, C_expected, beta == 0.0);

                    engine->set_mode(ExecutionMode::ForceSycl);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C_actual, Core::Transpose::NoTrans,
                                                     Core::Transpose::NoTrans, alpha, beta);
                    engine->execute();

                    engine->set_mode(ExecutionMode::ForceReference);
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C_expected, Core::Transpose::NoTrans,
                                                     Core::Transpose::NoTrans, alpha, beta);
                    engine->execute();

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
                        // A read of C at beta == 0 yields NaN. See tests/unit/README.md.
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
                * The SYCL kernel accumulates in float for half and in T otherwise, the reference in double.
                * They differ only in summation order, so the gap they can open grows with
                * the length of the reduction instead of staying fixed. That only matters
                * for half, which has about three decimal digits to spend in the result.
                */
                [[nodiscard]] static double tolerance_for(Core::DataType dtype, int64_t k)
                {
                    if (dtype == Core::DataType::Half)
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
                this->test_accuracy(4, 4, 4, Core::Layout::ColumnMajor);
            }

            TYPED_TEST(Unit_GEMMTest, Rectangular)
            {
                this->test_accuracy(8, 16, 4);
            }

            TYPED_TEST(Unit_GEMMTest, RectangularColMajor)
            {
                this->test_accuracy(8, 16, 4, Core::Layout::ColumnMajor);
            }

            /**
             * @brief Tests prime extents, which leave the row, column and K tails ragged at once.
             * @see tests/unit/README.md
             */
            TYPED_TEST(Unit_GEMMTest, RaggedOnEveryAxis)
            {
                this->test_accuracy(37, 23, 13);
            }

            TYPED_TEST(Unit_GEMMTest, RaggedOnEveryAxisColMajor)
            {
                this->test_accuracy(37, 23, 13, Core::Layout::ColumnMajor);
            }

            /**
             * @brief Tests a K shorter than one local-memory K-tile under a multi-group output.
             * @see tests/unit/README.md
             */
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
                this->test_accuracy(17, 19, 11, Core::Layout::RowMajor, false, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeA)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::RowMajor, true, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeB)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::RowMajor, false, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBoth)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::RowMajor, true, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBothColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ColumnMajor, true, true, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeAColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ColumnMajor, true, false, 1.5, 0.5);
            }

            TYPED_TEST(Unit_GEMMTest, TransposeBColMajor)
            {
                this->test_accuracy(17, 19, 11, Core::Layout::ColumnMajor, false, true, 1.5, 0.5);
            }

            /**
             * @brief Tests that beta == 0 never reads C, which starts full of NaN.
             * @see tests/unit/README.md
             */
            TYPED_TEST(Unit_GEMMTest, BetaZeroNeverReadsC)
            {
                this->test_accuracy(37, 23, 13, Core::Layout::RowMajor, false, false, 1.0, 0.0);
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
             * @brief Holds the hand-computed pins and the direct calls to both kernel variants.
             * @see tests/unit/README.md
             */
            class Unit_GEMMKernelTest : public SushiBLASTest
            {
                protected:
                    /** @brief Row-major strides for an operand of leading dimension @p ld. */
                    [[nodiscard]] static Kernels::OperandAccess row_major(int64_t ld)
                    {
                        return Kernels::operand_access(Core::Layout::RowMajor,
                                                       Core::Transpose::NoTrans, ld);
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
                auto C = engine->create_tensor({2, 2}, Core::DataType::Float32);

                engine->set_mode(ExecutionMode::ForceSycl);
                SushiBLAS::BLASOps(*engine).gemm(A, B, C);
                engine->execute();

                // C = [[1+3, 2+3], [4+6, 5+6]] = [[4,5],[10,11]], row-major.
                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 5, 10, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, ColumnMajorMatchesHandComputedResult)
            {
                reinit_engine(Core::Layout::ColumnMajor);

                // The same problem, stored column-major. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 1, 0, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::Float32);

                engine->set_mode(ExecutionMode::ForceSycl);
                SushiBLAS::BLASOps(*engine).gemm(A, B, C);
                engine->execute();

                // C = [[4,5],[10,11]] again, stored column-major with ldc = 2.
                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 10, 5, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, ColumnMajorReferenceMatchesHandComputedResult)
            {
                // The oracle is pinned to the same numbers. See tests/unit/README.md.
                reinit_engine(Core::Layout::ColumnMajor);

                auto A = create_initialized_tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 1, 0, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::Float32);

                engine->set_mode(ExecutionMode::ForceReference);
                SushiBLAS::BLASOps(*engine).gemm(A, B, C);
                engine->execute();

                EXPECT_TENSOR_NEAR(C, (std::vector<float>{4, 10, 5, 11}), 1e-5);
            }

            TEST_F(Unit_GEMMKernelTest, TransposedAMatchesHandComputedResult)
            {
                // A is stored 3x2 as the transpose of op(A). See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, {1, 4, 2, 5, 3, 6});
                auto B = create_initialized_tensor<float>({3, 2}, {1, 0, 0, 1, 1, 1});
                auto C = engine->create_tensor({2, 2}, Core::DataType::Float32);

                engine->set_mode(ExecutionMode::ForceSycl);
                SushiBLAS::BLASOps(*engine).gemm(A, B, C, Core::Transpose::Trans,
                                                 Core::Transpose::NoTrans);
                engine->execute();

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
                // 2*(A*B), plus both biases, clamped at 0. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({2, 2}, {1, 2, 3, 4});
                auto B = create_initialized_tensor<float>({2, 3}, {1, 0, -1, 0, 1, -1});
                auto C = engine->create_tensor({2, 3}, Core::DataType::Float32);
                auto row_bias = create_initialized_tensor<float>({2}, {0.0f, 1.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {0.5f, -1.0f, 5.0f});

                auto& q = TestEnvironment::get_context().get_queue();
                const BiasReluEpilogue epilogue{static_cast<const float*>(row_bias.data()),
                                                static_cast<const float*>(col_bias.data())};

                Kernels::SYCL::gemm<float>(
                    q, Core::Layout::RowMajor, Core::Transpose::NoTrans,
                    Core::Transpose::NoTrans, 2, 3, 2,
                    2.0f, static_cast<const float*>(A.data()), 2, 0,
                    static_cast<const float*>(B.data()), 3, 0,
                    0.0f, static_cast<float*>(C.data()), 3, 0,
                    1, {}, epilogue).wait();

                EXPECT_TENSOR_NEAR(C, (std::vector<float>{2.5f, 3.0f, 0.0f, 7.5f, 8.0f, 0.0f}), 1e-5);
            }

            /**
             * @brief Holds the operands and the hand-computed result of the fused seam tests.
             * @see tests/unit/README.md
             */
            class Unit_GEMMSeamTest : public SushiBLASTest
            {
                protected:
                    /** @brief A = 3x2, chosen so A*B is square and not symmetric. */
                    static std::vector<float> a_values() { return {1, 0, 0, 1, 1, 1}; }

                    /** @brief B = 2x3. */
                    static std::vector<float> b_values() { return {1, 2, 3, 4, 5, 6}; }

                    /**
                     * @brief Returns the hand-computed result of the fused call.
                     * @see tests/unit/README.md
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
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto row_bias = create_initialized_tensor<float>({3}, {0.0f, -6.0f, 5.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

                const BiasReluEpilogue epilogue{row_bias.data_as<float>(),
                                                col_bias.data_as<float>()};

                engine->set_mode(ExecutionMode::ForceSycl);
                Dispatch::execute_gemm(engine->get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_bias_relu", "test.gemm_bias_relu"_op,
                                       epilogue);
                engine->execute();

                EXPECT_TENSOR_NEAR(C, expected(), 1e-5);
            }

            TEST_F(Unit_GEMMSeamTest, ExecuteGemmAppliesTheSameEpilogueOnTheReferencePath)
            {
                // The reference kernel applies the epilogue. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto row_bias = create_initialized_tensor<float>({3}, {0.0f, -6.0f, 5.0f});
                auto col_bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

                const BiasReluEpilogue epilogue{row_bias.data_as<float>(),
                                                col_bias.data_as<float>()};

                engine->set_mode(ExecutionMode::ForceReference);
                Dispatch::execute_gemm(engine->get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_bias_relu_ref", "test.gemm_bias_relu_ref"_op,
                                       epilogue);
                engine->execute();

                EXPECT_TENSOR_NEAR(C, expected(), 1e-5);
            }

            TEST_F(Unit_GEMMSeamTest, ExecuteGemmWithTheIdentityEpilogueIsPlainGemm)
            {
                // Level3::gemm is the seam with the identity. See tests/unit/README.md.
                auto A = engine->create_tensor({13, 7}, Core::DataType::Float32);
                auto B = engine->create_tensor({7, 11}, Core::DataType::Float32);
                auto via_api = engine->create_tensor({13, 11}, Core::DataType::Float32);
                auto via_seam = engine->create_tensor({13, 11}, Core::DataType::Float32);
                fill_random<float>(A);
                fill_random<float>(B);

                SushiBLAS::BLASOps(*engine).gemm(A, B, via_api, Core::Transpose::NoTrans,
                                                 Core::Transpose::NoTrans, 1.5, 0.0);
                Dispatch::execute_gemm(engine->get_recorder(), A, B, via_seam,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.5, 0.0,
                                       "test.gemm_identity", "test.gemm_identity"_op,
                                       Kernels::GemmIdentityEpilogue{});
                engine->execute();

                SB_ASSERT_TENSOR_NEAR(via_seam, via_api, 1e-6, float);
            }

            // An undeclared epilogue operand gets no edge. See tests/unit/README.md.

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
             * @brief Adds a bias and also writes its result to an auxiliary buffer.
             * @see tests/unit/README.md
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
                GraphProbe probe;

                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
                auto bias = engine->create_tensor({3}, Core::DataType::Float32);

                // The in-degree counts the bias edge alone. See tests/unit/README.md.
                SushiBLAS::ElementwiseOps(probe.get_recorder()).mul(seed, seed, bias);

                Dispatch::execute_gemm(probe.get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_declared_bias", "test.gemm_declared_bias"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

                auto nodes = probe.take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The GEMM whose epilogue dereferences `bias` must depend on the op "
                       "that wrote it. An in-degree of 0 means the declaration made through "
                       "Dispatch::gemm_epilogue never reached the task's read set.";
            }

            TEST_F(Unit_GEMMSeamTest, AnUndeclaredEpilogueOperandGetsNoEdgeAtAll)
            {
                GraphProbe probe;

                // The same functor passed bare gets no edge. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
                auto bias = engine->create_tensor({3}, Core::DataType::Float32);

                SushiBLAS::ElementwiseOps(probe.get_recorder()).mul(seed, seed, bias);

                Dispatch::execute_gemm(probe.get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_bare_bias", "test.gemm_bare_bias"_op,
                                       BiasColumnEpilogue{bias.data_as<float>()});

                auto nodes = probe.take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(0, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "A bare epilogue declares nothing, so the GEMM must be a root - "
                       "unordered against the op writing the bias it reads. If this is 1, "
                       "something is inferring the epilogue's operands, which this layer "
                       "cannot do correctly and must not appear to do.";
            }

            TEST_F(Unit_GEMMSeamTest, DeclaredOperandsOrderAcrossTwoViewsOfOneAllocation)
            {
                GraphProbe probe;

                // The declared read uses Tensor::region(). See tests/unit/README.md.
                constexpr int64_t kN = 3;

                auto storage = SushiRuntime::make_sushi<Storage>(
                    TestEnvironment::get_context(),
                    static_cast<size_t>(2 * kN) * sizeof(float),
                    SushiRuntime::API::Residency::Shared);

                Tensor seed(storage, {kN}, Core::DataType::Float32, 0);
                Tensor bias(storage, {kN}, Core::DataType::Float32, kN);

                fill_tensor(seed, {1.0f, 2.0f, 3.0f});

                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);

                SushiBLAS::ElementwiseOps(probe.get_recorder()).mul(seed, seed, bias);

                Dispatch::execute_gemm(probe.get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_arena_bias", "test.gemm_arena_bias"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));

                auto nodes = probe.take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The declared read must be described by Tensor::region() - keyed on "
                       "the allocation base and narrowed to the view's byte interval. An "
                       "in-degree of 0 means it was described by a bare pointer, which keys "
                       "on a base nothing else in this graph shares.";
            }

            TEST_F(Unit_GEMMSeamTest, ADeclaredEpilogueWriteOrdersItsReaderBehindTheGemm)
            {
                GraphProbe probe;

                // Only a declared write orders the aux reader. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto aux = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto consumer = engine->create_tensor({3, 3}, Core::DataType::Float32);

                Dispatch::execute_gemm(probe.get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_spill", "test.gemm_spill"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasSpillEpilogue{bias.data_as<float>(),
                                                             aux.data_as<float>(), 3})
                                           .reads(bias)
                                           .writes(aux));

                SushiBLAS::ElementwiseOps(probe.get_recorder()).mul(aux, aux, consumer);

                auto nodes = probe.take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "The op reading the epilogue's spill buffer must depend on the GEMM "
                       "that filled it. An in-degree of 0 means the declared write never "
                       "reached the task's write set, and the spill buys nothing.";
            }

            TEST_F(Unit_GEMMSeamTest, AnUndeclaredEpilogueWriteGetsNoEdgeAtAll)
            {
                GraphProbe probe;

                // An undeclared aux write gets no edge. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto C = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto aux = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto consumer = engine->create_tensor({3, 3}, Core::DataType::Float32);

                Dispatch::execute_gemm(probe.get_recorder(), A, B, C,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_spill_bare", "test.gemm_spill_bare"_op,
                                       BiasSpillEpilogue{bias.data_as<float>(),
                                                         aux.data_as<float>(), 3});

                SushiBLAS::ElementwiseOps(probe.get_recorder()).mul(aux, aux, consumer);

                auto nodes = probe.take_nodes();
                ASSERT_EQ(2u, nodes.size());
                EXPECT_EQ(0, nodes[1]->in_degree.load(std::memory_order_relaxed))
                    << "Nothing can see an undeclared write, so the reader must be a root.";
            }

            TEST_F(Unit_GEMMSeamTest, DeclaringOperandsChangesNoArithmetic)
            {
                // Both spellings must agree to the bit. See tests/unit/README.md.
                auto A = create_initialized_tensor<float>({3, 2}, a_values());
                auto B = create_initialized_tensor<float>({2, 3}, b_values());
                auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});
                auto bare = engine->create_tensor({3, 3}, Core::DataType::Float32);
                auto declared = engine->create_tensor({3, 3}, Core::DataType::Float32);

                engine->set_mode(ExecutionMode::ForceSycl);
                Dispatch::execute_gemm(engine->get_recorder(), A, B, bare,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_bias_bare", "test.gemm_bias_bare"_op,
                                       BiasColumnEpilogue{bias.data_as<float>()});
                Dispatch::execute_gemm(engine->get_recorder(), A, B, declared,
                                       Core::Transpose::NoTrans, Core::Transpose::NoTrans,
                                       1.0, 0.0,
                                       "test.gemm_bias_declared", "test.gemm_bias_declared"_op,
                                       Dispatch::gemm_epilogue(
                                           BiasColumnEpilogue{bias.data_as<float>()}).reads(bias));
                engine->execute();

                // A*B = [[1,2,3], [4,5,6], [5,7,9]] plus the column bias [1, -8, 3].
                EXPECT_TENSOR_NEAR(declared, (std::vector<float>{2, -6, 6,
                                                                 5, -3, 9,
                                                                 6, -1, 12}), 1e-5);
                SB_ASSERT_TENSOR_NEAR(declared, bare, 0.0, float);
            }

            TEST_F(Unit_GEMMSeamTest, DeclaringAnOperandWithNoStorageIsRefused)
            {
                // A base-less region orders against nothing. See tests/unit/README.md.
                Tensor nothing;

                EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).reads(nothing),
                             std::invalid_argument);
                EXPECT_THROW(Dispatch::gemm_epilogue(BiasColumnEpilogue{nullptr}).writes(nothing),
                             std::invalid_argument);
            }

            TEST_F(Unit_GEMMKernelTest, NoLocalVariantMatchesReferenceOnRaggedShape)
            {
                constexpr int64_t m = 37, n = 45, k = 23;
                constexpr float alpha = 1.5f;

                auto A = engine->create_tensor({m, k}, Core::DataType::Float32);
                auto B = engine->create_tensor({k, n}, Core::DataType::Float32);
                auto C = engine->create_tensor({m, n}, Core::DataType::Float32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                                Core::Transpose::NoTrans, m, n, k,
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
                // Only an OpenCL CPU runs this variant here. See tests/unit/README.md.
                constexpr int64_t m = 37, n = 45, k = 23;
                constexpr float alpha = 1.5f;

                auto& q = TestEnvironment::get_context().get_queue();
                const Device::Profile& profile = Device::profile_for(q);
                using Cfg = Kernels::SYCL::Detail::GemmLocalTile<float>;

                if (profile.local_mem_bytes < Cfg::local_mem_bytes() ||
                    profile.max_work_group_size < static_cast<std::size_t>(Cfg::K_WG_ITEMS))
                    GTEST_SKIP() << "device cannot host the local-tiled work-group";

                auto A = engine->create_tensor({m, k}, Core::DataType::Float32);
                auto B = engine->create_tensor({k, n}, Core::DataType::Float32);
                auto C = engine->create_tensor({m, n}, Core::DataType::Float32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                                Core::Transpose::NoTrans, m, n, k,
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
                // The profile never selects this path here. See tests/unit/README.md.
                constexpr int64_t m = 19, n = 21, k = 17;

                auto& q = TestEnvironment::get_context().get_queue();
                const Device::Profile& profile = Device::profile_for(q);
                using Cfg = Kernels::SYCL::Detail::GemmLocalTile<float>;

                if (profile.local_mem_bytes < Cfg::local_mem_bytes() ||
                    profile.max_work_group_size < static_cast<std::size_t>(Cfg::K_WG_ITEMS))
                    GTEST_SKIP() << "device cannot host the local-tiled work-group";

                // A is stored k x m and used transposed; B is stored n x k likewise.
                auto A = engine->create_tensor({k, m}, Core::DataType::Float32);
                auto B = engine->create_tensor({n, k}, Core::DataType::Float32);
                auto C = engine->create_tensor({m, n}, Core::DataType::Float32);
                fill_random<float>(A);
                fill_random<float>(B);

                const std::vector<float> host_a = download(A);
                const std::vector<float> host_b = download(B);
                std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
                Kernels::Reference::gemm<float>(Core::Layout::ColumnMajor, Core::Transpose::Trans,
                                                Core::Transpose::Trans, m, n, k,
                                                1.0f, host_a.data(), k, host_b.data(), n,
                                                0.0f, expected.data(), m);

                Kernels::SYCL::Detail::gemm_local_tiled<float>(
                    q, m, n, k,
                    1.0f, static_cast<const float*>(A.data()),
                    Kernels::operand_access(Core::Layout::ColumnMajor,
                                            Core::Transpose::Trans, k), 0,
                    static_cast<const float*>(B.data()),
                    Kernels::operand_access(Core::Layout::ColumnMajor,
                                            Core::Transpose::Trans, n), 0,
                    0.0f, static_cast<float*>(C.data()),
                    Kernels::operand_access(Core::Layout::ColumnMajor,
                                            Core::Transpose::NoTrans, m), 0,
                    1, {}, Kernels::GemmIdentityEpilogue{}).wait();

                EXPECT_TENSOR_NEAR(C, expected, 1e-4);
            }

            TEST_F(Unit_GEMMKernelTest, StrideTableMatchesTheStandardIdentity)
            {
                // A transpose equals a flipped layout. See tests/unit/README.md.
                constexpr int64_t ld = 7;

                const auto row_plain = Kernels::operand_access(
                    Core::Layout::RowMajor, Core::Transpose::NoTrans, ld);
                const auto row_trans = Kernels::operand_access(
                    Core::Layout::RowMajor, Core::Transpose::Trans, ld);
                const auto col_plain = Kernels::operand_access(
                    Core::Layout::ColumnMajor, Core::Transpose::NoTrans, ld);
                const auto col_trans = Kernels::operand_access(
                    Core::Layout::ColumnMajor, Core::Transpose::Trans, ld);

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
