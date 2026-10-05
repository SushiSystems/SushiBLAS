/****************************************************************/
/* test_channel_bias_epilogue.cpp                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>
#include <SushiBLAS/kernels/blas/level3/channel_bias_epilogue.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Graph::Literals;

        namespace
        {
            /**
            * @brief The bit pattern of @p value, read without type punning.
            *
            * `EXPECT_FLOAT_EQ` and a zero tolerance both compare *values*, and both
            * therefore call +0 and -0 equal. The identity check below is about the
            * generated arithmetic being unchanged, so it compares the stored bits -
            * via memcpy, because reading a float through a uint32_t lvalue is
            * undefined behaviour that this compiler is free to optimise on.
            *
            * @param value The float to reinterpret.
            * @return The IEEE-754 encoding of @p value as an unsigned integer.
            */
            [[nodiscard]] uint32_t bits_of(float value)
            {
                uint32_t pattern = 0;
                std::memcpy(&pattern, &value, sizeof(pattern));
                return pattern;
            }
        } // namespace

        /**
        * @brief The stride pair on @ref Kernels::ChannelBiasEpilogue, both ways round.
        *
        * A `(row_stride, col_stride)` that is only ever exercised at `(0, 1)` is a
        * parameter nobody proved: the column arm and a hard-coded `bias[col]` are
        * indistinguishable from outside. Each direction is therefore checked against
        * a hand-computed expectation, and C is square (3x3) so that a transposed
        * `row`/`col` is a wrong answer rather than an out-of-range read.
        *
        * Everything runs through `Dispatch::execute_gemm`, not through
        * `Kernels::SYCL::gemm`, because the seam is what a consumer can actually
        * reach and is where the epilogue is instantiated in the consumer's own
        * translation unit.
        */
        class Unit_ChannelBiasEpilogueTest : public SushiBLASTest
        {
            protected:
                /** @brief A = 3x2, chosen so A*B is square and not symmetric. */
                static std::vector<float> a_values() { return {1, 0, 0, 1, 1, 1}; }

                /** @brief B = 2x3. */
                static std::vector<float> b_values() { return {1, 2, 3, 4, 5, 6}; }

                /**
                * @brief Copies a float tensor's contents out to a host vector.
                *
                * @param t The tensor to read; must hold FLOAT32 elements.
                * @return Its elements in storage order.
                */
                [[nodiscard]] static std::vector<float> download(const Tensor& t)
                {
                    std::vector<float> host(static_cast<std::size_t>(t.num_elements));
                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memcpy(host.data(), t.data(), host.size() * sizeof(float)).wait();
                    return host;
                }
        };

        TEST_F(Unit_ChannelBiasEpilogueTest, AColumnChannelBiasMatchesAPlainRowVectorBroadcast)
        {
            // (0, 1): the channel is the column, which is what the NHWC convolution
            // GEMM [N*H*W, C_in*R*S] x [C_in*R*S, C_out] produces, and what the
            // existing GEMM_BIAS row-vector broadcast already does.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 0, 1};

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                   "test.gemm_channel_bias_col",
                                   "test.gemm_channel_bias_col"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            engine->execute().wait();

            // [[1,2,3],[4,5,6],[5,7,9]] + bias[col] = [1,-8,3] down every row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, -6, 6,
                                                      5, -3, 9,
                                                      6, -1, 12}), 1e-5);
        }

        TEST_F(Unit_ChannelBiasEpilogueTest, ARowChannelBiasBroadcastsDownTheOtherAxis)
        {
            // (1, 0): the channel is the row, which is the NCHW orientation. Same
            // bias vector as above, so the two expectations differ only in which
            // axis it was broadcast along - a swapped stride pair fails one of them.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 1, 0};

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                   "test.gemm_channel_bias_row",
                                   "test.gemm_channel_bias_row"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            engine->execute().wait();

            // [[1,2,3],[4,5,6],[5,7,9]] + bias[row] = +1, -8, +3 across each row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 3, 4,
                                                      -4, -3, -2,
                                                      8, 10, 12}), 1e-5);
        }

        TEST_F(Unit_ChannelBiasEpilogueTest, TheSameEpilogueAppliesOnTheReferencePath)
        {
            // The reference kernel is this operation's specification and invokes the
            // epilogue on the host, so an addressing arm that only works in the SYCL
            // instantiation would make VERIFY mode declare every fused GEMM wrong.
            // The row orientation is the one worth pinning here, since it is the arm
            // nothing else in the library exercises.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto bias = create_initialized_tensor<float>({3}, {1.0f, -8.0f, 3.0f});

            const Kernels::ChannelBiasEpilogue epilogue{bias.data(), 1, 0};

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                   "test.gemm_channel_bias_row_ref",
                                   "test.gemm_channel_bias_row_ref"_op,
                                   Dispatch::gemm_epilogue(epilogue).reads(bias));
            engine->execute().wait();

            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 3, 4,
                                                      -4, -3, -2,
                                                      8, 10, 12}), 1e-5);
        }

        TEST_F(Unit_ChannelBiasEpilogueTest, AZeroBiasIsTheIdentityEpilogue)
        {
            // The check that the fusion hook costs nothing when it does nothing:
            // against the identity epilogue, bit for bit rather than to a tolerance.
            // A tolerance would pass on a kernel that had silently changed its
            // accumulation order, which is exactly the regression this rules out.
            //
            // The operands are chosen so every element of A*B is strictly positive.
            // `x + 0.0f` is the identity on every float except -0.0f, which it turns
            // into +0.0f, and that one case is the only way this could fail for a
            // reason that is not a defect.
            auto A = create_initialized_tensor<float>({3, 2}, {1, 2, 3, 4, 5, 6});
            auto B = create_initialized_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
            auto fused = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto plain = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto bias = create_initialized_tensor<float>({3}, {0.0f, 0.0f, 0.0f});

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            Dispatch::execute_gemm(*engine, A, B, fused, false, false, 1.5, 0.0,
                                   "test.gemm_zero_channel_bias",
                                   "test.gemm_zero_channel_bias"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));
            Dispatch::execute_gemm(*engine, A, B, plain, false, false, 1.5, 0.0,
                                   "test.gemm_zero_identity",
                                   "test.gemm_zero_identity"_op,
                                   Kernels::GemmIdentityEpilogue{});
            engine->execute().wait();

            const std::vector<float> fused_host = download(fused);
            const std::vector<float> plain_host = download(plain);
            ASSERT_EQ(plain_host.size(), fused_host.size());

            for (std::size_t i = 0; i < fused_host.size(); ++i)
            {
                EXPECT_EQ(bits_of(plain_host[i]), bits_of(fused_host[i]))
                    << "Element " << i << ": a zero ChannelBiasEpilogue must produce the "
                       "identity kernel's result to the bit. A difference means the fused "
                       "store changed the arithmetic, so fusion is not free when unused.";
            }
        }

        TEST_F(Unit_ChannelBiasEpilogueTest, TheEpiloguesAccessSetReachesTheTask)
        {
            // The bias is written by a task recorded *before* the GEMM, so there is a
            // real producer to be ordered behind. The assertion is on `in_degree`,
            // never on an output value, for the reason test_gemm.cpp gives at length:
            // with no edge the two tasks may still come out in the right order by
            // luck, so a correct answer proves nothing about ordering. A and B have
            // no producer here, so the GEMM's in-degree counts the bias edge alone.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, Core::DataType::FLOAT32);

            engine->elementwise().mul(seed, seed, bias);

            Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                   "test.gemm_channel_bias_declared",
                                   "test.gemm_channel_bias_declared"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The GEMM whose ChannelBiasEpilogue dereferences `bias` must depend on "
                   "the op that wrote it. An in-degree of 0 means the declaration made "
                   "through Dispatch::gemm_epilogue never reached the task's read set, and "
                   "the GEMM is free to run against a bias that has not arrived.";
        }

        TEST_F(Unit_ChannelBiasEpilogueTest, TheGemmSeesTheProducedBiasValues)
        {
            // The other half of the test above: the edge exists, and the values that
            // cross it are the produced ones. On its own this proves nothing about
            // ordering - it can pass by luck without the declared read - so it is the
            // companion to the in-degree assertion, not a replacement for it.
            auto A = create_initialized_tensor<float>({3, 2}, a_values());
            auto B = create_initialized_tensor<float>({2, 3}, b_values());
            auto C = engine->create_tensor({3, 3}, Core::DataType::FLOAT32);
            auto seed = create_initialized_tensor<float>({3}, {1.0f, 2.0f, 3.0f});
            auto bias = engine->create_tensor({3}, Core::DataType::FLOAT32);

            // bias = seed*seed = [1, 4, 9], a value no uninitialised buffer would hold.
            engine->elementwise().mul(seed, seed, bias);

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            Dispatch::execute_gemm(*engine, A, B, C, false, false, 1.0, 0.0,
                                   "test.gemm_channel_bias_produced",
                                   "test.gemm_channel_bias_produced"_op,
                                   Dispatch::gemm_epilogue(
                                       Kernels::ChannelBiasEpilogue{bias.data(), 0, 1})
                                       .reads(bias));
            engine->execute().wait();

            // [[1,2,3],[4,5,6],[5,7,9]] + [1,4,9] down every row.
            EXPECT_TENSOR_NEAR(C, (std::vector<float>{2, 6, 12,
                                                      5, 9, 15,
                                                      6, 11, 18}), 1e-5);
        }

    } // namespace Test
} // namespace SushiBLAS
