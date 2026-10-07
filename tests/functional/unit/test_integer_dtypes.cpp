/**************************************************************************/
/* test_integer_dtypes.cpp                                                */
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
 * @file test_integer_dtypes.cpp
 * @brief Tests which operations serve INT32 and INT64 tensors and which refuse them.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_IntegerDTypeTest : public SushiBLASTest
        {
            protected:
                /** @brief Reads a whole INT64 tensor back to the host. */
                std::vector<int64_t> read_i64(const Tensor& t)
                {
                    std::vector<int64_t> host(static_cast<size_t>(t.num_elements));
                    TestEnvironment::get_context().get_queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(int64_t)).wait();
                    return host;
                }

                /** @brief Reads a whole INT32 tensor back to the host. */
                std::vector<int32_t> read_i32(const Tensor& t)
                {
                    std::vector<int32_t> host(static_cast<size_t>(t.num_elements));
                    TestEnvironment::get_context().get_queue()
                        .memcpy(host.data(), t.data(), host.size() * sizeof(int32_t)).wait();
                    return host;
                }
        };

        /**
         * @brief Tests the element widths and names of the integer dtypes.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, ElementWidthsAndNamesAreDeclared)
        {
            EXPECT_EQ(4u, Core::bytes_per_element(Core::DataType::INT32));
            EXPECT_EQ(8u, Core::bytes_per_element(Core::DataType::INT64));
            EXPECT_STREQ("INT32", Core::to_string(Core::DataType::INT32));
            EXPECT_STREQ("INT64", Core::to_string(Core::DataType::INT64));

            EXPECT_TRUE(Core::is_integer_dtype(Core::DataType::INT32));
            EXPECT_TRUE(Core::is_integer_dtype(Core::DataType::INT64));
            EXPECT_FALSE(Core::is_integer_dtype(Core::DataType::FLOAT32));
            EXPECT_FALSE(Core::is_real_dtype(Core::DataType::INT64));
            EXPECT_FALSE(Core::is_complex_dtype(Core::DataType::INT64));

            auto t = engine->create_tensor({4}, Core::DataType::INT64);
            EXPECT_EQ(32, t.element_span() * static_cast<int64_t>(Core::bytes_per_element(t.dtype)));
        }

        /**
         * @brief Tests that constant fills an integer tensor exactly.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, ConstantFillsAnIntegerTensorExactly)
        {
            auto a = engine->create_tensor({4}, Core::DataType::INT32);
            auto b = engine->create_tensor({4}, Core::DataType::INT64);

            SushiBLAS::RandomOps(*engine).constant(a, 7.0);
            SushiBLAS::RandomOps(*engine).constant(b, 1234567890.0);
            engine->execute();

            EXPECT_EQ(std::vector<int32_t>(4, 7), read_i32(a));
            EXPECT_EQ(std::vector<int64_t>(4, 1234567890), read_i64(b));
        }

        /**
         * @brief Tests that copy moves integer elements.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, CopyMovesIntegerElements)
        {
            auto src = engine->create_tensor({3}, Core::DataType::INT64);
            auto dst = engine->create_tensor({3}, Core::DataType::INT64);

            fill_tensor(src, {10.0f, -20.0f, 30.0f});
            SushiBLAS::RandomOps(*engine).constant(dst, 0.0);
            SushiBLAS::BLASOps(*engine).copy(src, dst);
            engine->execute();

            EXPECT_EQ((std::vector<int64_t>{10, -20, 30}), read_i64(dst));
        }

        /**
         * @brief Tests that the comparisons work on integer tensors.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, ComparisonsWorkOnIntegerTensors)
        {
            auto a = engine->create_tensor({4}, Core::DataType::INT32);
            auto b = engine->create_tensor({4}, Core::DataType::INT32);
            auto r = engine->create_tensor({4}, Core::DataType::INT32);

            fill_tensor(a, {1.0f, 5.0f, 3.0f, 9.0f});
            fill_tensor(b, {2.0f, 5.0f, 1.0f, 0.0f});

            SushiBLAS::LogicOps(*engine).less(a, b, r);
            engine->execute();
            EXPECT_EQ((std::vector<int32_t>{1, 0, 0, 0}), read_i32(r));

            SushiBLAS::LogicOps(*engine).equal(a, b, r);
            engine->execute();
            EXPECT_EQ((std::vector<int32_t>{0, 1, 0, 0}), read_i32(r));

            SushiBLAS::LogicOps(*engine).greater_equal(a, b, r);
            engine->execute();
            EXPECT_EQ((std::vector<int32_t>{0, 1, 1, 1}), read_i32(r));
        }

        // The boolean family and `where`, which selects rather than compares.
        TEST_F(Unit_IntegerDTypeTest, BooleanOpsAndWhereWorkOnIntegerTensors)
        {
            auto a = engine->create_tensor({4}, Core::DataType::INT64);
            auto b = engine->create_tensor({4}, Core::DataType::INT64);
            auto r = engine->create_tensor({4}, Core::DataType::INT64);

            fill_tensor(a, {1.0f, 1.0f, 0.0f, 0.0f});
            fill_tensor(b, {1.0f, 0.0f, 1.0f, 0.0f});

            SushiBLAS::LogicOps(*engine).logical_and(a, b, r);
            engine->execute();
            EXPECT_EQ((std::vector<int64_t>{1, 0, 0, 0}), read_i64(r));

            SushiBLAS::LogicOps(*engine).logical_xor(a, b, r);
            engine->execute();
            EXPECT_EQ((std::vector<int64_t>{0, 1, 1, 0}), read_i64(r));

            auto lhs = engine->create_tensor({4}, Core::DataType::INT64);
            auto rhs = engine->create_tensor({4}, Core::DataType::INT64);
            fill_tensor(lhs, {100.0f, 200.0f, 300.0f, 400.0f});
            fill_tensor(rhs, {-1.0f, -2.0f, -3.0f, -4.0f});

            SushiBLAS::LogicOps(*engine).where(a, lhs, rhs, r);
            engine->execute();
            EXPECT_EQ((std::vector<int64_t>{100, 200, -3, -4}), read_i64(r));
        }

        // `all` and `any` reduce a truth vector, so they take integer operands too.
        TEST_F(Unit_IntegerDTypeTest, AllAndAnyReduceIntegerTruthVectors)
        {
            auto mixed = engine->create_tensor({4}, Core::DataType::INT32);
            auto ones = engine->create_tensor({4}, Core::DataType::INT32);
            auto r = engine->create_tensor({1}, Core::DataType::INT32);

            fill_tensor(mixed, {1.0f, 0.0f, 1.0f, 1.0f});
            fill_tensor(ones, {1.0f, 1.0f, 1.0f, 1.0f});

            SushiBLAS::LogicOps(*engine).all(mixed, r);
            engine->execute();
            EXPECT_EQ(0, read_i32(r)[0]);

            SushiBLAS::LogicOps(*engine).all(ones, r);
            engine->execute();
            EXPECT_EQ(1, read_i32(r)[0]);

            SushiBLAS::LogicOps(*engine).any(mixed, r);
            engine->execute();
            EXPECT_EQ(1, read_i32(r)[0]);
        }

        /**
         * @brief Tests that argmax over an integer tensor returns an INT64 index.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, ArgmaxSearchesAnIntegerTensor)
        {
            auto t = engine->create_tensor({5}, Core::DataType::INT32);
            auto r = engine->create_tensor({1}, Core::DataType::INT64);

            fill_tensor(t, {3.0f, -1.0f, 42.0f, 7.0f, 41.0f});

            SushiBLAS::ReductionOps(*engine).argmax(t, r);
            engine->execute();
            EXPECT_EQ(2, read_i64(r)[0]);

            SushiBLAS::ReductionOps(*engine).argmin(t, r);
            engine->execute();
            EXPECT_EQ(1, read_i64(r)[0]);
        }

        /**
         * @brief Tests that argmax is exact for an index past 2^24.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, ArgmaxIsExactPastTheFloatMantissa)
        {
            constexpr int64_t kN = 16777218;
            constexpr int64_t kPeak = 16777217;

            auto t = engine->create_tensor({kN});
            auto r = engine->create_tensor({1}, Core::DataType::INT64);

            SushiBLAS::RandomOps(*engine).constant(t, 1.0);
            engine->execute();

            // Plant the maximum at an index a float cannot name.
            const float peak = 2.0f;
            TestEnvironment::get_context().get_queue()
                .memcpy(t.data_as<float>() + kPeak, &peak, sizeof(float)).wait();

            SushiBLAS::ReductionOps(*engine).argmax(t, r);
            engine->execute();

            EXPECT_EQ(kPeak, read_i64(r)[0])
                << "An index stored as a float rounds to " << static_cast<int64_t>(static_cast<float>(kPeak))
                << " here, which is a different element.";
        }

        /**
         * @brief Tests that every unsupported op refuses an integer operand by name.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, UnsupportedOpsRejectIntegerOperandsByName)
        {
            GraphProbe probe;

            auto a = engine->create_tensor({4}, Core::DataType::INT32);
            auto b = engine->create_tensor({4}, Core::DataType::INT32);
            auto scalar = engine->create_tensor({1}, Core::DataType::INT32);

            // Elementwise transcendentals.
            EXPECT_THROW(
                SushiBLAS::ElementwiseOps(probe.get_recorder()).exp(a),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::ElementwiseOps(probe.get_recorder()).sqrt(a),
                std::runtime_error);

            // Non-linearities.
            EXPECT_THROW(
                SushiBLAS::NonLinearOps(probe.get_recorder()).relu(a),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::NonLinearOps(probe.get_recorder()).sigmoid(a),
                std::runtime_error);

            // Folding reductions - a sum or a norm of an index is not a thing.
            EXPECT_THROW(
                SushiBLAS::ReductionOps(probe.get_recorder()).sum(a, scalar),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::ReductionOps(probe.get_recorder()).nrm2(a, scalar),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::ReductionOps(probe.get_recorder()).mean(a, scalar),
                std::runtime_error);

            // Random distributions - no fractional part for a continuous draw to land in.
            EXPECT_THROW(
                SushiBLAS::RandomOps(probe.get_recorder(), engine->get_rng()).uniform(a, 0.0, 1.0),
                std::invalid_argument);
            EXPECT_THROW(
                SushiBLAS::RandomOps(probe.get_recorder(), engine->get_rng()).normal(a, 0.0, 1.0),
                std::invalid_argument);

            // BLAS arithmetic. `copy` is the exception and is covered above.
            EXPECT_THROW(
                SushiBLAS::BLASOps(probe.get_recorder()).axpy(1.0, a, b),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::BLASOps(probe.get_recorder()).scal(2.0, a),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::BLASOps(probe.get_recorder()).dot(a, b, scalar),
                std::runtime_error);
            EXPECT_THROW(
                SushiBLAS::BLASOps(probe.get_recorder()).nrm2(a, scalar),
                std::runtime_error);

            // Nothing above should have reached the graph.
            EXPECT_TRUE(probe.take_nodes().empty())
                << "A rejected op must throw before it records a task, not while executing one.";
        }

        /**
         * @brief Tests that the ordering comparisons still refuse complex operands.
         * @see tests/functional/unit/README.md
         */
        TEST_F(Unit_IntegerDTypeTest, OrderingComparisonsStillRejectComplex)
        {
            auto a = engine->create_tensor({4}, Core::DataType::COMPLEX32);
            auto b = engine->create_tensor({4}, Core::DataType::COMPLEX32);
            auto r = engine->create_tensor({4}, Core::DataType::COMPLEX32);

            EXPECT_THROW(SushiBLAS::LogicOps(*engine).less(a, b, r), std::runtime_error);
            EXPECT_THROW(SushiBLAS::LogicOps(*engine).greater(a, b, r), std::runtime_error);

            // Equality has no such problem and must still work.
            EXPECT_NO_THROW(SushiBLAS::LogicOps(*engine).equal(a, b, r));
            engine->execute();
        }

    } // namespace Test
} // namespace SushiBLAS
