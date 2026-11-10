/**************************************************************************/
/* test_syr2k_refusals.cpp                                                */
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
 * @file test_syr2k_refusals.cpp
 * @brief Tests that syr2k and her2k refuse each malformed call with std::invalid_argument and
 *        record no task, on host tensors and a HostGraphProbe.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <complex>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/graph_probe/host_graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief A scalar with a non-zero imaginary part, which a real tensor refuses.
            const BLASScalar K_COMPLEX_SCALAR = std::complex<double>(1.0, 2.0);

            /** @brief Holds the well-formed operands a refusal starts from. */
            struct RankTwoOperands
            {
                Tensor A;         /**< A 3 x 2 FLOAT32 matrix. */
                Tensor B;         /**< A 3 x 2 FLOAT32 matrix. */
                Tensor C;         /**< A 3 x 3 FLOAT32 matrix. */
                Tensor complex_A; /**< A 3 x 2 COMPLEX32 matrix. */
                Tensor complex_B; /**< A 3 x 2 COMPLEX32 matrix. */
                Tensor complex_C; /**< A 3 x 3 COMPLEX32 matrix. */
                Tensor tall;      /**< A 4 x 2 FLOAT32 matrix. */
            };

            /** @brief Records one malformed call on ops; it must throw. */
            using Malformed = TaskHandle (*)(SushiBLAS::BLASOps& ops, Engine& engine,
                                             RankTwoOperands& operands);

            /** @brief Names a malformed call and holds the function that records it. */
            struct Refusal
            {
                const char* name; /**< The row name: what is wrong with the call. */
                Malformed record; /**< Records the call. */
            };

            /** @brief Returns the row name of a refusal. */
            [[nodiscard]] std::string refusal_name(const ::testing::TestParamInfo<Refusal>& info)
            {
                return info.param.name;
            }

            /** @brief Returns the fifteen malformed calls. */
            [[nodiscard]] std::vector<Refusal> refusals()
            {
                return {
                    {"UploHoldingZero",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.A, o.B, o.C, static_cast<Core::Uplo>(0));
                     }},
                    {"TransHoldingZero",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.A, o.B, o.C, Core::Uplo::Upper,
                                          static_cast<Core::Transpose>(0));
                     }},
                    {"Syr2kWithConjTransOnComplex32",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.complex_A, o.complex_B, o.complex_C, Core::Uplo::Upper,
                                          Core::Transpose::ConjTrans);
                     }},
                    {"Her2kWithTransOnComplex32",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.her2k(o.complex_A, o.complex_B, o.complex_C, Core::Uplo::Upper,
                                          Core::Transpose::Trans);
                     }},
                    {"BOfAnotherDtype",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands& o)
                     {
                         const Tensor other = engine.create_tensor({3, 2}, Core::DataType::Float64);
                         return ops.syr2k(o.A, other, o.C, Core::Uplo::Upper);
                     }},
                    {"COfAnotherDtype",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands& o)
                     {
                         Tensor other = engine.create_tensor({3, 3}, Core::DataType::Float64);
                         return ops.syr2k(o.A, o.B, other, Core::Uplo::Upper);
                     }},
                    {"ComplexAlphaOnFloat32ThroughSyr2k",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.A, o.B, o.C, Core::Uplo::Upper,
                                          Core::Transpose::NoTrans, K_COMPLEX_SCALAR);
                     }},
                    {"ComplexAlphaOnFloat32ThroughHer2k",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.her2k(o.A, o.B, o.C, Core::Uplo::Upper,
                                          Core::Transpose::NoTrans, K_COMPLEX_SCALAR);
                     }},
                    {"COfThreeByFour",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands& o)
                     {
                         Tensor other = engine.create_tensor({3, 4});
                         return ops.syr2k(o.A, o.B, other, Core::Uplo::Upper);
                     }},
                    {"OpAWithNPlusOneRows",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.tall, o.B, o.C, Core::Uplo::Upper);
                     }},
                    {"OpBWithNPlusOneRows",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         return ops.syr2k(o.A, o.tall, o.C, Core::Uplo::Upper);
                     }},
                    {"OpBWithKPlusOneColumns",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands& o)
                     {
                         const Tensor other = engine.create_tensor({3, 3});
                         return ops.syr2k(o.A, other, o.C, Core::Uplo::Upper);
                     }},
                    {"ColumnMajorAWithRowMajorC",
                     [](SushiBLAS::BLASOps& ops, Engine&, RankTwoOperands& o)
                     {
                         const Tensor other(o.A.storage, {3, 2}, Core::DataType::Float32, 0,
                                            Core::Layout::ColumnMajor);
                         return ops.syr2k(other, o.B, o.C, Core::Uplo::Upper);
                     }},
                    {"BatchOfThreeAgainstBatchOfFour",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands&)
                     {
                         const Tensor batch_A = engine.create_tensor({3, 3, 2});
                         const Tensor batch_B = engine.create_tensor({4, 3, 2});
                         Tensor batch_C = engine.create_tensor({4, 3, 3});
                         return ops.syr2k(batch_A, batch_B, batch_C, Core::Uplo::Upper);
                     }},
                    {"Int32A",
                     [](SushiBLAS::BLASOps& ops, Engine& engine, RankTwoOperands&)
                     {
                         const Tensor int_A = engine.create_tensor({3, 2}, Core::DataType::Int32);
                         const Tensor int_B = engine.create_tensor({3, 2}, Core::DataType::Int32);
                         Tensor int_C = engine.create_tensor({3, 3}, Core::DataType::Int32);
                         return ops.syr2k(int_A, int_B, int_C, Core::Uplo::Upper);
                     }}};
            }
        } // namespace

        /**
         * @brief Records each malformed syr2k or her2k call on a HostGraphProbe: the checks read
         *        shape, dtype, flags and scalars, and no refusal here needs a device (decision 75).
         */
        class Small_Syr2kRefusals : public HostEngineTest,
                                    public ::testing::WithParamInterface<Refusal>
        {

        };

        TEST_P(Small_Syr2kRefusals, TheCallIsRefusedWithInvalidArgumentAndRecordsNoTask)
        {
            HostGraphProbe probe;
            RankTwoOperands operands{
                engine->create_tensor({3, 2}),
                engine->create_tensor({3, 2}),
                engine->create_tensor({3, 3}),
                engine->create_tensor({3, 2}, Core::DataType::Complex32),
                engine->create_tensor({3, 2}, Core::DataType::Complex32),
                engine->create_tensor({3, 3}, Core::DataType::Complex32),
                engine->create_tensor({4, 2})};
            SushiBLAS::BLASOps ops(probe.recorder());

            EXPECT_THROW((void)GetParam().record(ops, *engine, operands), std::invalid_argument);
            EXPECT_EQ(0u, probe.recorded_count());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_Syr2kRefusals, ::testing::ValuesIn(refusals()),
                                 refusal_name);
    } // namespace Test
} // namespace SushiBLAS
