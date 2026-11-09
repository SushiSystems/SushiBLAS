/**************************************************************************/
/* test_argument_check.cpp                                                */
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
 * @file test_argument_check.cpp
 * @brief Tests the record-time checks on a flag, a scalar, a dtype and a refused transpose.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a tensor of two elements and the dtype, with no storage. */
            Tensor tensor_without_storage(Core::DataType dtype)
            {
                const std::vector<std::int64_t> dims = {2};
                return Tensor(SushiRuntime::sushi_ptr<Storage>(),
                              SushiRuntime::span<const std::int64_t>(dims.data(), dims.size()),
                              dtype);
            }

            /** @brief Names a scalar, a tensor dtype and whether the scalar check accepts the pair. */
            struct RealScalarRow
            {
                const char* name;      /**< The row name a failure shows. */
                BLASScalar scalar;     /**< The scalar given. */
                Core::DataType dtype;  /**< The dtype of the tensor it scales. */
                bool accepted;         /**< Whether the check lets the pair through. */
            };

            /** @brief Returns one row per accepted and refused pair. */
            std::vector<RealScalarRow> real_scalar_rows()
            {
                const BLASScalar real(2.0);
                const BLASScalar complex(std::complex<double>(1.0, 1.0));
                return {{"ARealScalarForFloat32", real, Core::DataType::Float32, true},
                        {"AComplexScalarForComplex64", complex, Core::DataType::Complex64, true},
                        {"AComplexScalarForFloat64", complex, Core::DataType::Float64, false}};
            }
        } // namespace

        TEST(Small_ArgumentCheck, AFlagMustBeAnEnumerator)
        {
            EXPECT_NO_THROW(Dispatch::require_flag(Core::Uplo::Lower, "op", "uplo"));
            EXPECT_THROW(Dispatch::require_flag(Core::Uplo{}, "op", "uplo"),
                         std::invalid_argument);
        }

        /** @brief Runs one row of real_scalar_rows(). */
        class Small_ArgumentCheckRealScalar : public ::testing::TestWithParam<RealScalarRow>
        {
        };

        TEST_P(Small_ArgumentCheckRealScalar, ARealDtypeTakesOnlyARealScalar)
        {
            const RealScalarRow& row = GetParam();
            const auto check = [&] {
                Dispatch::require_real_scalar(row.scalar, row.dtype, "op", "alpha");
            };

            if (row.accepted)
                EXPECT_NO_THROW(check());
            else
                EXPECT_THROW(check(), std::invalid_argument);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ArgumentCheckRealScalar, ::testing::ValuesIn(real_scalar_rows()),
            [](const ::testing::TestParamInfo<RealScalarRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_ArgumentCheck, ARealRoutineRefusesAComplexTensor)
        {
            const Tensor real = tensor_without_storage(Core::DataType::Float32);
            const Tensor complex = tensor_without_storage(Core::DataType::Complex32);

            EXPECT_NO_THROW(Dispatch::require_real_dtype(real, "op", "x", "the H routine"));
            EXPECT_THROW(Dispatch::require_real_dtype(complex, "op", "x", "the H routine"),
                         std::invalid_argument);
        }

        TEST(Small_ArgumentCheck, ARefusedTransposeNamesTheSibling)
        {
            std::string message;

            EXPECT_NO_THROW(Dispatch::require_transpose_not(Core::Transpose::Trans,
                                                            Core::Transpose::ConjTrans, "op",
                                                            "sibling"));
            try
            {
                Dispatch::require_transpose_not(Core::Transpose::ConjTrans,
                                                Core::Transpose::ConjTrans, "op", "sibling");
            }
            catch (const std::invalid_argument& error)
            {
                message = error.what();
            }

            EXPECT_NE(message.find("sibling"), std::string::npos) << message;
        }
    } // namespace Test
} // namespace SushiBLAS
