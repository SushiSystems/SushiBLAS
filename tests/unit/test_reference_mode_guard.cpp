/**************************************************************************/
/* test_reference_mode_guard.cpp                                          */
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
 * @file test_reference_mode_guard.cpp
 * @brief Tests that ForceReference refuses device-only USM and runs on host-reachable USM.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include "../common/test_utils.hpp"

using namespace SushiBLAS;
using namespace SushiBLAS::Test;
using namespace SushiRuntime::Graph::Literals;

namespace
{
    /// @brief The 2x3 row-major left operand.
    const std::vector<float> K_A_VALUES = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    /// @brief The 3x2 row-major right operand.
    const std::vector<float> K_B_VALUES = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    /// @brief A * B: {1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12}.
    const std::vector<float> K_PRODUCT = {58.0f, 64.0f, 139.0f, 154.0f};

    /// @brief Absolute tolerance for a product of small integers held in FLOAT32.
    constexpr double K_TOLERANCE = 1e-5;

    /// @brief The operation name the guard's message carries.
    const std::string K_GEMM_NAME = "blas.lvl3.gemm";
} // namespace

/// @brief Fixture for the ForceReference host-pointer guard tests.
class Unit_ReferenceModeGuardTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing without such USM.
         * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
         */
        std::optional<Tensor> make_device_tensor(std::initializer_list<int64_t> dims)
        {
            try
            {
                Tensor t = engine->create_tensor(dims, Core::DataType::Float32,
                                                 SushiRuntime::API::Residency::Device);
                const sycl::context context =
                    TestEnvironment::get_context().get_queue().get_context();
                if (sycl::get_pointer_type(t.data(), context) != sycl::usm::alloc::device)
                    return std::nullopt;
                return t;
            }
            catch (const std::exception&)
            {
                return std::nullopt;
            }
        }

        /** @brief Returns the 2x3 operand A in device-only USM, or nothing without such USM. */
        std::optional<Tensor> make_device_a()
        {
            std::optional<Tensor> a = make_device_tensor({2, 3});
            if (a)
                fill_tensor(*a, K_A_VALUES);
            return a;
        }

        /** @brief Returns a 2x3 shared-USM tensor holding K_A_VALUES. */
        Tensor make_shared_a()
        {
            Tensor a = engine->create_tensor({2, 3});
            fill_tensor(a, K_A_VALUES);
            return a;
        }

        /** @brief Returns a 3x2 shared-USM tensor holding K_B_VALUES. */
        Tensor make_shared_b()
        {
            Tensor b = engine->create_tensor({3, 2});
            fill_tensor(b, K_B_VALUES);
            return b;
        }

        /**
         * @brief Records and executes C = A * B under mode and returns C's elements.
         */
        std::vector<float> run_gemm(ExecutionMode mode, const Tensor& a, const Tensor& b)
        {
            engine->set_mode(mode);
            Tensor c = engine->create_tensor({2, 2});
            fill_tensor(c, {0.0f, 0.0f, 0.0f, 0.0f});
            SushiBLAS::BLASOps(*engine).gemm(a, b, c);
            engine->execute();
            return SushiBLAS::IO(*engine).to_vector<float>(c);
        }
};

TEST_F(Unit_ReferenceModeGuardTest, ForceReferenceGemmRefusesADeviceOperand)
{
    std::optional<Tensor> a = make_device_a();
    if (!a)
        GTEST_SKIP() << "This device offers no device-only USM.";
    const Tensor b = make_shared_b();
    Tensor c = engine->create_tensor({2, 2});

    engine->set_mode(ExecutionMode::ForceReference);
    try
    {
        SushiBLAS::BLASOps(*engine).gemm(*a, b, c);
        FAIL() << "ForceReference must refuse a device-only operand.";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find(K_GEMM_NAME), std::string::npos) << message;
    }
}

TEST_F(Unit_ReferenceModeGuardTest, ForceReferenceRefusesADeviceOperandDeclaredByAnEpilogue)
{
    std::optional<Tensor> bias = make_device_tensor({2});
    if (!bias)
        GTEST_SKIP() << "This device offers no device-only USM.";
    const Tensor a = make_shared_a();
    const Tensor b = make_shared_b();
    Tensor c = engine->create_tensor({2, 2});
    const Kernels::ChannelBiasEpilogue epilogue{bias->data(), 0, 1};

    engine->set_mode(ExecutionMode::ForceReference);
    EXPECT_THROW(Dispatch::execute_gemm(engine->get_recorder(), a, b, c, Core::Transpose::NoTrans,
                                        Core::Transpose::NoTrans, 1.0, 0.0,
                                        "test.gemm_device_bias", "test.gemm_device_bias"_op,
                                        Dispatch::gemm_epilogue(epilogue).reads(*bias)),
                 std::invalid_argument);
}

TEST_F(Unit_ReferenceModeGuardTest, ForceReferenceGemmOnSharedOperandsMatchesSycl)
{
    const Tensor a = make_shared_a();
    const Tensor b = make_shared_b();

    const std::vector<float> reference = run_gemm(ExecutionMode::ForceReference, a, b);
    const std::vector<float> device = run_gemm(ExecutionMode::ForceSycl, a, b);

    ASSERT_EQ(K_PRODUCT.size(), reference.size());
    ASSERT_EQ(K_PRODUCT.size(), device.size());
    for (std::size_t i = 0; i < K_PRODUCT.size(); ++i)
    {
        EXPECT_NEAR(K_PRODUCT[i], reference[i], K_TOLERANCE) << "element " << i;
        EXPECT_NEAR(device[i], reference[i], K_TOLERANCE) << "element " << i;
    }
}

TEST_F(Unit_ReferenceModeGuardTest, ForceSyclGemmRunsOnADeviceOperand)
{
    std::optional<Tensor> a = make_device_a();
    if (!a)
        GTEST_SKIP() << "This device offers no device-only USM.";
    const Tensor b = make_shared_b();

    const std::vector<float> result = run_gemm(ExecutionMode::ForceSycl, *a, b);

    ASSERT_EQ(K_PRODUCT.size(), result.size());
    for (std::size_t i = 0; i < K_PRODUCT.size(); ++i)
        EXPECT_NEAR(K_PRODUCT[i], result[i], K_TOLERANCE) << "element " << i;
}
