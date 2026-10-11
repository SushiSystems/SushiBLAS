/**************************************************************************/
/* test_unsupported_execution.cpp                                         */
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
 * @file test_unsupported_execution.cpp
 * @brief Tests which record-time refusals throw UnsupportedExecution and which do not.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names how a call ended. */
            enum class Outcome : std::uint8_t
            {
                Returned,              /**< The call threw nothing. */
                UnsupportedExecution, /**< The call threw UnsupportedExecution. */
                InvalidArgument       /**< The call threw another std::invalid_argument. */
            };

            /** @brief Returns how call ended; any other exception leaves the test. */
            template<typename Call>
            [[nodiscard]] Outcome outcome_of(const Call& call)
            {
                try
                {
                    call();
                }
                catch (const UnsupportedExecution&)
                {
                    return Outcome::UnsupportedExecution;
                }
                catch (const std::invalid_argument&)
                {
                    return Outcome::InvalidArgument;
                }

                return Outcome::Returned;
            }

            /** @brief Returns how a call that needs a device aspect ends, given the aspect. */
            [[nodiscard]] constexpr Outcome outcome_on(bool device_has_aspect) noexcept
            {
                return device_has_aspect ? Outcome::Returned : Outcome::UnsupportedExecution;
            }
        } // namespace

        TEST(Unit_UnsupportedExecution, IsAFinalInvalidArgumentBuiltFromAMessage)
        {
            static_assert(std::is_base_of<std::invalid_argument, UnsupportedExecution>::value,
                          "UnsupportedExecution derives from std::invalid_argument.");
            static_assert(std::is_final<UnsupportedExecution>::value,
                          "UnsupportedExecution is final.");
            static_assert(std::is_constructible<UnsupportedExecution, const std::string&>::value,
                          "UnsupportedExecution takes a std::string.");
            static_assert(std::is_constructible<UnsupportedExecution, const char*>::value,
                          "UnsupportedExecution takes a C string.");
            static_assert(!std::is_convertible<std::string, UnsupportedExecution>::value,
                          "A std::string does not convert to UnsupportedExecution implicitly.");

            const UnsupportedExecution from_string(std::string("mode"));
            const UnsupportedExecution from_literal("device");
            EXPECT_STREQ(from_string.what(), "mode");
            EXPECT_STREQ(from_literal.what(), "device");
        }

        TEST(Unit_UnsupportedExecution, IsCaughtAsItselfAndAsInvalidArgument)
        {
            const auto raise = [] { throw UnsupportedExecution("refused"); };

            EXPECT_THROW(raise(), UnsupportedExecution);
            EXPECT_THROW(raise(), std::invalid_argument);
            EXPECT_EQ(outcome_of(raise), Outcome::UnsupportedExecution);
        }

        TEST(Unit_UnsupportedExecution, APlainInvalidArgumentIsNotCaughtAsIt)
        {
            const auto raise = [] { throw std::invalid_argument("wrong operand"); };

            EXPECT_EQ(outcome_of(raise), Outcome::InvalidArgument);
        }

        TEST(Unit_UnsupportedExecution, TheExecutionHelperThrowsItNamingTheOperation)
        {
            const auto raise = []
            {
                Dispatch::Detail::throw_execution_error("op", "cannot run under {}", "a mode");
            };

            EXPECT_EQ(outcome_of(raise), Outcome::UnsupportedExecution);
            try
            {
                raise();
            }
            catch (const UnsupportedExecution& refusal)
            {
                EXPECT_STREQ(refusal.what(), "op: cannot run under a mode");
            }
        }

        TEST(Unit_UnsupportedExecution, TheOperandAndDtypeHelpersDoNotThrowIt)
        {
            const auto wrong_operand = []
            {
                Dispatch::Detail::throw_operand_error("op", "x", "has rank {}", 3);
            };
            const auto wrong_dtype = []
            {
                Dispatch::Detail::throw_dtype_error("Elementwise operation", "sqrt",
                                                    Core::DataType::Int32);
            };

            EXPECT_EQ(outcome_of(wrong_operand), Outcome::InvalidArgument);
            EXPECT_EQ(outcome_of(wrong_dtype), Outcome::InvalidArgument);
        }

        /** @brief Records calls on an engine and reports how each one ended. */
        class Unit_UnsupportedExecutionSites : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Returns a FLOAT32 tensor of dims in device-only USM, or nothing.
                 * @return Empty when the allocation fails or the pointer is not usm::alloc::device.
                 */
                std::optional<Tensor> make_device_tensor(std::initializer_list<int64_t> dims)
                {
                    try
                    {
                        Tensor t = engine->create_tensor(dims, Core::DataType::Float32,
                                                         SushiRuntime::Residency::Device);
                        const sycl::context context =
                            TestEnvironment::queue().get_context();
                        if (sycl::get_pointer_type(t.data(), context) != sycl::usm::alloc::device)
                            return std::nullopt;
                        return t;
                    }
                    catch (const std::exception&)
                    {
                        return std::nullopt;
                    }
                }
        };

        TEST_F(Unit_UnsupportedExecutionSites, ForceReferenceRefusesADeviceOnlyOperand)
        {
            std::optional<Tensor> a = make_device_tensor({2, 3});
            if (!a)
                GTEST_SKIP() << "This device offers no device-only USM.";
            const Tensor b = engine->create_tensor({3, 2});
            Tensor c = engine->create_tensor({2, 2});

            engine->set_mode(ExecutionMode::ForceReference);
            const Outcome outcome =
                outcome_of([&] { SushiBLAS::BLASOps(*engine).gemm(*a, b, c); });

            EXPECT_EQ(outcome, Outcome::UnsupportedExecution);
        }

        TEST_F(Unit_UnsupportedExecutionSites, AWrongDtypeUnderForceReferenceIsNotARefusal)
        {
            const Tensor a = engine->create_tensor({2, 3});
            const Tensor b = engine->create_tensor({3, 2}, Core::DataType::Int32);
            Tensor c = engine->create_tensor({2, 2});

            engine->set_mode(ExecutionMode::ForceReference);
            const Outcome outcome =
                outcome_of([&] { SushiBLAS::BLASOps(*engine).gemm(a, b, c); });

            EXPECT_EQ(outcome, Outcome::InvalidArgument);
        }

        TEST_F(Unit_UnsupportedExecutionSites, AWrongRankIsNotARefusal)
        {
            const Tensor a = engine->create_tensor({6});
            const Tensor b = engine->create_tensor({3, 2});
            Tensor c = engine->create_tensor({2, 2});

            const Outcome outcome =
                outcome_of([&] { SushiBLAS::BLASOps(*engine).gemm(a, b, c); });

            EXPECT_EQ(outcome, Outcome::InvalidArgument);
        }

        TEST_F(Unit_UnsupportedExecutionSites, ADtypeTheDeviceLacksIsARefusal)
        {
            const Tensor wide = engine->create_tensor({4}, Core::DataType::Float64);
            const Tensor half = engine->create_tensor({4}, Core::DataType::Half);

            EXPECT_EQ(outcome_of([&] { Dispatch::require_device_dtype(wide, "op", "x"); }),
                      outcome_on(Device::profile_for(wide.device()).supports_fp64));
            EXPECT_EQ(outcome_of([&] { Dispatch::require_device_dtype(half, "op", "x"); }),
                      outcome_on(Device::profile_for(half.device()).supports_fp16));
        }

        TEST_F(Unit_UnsupportedExecutionSites, PoissonWithoutFp64IsARefusal)
        {
            Tensor t = engine->create_tensor({4});

            const Outcome outcome =
                outcome_of([&] { SushiBLAS::RandomOps(*engine).poisson(t, 1.0); });
            if (outcome == Outcome::Returned)
                engine->execute().value();

            EXPECT_EQ(outcome, outcome_on(Device::profile_for(t.device()).supports_fp64));
        }
    } // namespace Test
} // namespace SushiBLAS
