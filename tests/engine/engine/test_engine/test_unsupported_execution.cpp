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
 * @file test_unsupported_execution.cpp
 * @brief Tests which record-time refusals throw UnsupportedExecution and which do not.
 * @author Mustafa Garip
 */

#include <string>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
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

            /** @brief Returns the text of the UnsupportedExecution a call throws, or "no refusal". */
            template<typename Call>
            [[nodiscard]] std::string refusal_text_of(const Call& call)
            {
                try
                {
                    call();
                }
                catch (const UnsupportedExecution& refusal)
                {
                    return refusal.what();
                }

                return "no refusal";
            }
        } // namespace

        /** @brief Tests that the execution helper throws UnsupportedExecution with the text op: message. */
        TEST(Small_UnsupportedExecution, TheExecutionHelperThrowsItNamingTheOperation)
        {
            const auto raise = []
            {
                Dispatch::Detail::throw_execution_error("op", "cannot run under {}", "a mode");
            };

            EXPECT_EQ(refusal_text_of(raise), "op: cannot run under a mode");
        }

        /** @brief Tests that a dtype other than the one required is an invalid_argument, not a refusal. */
        TEST(Small_UnsupportedExecution, AWrongDtypeIsNotARefusal)
        {
            const Tensor integers(SushiRuntime::sushi_ptr<Storage>(), {6}, Core::DataType::Int32);

            const Outcome outcome = outcome_of(
                [&] { Dispatch::require_dtype(integers, Core::DataType::Float32, "gemm", "b"); });

            EXPECT_EQ(outcome, Outcome::InvalidArgument);
        }

        /** @brief Tests that a rank other than the one required is an invalid_argument, not a refusal. */
        TEST(Small_UnsupportedExecution, AWrongRankIsNotARefusal)
        {
            const Tensor vector(SushiRuntime::sushi_ptr<Storage>(), {6}, Core::DataType::Float32);

            const Outcome outcome =
                outcome_of([&] { Dispatch::require_rank(vector, 2, "gemm", "a"); });

            EXPECT_EQ(outcome, Outcome::InvalidArgument);
        }

        /** @brief Records calls on an engine and reports how each one ended. */
        class Medium_UnsupportedExecutionSites : public SushiBLASTest
        {
            protected:
                /** @brief Returns how a recording call ended, running the graph when it recorded. */
                template<typename Call>
                [[nodiscard]] Outcome outcome_of_run(const Call& call)
                {
                    const Outcome outcome = outcome_of(call);
                    run_when_recorded(outcome);
                    return outcome;
                }

            private:
                /** @brief Runs the recorded graph when a call returned. */
                void run_when_recorded(Outcome outcome)
                {
                    if (outcome == Outcome::Returned)
                        recording->run().value();
                }
        };

        /** @brief Gives each test a tensor in device-only USM, or skips when the device has none. */
        class Medium_UnsupportedExecutionDeviceOperand : public SushiBLASTest
        {
            protected:
                /** @brief Makes a FLOAT32 matrix of 2 by 3 in device-only USM and skips when it cannot. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    device_operand = make_device_tensor({2, 3});
                    if (!device_operand)
                        GTEST_SKIP() << "This device offers no device-only USM.";
                }

                /// @brief The device-only operand; empty only when the test was skipped.
                std::optional<Tensor> device_operand;

            private:
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

        TEST_F(Medium_UnsupportedExecutionDeviceOperand, ForceReferenceRefusesADeviceOnlyOperand)
        {
            const Tensor b = engine->create_tensor({3, 2});
            Tensor c = engine->create_tensor({2, 2});

            set_mode(ExecutionMode::ForceReference);
            const Outcome outcome =
                outcome_of([&] { SushiBLAS::BLASOps(recorder()).gemm(*device_operand, b, c); });

            EXPECT_EQ(outcome, Outcome::UnsupportedExecution);
        }

        TEST_F(Medium_UnsupportedExecutionSites, PoissonWithoutFp64IsARefusal)
        {
            Tensor t = engine->create_tensor({4});

            const Outcome outcome = outcome_of_run(
                [&] { SushiBLAS::RandomOps(recorder(), engine->rng()).poisson(t, 1.0); });

            EXPECT_EQ(outcome, outcome_on(Device::profile_for(t.device()).supports_fp64));
        }
    } // namespace Test
} // namespace SushiBLAS
