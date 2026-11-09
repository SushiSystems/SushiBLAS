/**************************************************************************/
/* test_dispatch.cpp                                                      */
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
 * @file test_dispatch.cpp
 * @brief Tests the element domains, the dtype switch and the task parameters of the dispatch seam.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Dispatch::Domain;

        /** @brief Records the size of the element type an arm was called with. */
        struct SizeArm
        {
            int* bytes; /**< Receives sizeof the element type. */

            /** @brief Records sizeof T from the tag form. */
            template<typename T>
            void operator()(Dispatch::ElementTag<T>) const
            {
                *bytes = static_cast<int>(sizeof(T));
            }
        };

        /** @brief Tests that domain_admits() admits exactly the families the bits of a domain name. */
        TEST(Small_Dispatch, TheDomainsAreBitsAndTheirUnions)
        {
            EXPECT_TRUE(Dispatch::domain_admits(Domain::RealAndComplex, Domain::Complex));
            EXPECT_FALSE(Dispatch::domain_admits(Domain::RealAndComplex, Domain::Integral));
            EXPECT_TRUE(Dispatch::domain_admits(Domain::All, Domain::Integral));
            EXPECT_FALSE(Dispatch::domain_admits(Domain::Real, Domain::Complex));
        }

        namespace
        {
            /** @brief Names a dtype and the family domain_of() gives it. */
            struct DomainRow
            {
                const char* name;      /**< The row name a failure shows. */
                Core::DataType dtype;  /**< The dtype whose family is read. */
                Domain domain;         /**< The family the dtype belongs to. */
            };

            /** @brief Returns one row per dtype. */
            std::vector<DomainRow> domain_rows()
            {
                return {{"Half", Core::DataType::Half, Domain::Real},
                        {"Float32", Core::DataType::Float32, Domain::Real},
                        {"Float64", Core::DataType::Float64, Domain::Real},
                        {"Complex32", Core::DataType::Complex32, Domain::Complex},
                        {"Complex64", Core::DataType::Complex64, Domain::Complex},
                        {"Int32", Core::DataType::Int32, Domain::Integral},
                        {"Int64", Core::DataType::Int64, Domain::Integral},
                        {"UInt32", Core::DataType::UInt32, Domain::None}};
            }
        } // namespace

        /** @brief Runs one row of domain_rows(). */
        class Small_DispatchDomain : public ::testing::TestWithParam<DomainRow>
        {

        };

        TEST_P(Small_DispatchDomain, DomainOfNamesTheFamilyOfTheDtype)
        {
            EXPECT_EQ(Dispatch::domain_of(GetParam().dtype), GetParam().domain);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_DispatchDomain, ::testing::ValuesIn(domain_rows()),
            [](const ::testing::TestParamInfo<DomainRow>& info)
            { return std::string(info.param.name); });

        /** @brief Tests that no domain, All included, admits the family of UInt32. */
        TEST(Small_Dispatch, NoDomainAdmitsUInt32)
        {
            EXPECT_FALSE(Dispatch::domain_admits(
                Domain::All, Dispatch::domain_of(Core::DataType::UInt32)));
        }

        /** @brief Tests that require_domain() refuses UInt32 under every domain with invalid_argument. */
        TEST(Small_Dispatch, RequireDomainRefusesUInt32)
        {
            EXPECT_THROW(Dispatch::require_domain<Domain::All>(Core::DataType::UInt32, "f", "op"),
                         std::invalid_argument);
        }

        /** @brief Tests that with_element_type() refuses UInt32 and never calls its arm. */
        TEST(Small_Dispatch, WithElementTypeRefusesUInt32WithoutCallingTheArm)
        {
            int bytes = 0;

            EXPECT_THROW(Dispatch::with_element_type<Domain::All>(
                             Core::DataType::UInt32, "f", "op", SizeArm{&bytes}),
                         std::runtime_error);
            EXPECT_EQ(bytes, 0);
        }

        /** @brief Tests that require_domain() admits a dtype of its domain and refuses one outside it. */
        TEST(Small_Dispatch, RequireDomainAdmitsItsDtypesAndRefusesTheRest)
        {
            EXPECT_NO_THROW(Dispatch::require_domain<Domain::Real>(Core::DataType::Float32, "f",
                                                                   "op"));
            EXPECT_THROW(Dispatch::require_domain<Domain::Real>(Core::DataType::Int32, "f", "op"),
                         std::invalid_argument);
        }

        /** @brief Tests that the arm of Complex64 is called with the 16-byte complex double. */
        TEST(Small_Dispatch, WithElementTypeCallsTheArmForComplex64)
        {
            int bytes = 0;

            Dispatch::with_element_type<Domain::All>(Core::DataType::Complex64, "f", "op",
                                                     SizeArm{&bytes});

            EXPECT_EQ(bytes, static_cast<int>(sizeof(std::complex<double>)));
        }

        /** @brief Tests that the arm of Half is called with sycl::half. */
        TEST(Small_Dispatch, WithElementTypeCallsTheArmForHalf)
        {
            int bytes = 0;

            Dispatch::with_element_type<Domain::All>(Core::DataType::Half, "f", "op",
                                                     SizeArm{&bytes});

            EXPECT_EQ(bytes, static_cast<int>(sizeof(sycl::half)));
        }

        /** @brief Tests that with_element_type() refuses a dtype its domain does not serve. */
        TEST(Small_Dispatch, WithElementTypeThrowsForADtypeTheDomainDoesNotServe)
        {
            int bytes = 0;

            EXPECT_THROW(Dispatch::with_element_type<Domain::Real>(
                             Core::DataType::Int32, "f", "op", SizeArm{&bytes}),
                         std::runtime_error);
            EXPECT_EQ(bytes, 0);
        }

        /** @brief Tests that set_task_params() writes its values to slots 0 to n - 1 in order. */
        TEST(Small_Dispatch, SetTaskParamsFillsTheSlotsInOrder)
        {
            TaskDescription description;

            Dispatch::set_task_params<int64_t>(description, {3, -4, 5}, "op");

            EXPECT_EQ(description.parameter<int64_t>(0), int64_t{3});
            EXPECT_EQ(description.parameter<int64_t>(1), int64_t{-4});
            EXPECT_EQ(description.parameter<int64_t>(2), int64_t{5});
        }

        /** @brief Tests that set_task_params() refuses values plus reserved slots above capacity. */
        TEST(Small_Dispatch, SetTaskParamsRefusesMoreValuesThanTheSlotsHold)
        {
            TaskDescription description;

            EXPECT_THROW(Dispatch::set_task_params<int64_t>(description, {1, 2, 3, 4, 5, 6, 7},
                                                            "op", 2),
                         std::invalid_argument);
        }
    } // namespace Test
} // namespace SushiBLAS
