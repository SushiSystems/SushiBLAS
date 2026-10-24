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
 * @file test_dispatch.cpp
 * @brief Tests the element domains, the dtype switch and the task parameters of the dispatch seam.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
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

        TEST(Medium_Dispatch, TheDomainsAreBitsAndTheirUnions)
        {
            EXPECT_TRUE(Dispatch::domain_admits(Domain::RealAndComplex, Domain::Complex));
            EXPECT_FALSE(Dispatch::domain_admits(Domain::RealAndComplex, Domain::Integral));
            EXPECT_TRUE(Dispatch::domain_admits(Domain::All, Domain::Integral));
            EXPECT_FALSE(Dispatch::domain_admits(Domain::Real, Domain::Complex));
        }

        TEST(Medium_Dispatch, EveryDtypeBelongsToItsDomain)
        {
            EXPECT_EQ(Dispatch::domain_of(Core::DataType::Half), Domain::Real);
            EXPECT_EQ(Dispatch::domain_of(Core::DataType::Float64), Domain::Real);
            EXPECT_EQ(Dispatch::domain_of(Core::DataType::Complex32), Domain::Complex);
            EXPECT_EQ(Dispatch::domain_of(Core::DataType::Int64), Domain::Integral);
            EXPECT_EQ(Dispatch::domain_of(Core::DataType::UInt32), Domain::None);
        }

        TEST(Medium_Dispatch, NoDomainAdmitsUInt32)
        {
            EXPECT_FALSE(Dispatch::domain_admits(
                Domain::All, Dispatch::domain_of(Core::DataType::UInt32)));
            EXPECT_THROW(Dispatch::require_domain<Domain::All>(Core::DataType::UInt32, "f", "op"),
                         std::invalid_argument);

            int bytes = 0;
            EXPECT_ANY_THROW(Dispatch::with_element_type<Domain::All>(
                Core::DataType::UInt32, "f", "op", SizeArm{&bytes}));
            EXPECT_EQ(bytes, 0);
        }

        TEST(Medium_Dispatch, RequireDomainRefusesADtypeOutsideIt)
        {
            EXPECT_NO_THROW(Dispatch::require_domain<Domain::Real>(Core::DataType::Float32, "f",
                                                                   "op"));
            EXPECT_ANY_THROW(Dispatch::require_domain<Domain::Real>(Core::DataType::Int32, "f",
                                                                    "op"));
        }

        TEST(Medium_Dispatch, WithElementTypeCallsTheArmForTheDtypesType)
        {
            int bytes = 0;

            Dispatch::with_element_type<Domain::All>(Core::DataType::Complex64, "f", "op",
                                                     SizeArm{&bytes});
            EXPECT_EQ(bytes, static_cast<int>(sizeof(std::complex<double>)));

            Dispatch::with_element_type<Domain::All>(Core::DataType::Half, "f", "op",
                                                     SizeArm{&bytes});
            EXPECT_EQ(bytes, static_cast<int>(sizeof(sycl::half)));
        }

        TEST(Medium_Dispatch, WithElementTypeThrowsForADtypeTheDomainDoesNotServe)
        {
            int bytes = 0;

            EXPECT_ANY_THROW(Dispatch::with_element_type<Domain::Real>(
                Core::DataType::Int32, "f", "op", SizeArm{&bytes}));
            EXPECT_EQ(bytes, 0);
        }

        TEST(Medium_Dispatch, SetTaskParamsFillsTheSlotsInOrderAndRefusesTooMany)
        {
            TaskDescription description;

            Dispatch::set_task_params<int64_t>(description, {3, -4, 5}, "op");

            EXPECT_EQ(description.parameter<int64_t>(0), int64_t{3});
            EXPECT_EQ(description.parameter<int64_t>(1), int64_t{-4});
            EXPECT_EQ(description.parameter<int64_t>(2), int64_t{5});
            EXPECT_ANY_THROW(Dispatch::set_task_params<int64_t>(description,
                                                                {1, 2, 3, 4, 5, 6, 7}, "op", 2));
        }
    } // namespace Test
} // namespace SushiBLAS
