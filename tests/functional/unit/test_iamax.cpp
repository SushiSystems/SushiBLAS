/****************************************************************/
/* test_iamax.cpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_iamax.cpp
 * @brief Tests that blas().iamax returns the zero-based index of the largest magnitude as INT64.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

namespace
{
    /** @brief Reads a scalar INT64 tensor back to the host. */
    int64_t index_of(const SushiBLAS::Tensor& t)
    {
        int64_t host = -1;
        SushiBLAS::Test::TestEnvironment::get_context().get_queue()
            .memcpy(&host, t.data(), sizeof(int64_t)).wait();
        return host;
    }
}

class Unit_IAMAXTest : public SushiBLASTest {};

TEST_F(Unit_IAMAXTest, SimpleIAMAX)
{
    auto X = engine->create_tensor({4});
    auto R = engine->create_tensor({1}, SushiBLAS::Core::DataType::INT64);

    fill_tensor(X, {1.0f, -5.0f, 3.0f, 2.0f});

    engine->blas().iamax(X, R);
    engine->execute().wait();

    // Largest magnitude is |-5| at index 1. Zero-based, as the C++ BLAS bindings are.
    EXPECT_EQ(1, index_of(R));
}

TEST_F(Unit_IAMAXTest, TiesGoToTheLowestIndex)
{
    auto X = engine->create_tensor({5});
    auto R = engine->create_tensor({1}, SushiBLAS::Core::DataType::INT64);

    fill_tensor(X, {2.0f, -7.0f, 1.0f, 7.0f, 0.0f});

    engine->blas().iamax(X, R);
    engine->execute().wait();

    // |-7| and |7| tie, and BLAS specifies the first. See tests/functional/unit/README.md.
    EXPECT_EQ(1, index_of(R));
}

/**
 * @brief Tests that iamax rejects a result tensor too narrow to hold the index.
 * @see tests/functional/unit/README.md
 */
TEST_F(Unit_IAMAXTest, RejectsAResultTensorTooNarrowToHoldTheIndex)
{
    auto X = engine->create_tensor({4});
    fill_tensor(X, {1.0f, -5.0f, 3.0f, 2.0f});

    for (auto dtype : {SushiBLAS::Core::DataType::FLOAT32,
                       SushiBLAS::Core::DataType::FLOAT64,
                       SushiBLAS::Core::DataType::INT32,
                       SushiBLAS::Core::DataType::HALF})
    {
        auto R = engine->create_tensor({1}, dtype);
        EXPECT_THROW(engine->blas().iamax(X, R), std::runtime_error)
            << "iamax must refuse a " << SushiBLAS::Core::to_string(dtype)
            << " result rather than write an int64_t into it.";
    }
}
