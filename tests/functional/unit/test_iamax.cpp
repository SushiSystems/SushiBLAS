/****************************************************************/
/* test_iamax.cpp                                               */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// IAMAX answers "which element has the largest |Re| + |Im|", so its result is an
// index and its result tensor is INT64.
//
// This file used to allocate two FLOAT32 elements and hope, with a comment reading
// "Wait, different sizes!", because there was no integer dtype to allocate instead.
// It then asserted EXPECT_TRUE(true) - the op's actual answer was never checked by
// anything. Both halves of that are fixed here: the result has a dtype that fits it,
// and the index is read back and compared.

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

    // |-7| and |7| tie; BLAS specifies the first. The reference and SYCL kernels both
    // scan in ascending order and only replace on a strict improvement, which is what
    // makes that reproducible rather than incidental.
    EXPECT_EQ(1, index_of(R));
}

// The defect this dtype was added for. iamax writes eight bytes; a FLOAT32 result
// element is four. With tensors packed into an arena the neighbouring four bytes
// belong to something else, and Tensor::region() declared a four-byte write, so the
// dependency tracker did not order anything against the corruption either.
//
// Nothing can detect that from outside, which is why the fix is a refusal rather
// than a repair: a result tensor that cannot hold the answer is rejected at record
// time, by name.
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
