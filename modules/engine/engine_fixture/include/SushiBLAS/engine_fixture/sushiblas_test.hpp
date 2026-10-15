/**************************************************************************/
/* sushiblas_test.hpp                                                     */
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
 * @file sushiblas_test.hpp
 * @brief Declares SushiBLASTest, the fixture that gives each test a fresh Engine on the shared runtime.
 * @author Mustafa Garip
 * @see modules/engine/engine_fixture/README.md
 */

#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
        * @brief Universal test fixture for SushiBLAS tests.
        */
        class SushiBLASTest : public ::testing::Test 
        {
            protected:
                std::unique_ptr<Engine> engine;

                void SetUp() override 
                {
                    engine = std::make_unique<Engine>(TestEnvironment::runtime(),
                                                      Core::Layout::RowMajor);
                    reset_fill_random_generator();
                }

                void TearDown() override 
                {
                    engine.reset();
                }

                void reinit_engine(Core::Layout layout)
                {
                    engine = std::make_unique<Engine>(TestEnvironment::runtime(), layout);
                }

                /**
                * @brief Creates a tensor of dtype `T` and initializes it from a host-side vector.
                */
                template<typename T>
                Tensor create_initialized_tensor(std::initializer_list<int64_t> dims, const std::vector<T>& data)
                {
                    auto t = engine->create_tensor(dims, sushiblas_dtype<T>());
                    auto& q = TestEnvironment::queue();
                    q.memcpy(t.data(), data.data(), data.size() * sizeof(T)).wait();
                    return t;
                }
        };
    } // namespace Test
} // namespace SushiBLAS
