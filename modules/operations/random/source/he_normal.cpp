/**************************************************************************/
/* he_normal.cpp                                                          */
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
 * @file he_normal.cpp
 * @brief Defines RandomOps::he_normal, which records the He normal fill of a tensor.
 * @author Mustafa Garip
 */

#include <cmath>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/random_kernels/random.hpp>
#include <SushiBLAS/random/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle RandomOps::he_normal(Tensor& t, int64_t n_in)
    {
        if (n_in <= 0)
            Dispatch::Detail::throw_operand_error("random.he_normal", "n_in",
                                                  "is {}, expected above 0", n_in);

        const double stddev = std::sqrt(2.0 / n_in);
        return Dispatch::execute_random<Dispatch::Domain::Real>(
            recorder_, rng_, t, "random.he_normal", "random.he_normal"_op,
            {static_cast<double>(n_in), stddev},
            [stddev](auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
                     uint64_t offset, int64_t size, auto* pT) -> sycl::event
            {
                using T = decltype(scalar_type);
                const T sd = static_cast<T>(stddev);

                return fill(size, pT, [=](uint64_t b)
                {
                    return Kernels::SYCL::Random::normal_block<T>(seed, offset, b,
                                                                  static_cast<T>(0.0), sd);
                });
            });
    }
} // namespace SushiBLAS
