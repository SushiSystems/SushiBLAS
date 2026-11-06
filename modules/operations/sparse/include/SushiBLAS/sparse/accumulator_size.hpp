/**************************************************************************/
/* accumulator_size.hpp                                                   */
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
 * @file accumulator_size.hpp
 * @brief Declares accumulator_bytes, the size of the type a kernel accumulates a dtype in.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Returns the bytes one Kernels::Accumulator of a dtype's element type takes.
         * @return 0 for an integer dtype, which no sparse kernel accumulates.
         * @see modules/operations/sparse/README.md
         */
        [[nodiscard]] inline constexpr std::size_t accumulator_bytes(Core::DataType dtype) noexcept
        {
            switch (dtype)
            {
                case Core::DataType::Half:
                    return sizeof(Kernels::Accumulator<sycl::half>);
                case Core::DataType::Float32:
                    return sizeof(Kernels::Accumulator<float>);
                case Core::DataType::Float64:
                    return sizeof(Kernels::Accumulator<double>);
                case Core::DataType::Complex32:
                    return sizeof(Kernels::Accumulator<std::complex<float>>);
                case Core::DataType::Complex64:
                    return sizeof(Kernels::Accumulator<std::complex<double>>);
                case Core::DataType::Int32:
                case Core::DataType::Int64:
                case Core::DataType::UInt32:
                    return 0;
            }
            return 0;
        }
    } // namespace Internal
} // namespace SushiBLAS
