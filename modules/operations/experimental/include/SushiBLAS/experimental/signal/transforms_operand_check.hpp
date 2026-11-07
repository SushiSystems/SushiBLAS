/**************************************************************************/
/* transforms_operand_check.hpp                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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

/**
 * @file transforms_operand_check.hpp
 * @brief Declares the record-time operand checks of TransformsOps::fft and inverse_fft, which
 *        read shapes and storage facts and so run without a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/experimental/dense_operand.hpp>
#include <SushiBLAS/experimental/signal/transforms.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Throws unless the operands and the shape of one transform are well formed.
         * @param forward True when input holds samples and output the spectrum, false for the inverse.
         * @param op The operation id the messages name.
         * @return The dtype of the real parts of the spectrum, which the plan is built in.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         */
        inline Core::DataType check_transform_operands(const Dispatch::PlacedTensor& input,
                                                       const Dispatch::PlacedTensor& output,
                                                       const FftShape& shape, bool forward,
                                                       const char* op)
        {
            Dispatch::require_operand(shape.rows >= 1 && shape.columns >= 1, op, "shape",
                                      "has {} rows of {} columns; both must be at least 1",
                                      shape.rows, shape.columns);

            const bool real = shape.kind == FftKind::RealToComplex;
            const Dispatch::PlacedTensor& samples = forward ? input : output;
            const Dispatch::PlacedTensor& spectrum = forward ? output : input;
            const char* samples_name = forward ? "input" : "output";
            const char* spectrum_name = forward ? "output" : "input";

            require_complex_dtype(spectrum.tensor, op, spectrum_name);
            const Core::DataType scalar = scalar_dtype_of(spectrum.tensor.dtype);
            Dispatch::require_dtype(samples.tensor, real ? scalar : spectrum.tensor.dtype, op,
                                    samples_name);

            const std::size_t spectrum_columns = real ? shape.columns / 2 + 1 : shape.columns;
            require_dense_elements(samples, shape.batch * shape.rows * shape.columns, op,
                                   samples_name);
            require_dense_elements(spectrum, shape.batch * shape.rows * spectrum_columns, op,
                                   spectrum_name);
            Dispatch::require_disjoint(output.storage, input.storage, op, "output", "input");
            return scalar;
        }
    } // namespace Internal
} // namespace SushiBLAS
