/**************************************************************************/
/* convolve_operand_check.hpp                                             */
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
 * @file convolve_operand_check.hpp
 * @brief Declares the record-time operand checks of SignalOps::convolve_1d, which read shapes
 *        and storage facts and so run without a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Throws unless a tensor is dense and its last dimension runs along memory.
         * @throws std::invalid_argument on a view, a slice or a column-major tensor of rank 2+.
         * @see modules/operations/signal/README.md
         */
        inline void require_dense_rows(const Tensor& t, const char* name, const char* operand)
        {
            Dispatch::require_contiguous(t, name, operand);
            Dispatch::require_operand(t.rank < 2 || t.layout == Core::Layout::RowMajor, name,
                                      operand, "is column-major; its rows must be contiguous");
        }

        /**
         * @brief Throws unless the three operands of a convolution are a well-formed call.
         * @param name The operation id the messages name.
         * @return How many rows the signal holds: the product of its leading dimensions.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         * @see modules/operations/signal/README.md
         */
        inline int64_t check_convolve_operands(const Dispatch::PlacedTensor& signal,
                                               const Dispatch::PlacedTensor& taps,
                                               const Dispatch::PlacedTensor& out, const char* name)
        {
            Dispatch::require_dtype(taps.tensor, signal.tensor.dtype, name, "taps");
            Dispatch::require_dtype(out.tensor, signal.tensor.dtype, name, "out");

            Dispatch::require_rank(taps.tensor, 1, name, "taps");
            Dispatch::require_operand(signal.tensor.rank >= 1, name, "signal",
                                      "has rank 0; a row needs one dimension");
            Dispatch::require_rank(out.tensor, signal.tensor.rank, name, "out");

            const int64_t signal_length = signal.tensor.shape[signal.tensor.rank - 1];
            const int64_t tap_count = taps.tensor.shape[0];
            Dispatch::require_operand(tap_count >= 1, name, "taps",
                                      "holds {} taps; at least 1 is required", tap_count);
            Dispatch::require_operand(signal_length >= tap_count, name, "signal",
                                      "has {} samples per row, fewer than the {} taps",
                                      signal_length, tap_count);

            int64_t rows = 1;
            for (int32_t i = 0; i + 1 < signal.tensor.rank; ++i)
            {
                Dispatch::require_extent(out.tensor.shape[i], signal.tensor.shape[i],
                                         "rows in a leading dimension", name, "out");
                rows *= signal.tensor.shape[i];
            }
            Dispatch::require_extent(out.tensor.shape[out.tensor.rank - 1],
                                     signal_length - tap_count + 1, "samples per row", name,
                                     "out");

            Dispatch::require_storage(signal.tensor, signal.storage, name, "signal");
            Dispatch::require_storage(taps.tensor, taps.storage, name, "taps");
            Dispatch::require_storage(out.tensor, out.storage, name, "out");
            require_dense_rows(signal.tensor, name, "signal");
            require_dense_rows(taps.tensor, name, "taps");
            require_dense_rows(out.tensor, name, "out");
            Dispatch::require_device_dtype(signal.tensor, signal.storage, name, "signal");
            Dispatch::require_device_dtype(taps.tensor, taps.storage, name, "taps");
            Dispatch::require_device_dtype(out.tensor, out.storage, name, "out");

            // A gather reads its inputs while it writes, so the output may share no byte with them.
            Dispatch::require_disjoint(out.storage, signal.storage, name, "out", "signal");
            Dispatch::require_disjoint(out.storage, taps.storage, name, "out", "taps");
            return rows;
        }
    } // namespace Internal
} // namespace SushiBLAS
