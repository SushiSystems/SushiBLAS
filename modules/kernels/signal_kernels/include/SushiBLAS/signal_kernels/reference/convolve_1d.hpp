/**************************************************************************/
/* convolve_1d.hpp                                                        */
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

/**
 * @file convolve_1d.hpp
 * @brief Defines the reference kernel of the valid-mode one-dimensional convolution.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/signal_kernels/convolve_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Writes `out[n] = sum_k signal[n + K - 1 - k] * taps[k]` per sample.
             * @param rows The number of dense rows; signal and out hold them back to back.
             * @param signal_length The samples in one signal row, L; at least tap_count.
             * @param tap_count The taps, K; at least 1.
             * @param out Holds `L - K + 1` samples per row; every one is written.
             * @see kernels/README.md
             */
            template<typename T>
            void convolve_1d(int64_t rows, int64_t signal_length, int64_t tap_count,
                             const T* signal, const T* taps, T* out)
            {
                using RW = ReferenceAccumulator<T>;
                const int64_t out_length = Indexing::convolve_out_length(signal_length, tap_count);

                for (int64_t row = 0; row < rows; ++row)
                {
                    const T* row_signal = signal + row * signal_length;
                    T* row_out = out + row * out_length;

                    for (int64_t n = 0; n < out_length; ++n)
                    {
                        // The taps are added in ascending k order; see kernels/README.md.
                        ReferenceSum<T> sum;

                        for (int64_t k = 0; k < tap_count; ++k)
                            sum.add(RW(row_signal[Indexing::convolve_source(n, tap_count, k)]) * RW(taps[k]));

                        row_out[n] = static_cast<T>(sum.value());
                    }
                }
            }
        }
    }
}
