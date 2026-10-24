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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */

/**
 * @file convolve_1d.hpp
 * @brief Defines the SYCL kernel of the valid-mode one-dimensional convolution.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Writes `out[n] = sum_k signal[n + K - 1 - k] * taps[k]` per sample.
             * @param rows The dense rows, back to back in signal and out; one work-item per sample.
             * @param signal_length The samples in one signal row, L; at least tap_count.
             * @param tap_count The taps, K; at least 1.
             * @param out Holds `L - K + 1` samples per row; every one is written.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event convolve_1d(sycl::queue& q, int64_t rows, int64_t signal_length,
                                    int64_t tap_count, const T* signal, const T* taps, T* out,
                                    const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                const int64_t out_length = signal_length - tap_count + 1;
                const int64_t total = rows * out_length;
                if (total <= 0)
                    return sycl::event();

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            // Neighbouring work-items take neighbouring samples, so reads coalesce.
                            const int64_t row = i / out_length;
                            const int64_t n = i % out_length;
                            const T* row_signal = signal + row * signal_length;

                            // The taps are added in ascending k order on every device.
                            W total_value = W(0);
                            for (int64_t k = 0; k < tap_count; ++k)
                                total_value += W(row_signal[n + tap_count - 1 - k]) * W(taps[k]);

                            out[i] = static_cast<T>(total_value);
                        }
                    });
                });
            }
        }
    }
}
