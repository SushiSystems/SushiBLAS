/**************************************************************************/
/* fft.hpp                                                                */
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
 * @file fft.hpp
 * @brief Defines the reference kernels of the batched forward and inverse FFT.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/signal_kernels/fft_line_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /** @brief Holds the host scratch of one reference transform. */
                template<typename T>
                struct FftHostScratch
                {
                        /** @brief Sizes every line for @p plan. */
                        explicit FftHostScratch(const FftPlan<T>& plan)
                            : line(2 * std::max(plan.rows(), plan.columns())),
                              work(2 * padded_length(plan)), spectrum(plan.spectrum_scalars())
                        {
                        }

                        /** @brief Returns the longer radix-2 length of the plan's two lines. */
                        static std::size_t padded_length(const FftPlan<T>& plan) noexcept
                        {
                            const T* values = plan.values().data();
                            const std::uint32_t* indices = plan.indices().data();
                            return std::max(plan.row_line(values, indices).power_size,
                                            plan.column_line(values, indices).power_size);
                        }

                        std::vector<T> line;     /**< One row or column, interleaved. */
                        std::vector<T> work;     /**< Bluestein's padded line. */
                        std::vector<T> spectrum; /**< Copy of one half spectrum. */
                };

                /** @brief Transforms every column of the spectrum_columns-wide @p data in place. */
                template<typename T>
                void run_columns(const FftPlan<T>& plan, T* data, FftHostScratch<T>& scratch,
                                 FftDirection direction)
                {
                    const FftLineView<T> column =
                        plan.column_line(plan.values().data(), plan.indices().data());
                    if (column.length == 0)
                        return;
                    const std::size_t width = plan.spectrum_columns();
                    for (std::size_t c = 0; c < width; ++c)
                    {
                        for (std::size_t r = 0; r < plan.rows(); ++r)
                        {
                            scratch.line[2 * r] = data[2 * (r * width + c)];
                            scratch.line[2 * r + 1] = data[2 * (r * width + c) + 1];
                        }
                        Kernels::Detail::run_line(column, scratch.line.data(), scratch.work.data(),
                                                  direction);
                        for (std::size_t r = 0; r < plan.rows(); ++r)
                        {
                            data[2 * (r * width + c)] = scratch.line[2 * r];
                            data[2 * (r * width + c) + 1] = scratch.line[2 * r + 1];
                        }
                    }
                }

                /** @brief Writes the half spectrum of the real row @p samples into @p out. */
                template<typename T>
                void forward_real_row(const FftPlan<T>& plan, const T* samples, T* out,
                                      FftHostScratch<T>& scratch)
                {
                    const FftLineView<T> row =
                        plan.row_line(plan.values().data(), plan.indices().data());
                    T* line = scratch.line.data();
                    if (!plan.is_packed())
                    {
                        for (std::size_t k = 0; k < plan.columns(); ++k)
                        {
                            line[2 * k] = samples[k];
                            line[2 * k + 1] = T(0);
                        }
                        Kernels::Detail::run_line(row, line, scratch.work.data(),
                                                  FftDirection::Forward);
                        std::copy(line, line + 2 * plan.spectrum_columns(), out);
                        return;
                    }

                    const std::size_t half = plan.columns() / 2;
                    std::copy(samples, samples + 2 * half, line);
                    Kernels::Detail::run_line(row, line, scratch.work.data(),
                                              FftDirection::Forward);
                    const T* roots = plan.half_roots(plan.values().data());
                    for (std::size_t k = 0; k <= half; ++k)
                    {
                        const std::size_t own = k % half;
                        const std::size_t mirror = Indexing::mirror_bin(k, half);
                        Kernels::Detail::unpack_forward_bin(
                            line[2 * own], line[2 * own + 1], line[2 * mirror],
                            -line[2 * mirror + 1], roots[2 * k], roots[2 * k + 1], out[2 * k],
                            out[2 * k + 1]);
                    }
                }

                /** @brief Writes rows * columns times the real row whose half spectrum is @p in. */
                template<typename T>
                void inverse_real_row(const FftPlan<T>& plan, const T* in, T* samples,
                                      FftHostScratch<T>& scratch)
                {
                    const FftLineView<T> row =
                        plan.row_line(plan.values().data(), plan.indices().data());
                    T* line = scratch.line.data();
                    const std::size_t columns = plan.columns();
                    if (!plan.is_packed())
                    {
                        for (std::size_t k = 0; k < columns; ++k)
                        {
                            const Indexing::HermitianSource from =
                                Indexing::hermitian_source(k, plan.spectrum_columns(), columns);
                            line[2 * k] = in[2 * from.bin];
                            line[2 * k + 1] = from.conjugated ? -in[2 * from.bin + 1]
                                                              : in[2 * from.bin + 1];
                        }
                        Kernels::Detail::run_line(row, line, scratch.work.data(),
                                                  FftDirection::Inverse);
                        for (std::size_t k = 0; k < columns; ++k)
                            samples[k] = line[2 * k];
                        return;
                    }

                    const std::size_t half = columns / 2;
                    const T* roots = plan.half_roots(plan.values().data());
                    for (std::size_t k = 0; k < half; ++k)
                    {
                        Kernels::Detail::pack_inverse_bin(
                            in[2 * k], in[2 * k + 1], in[2 * (half - k)],
                            -in[2 * (half - k) + 1], roots[2 * k], roots[2 * k + 1],
                            line[2 * k], line[2 * k + 1]);
                    }
                    Kernels::Detail::run_line(row, line, scratch.work.data(),
                                              FftDirection::Inverse);
                    std::copy(line, line + 2 * half, samples);
                }
            } // namespace Detail

            /**
             * @brief Writes the unscaled forward transform of each of @p batch entries.
             * @param output Shares no byte with @p input; laid out as plan.spectrum_scalars().
             * @see modules/kernels/signal_kernels/README.md
             */
            template<typename T>
            void fft_forward(const FftPlan<T>& plan, const T* input, T* output, std::size_t batch)
            {
                Detail::FftHostScratch<T> scratch(plan);
                const FftLineView<T> row =
                    plan.row_line(plan.values().data(), plan.indices().data());
                const std::size_t width = plan.spectrum_columns();
                for (std::size_t entry = 0; entry < batch; ++entry)
                {
                    const T* samples = input + entry * plan.sample_scalars();
                    T* spectrum = output + entry * plan.spectrum_scalars();
                    for (std::size_t r = 0; r < plan.rows(); ++r)
                    {
                        T* out_row = spectrum + 2 * r * width;
                        if (plan.pairing() == FftPairing::RealToComplex)
                        {
                            Detail::forward_real_row(plan, samples + r * plan.columns(), out_row,
                                                     scratch);
                            continue;
                        }
                        const T* in_row = samples + 2 * r * plan.columns();
                        std::copy(in_row, in_row + 2 * plan.columns(), out_row);
                        Kernels::Detail::run_line(row, out_row, scratch.work.data(),
                                                  FftDirection::Forward);
                    }
                    Detail::run_columns(plan, spectrum, scratch, FftDirection::Forward);
                }
            }

            /**
             * @brief Writes the inverse of fft_forward, scaled by 1 / (rows * columns).
             * @param output Shares no byte with @p input; laid out as plan.sample_scalars().
             * @see modules/kernels/signal_kernels/README.md
             */
            template<typename T>
            void fft_inverse(const FftPlan<T>& plan, const T* input, T* output, std::size_t batch)
            {
                Detail::FftHostScratch<T> scratch(plan);
                const FftLineView<T> row =
                    plan.row_line(plan.values().data(), plan.indices().data());
                const T scale = T(1) / static_cast<T>(plan.rows() * plan.columns());
                const std::size_t width = plan.spectrum_columns();
                for (std::size_t entry = 0; entry < batch; ++entry)
                {
                    const T* spectrum = input + entry * plan.spectrum_scalars();
                    T* samples = output + entry * plan.sample_scalars();
                    if (plan.pairing() == FftPairing::RealToComplex)
                    {
                        std::copy(spectrum, spectrum + plan.spectrum_scalars(),
                                  scratch.spectrum.begin());
                        Detail::run_columns(plan, scratch.spectrum.data(), scratch,
                                            FftDirection::Inverse);
                        for (std::size_t r = 0; r < plan.rows(); ++r)
                            Detail::inverse_real_row(plan, scratch.spectrum.data() + 2 * r * width,
                                                     samples + r * plan.columns(), scratch);
                    }
                    else
                    {
                        std::copy(spectrum, spectrum + plan.spectrum_scalars(), samples);
                        Detail::run_columns(plan, samples, scratch, FftDirection::Inverse);
                        for (std::size_t r = 0; r < plan.rows(); ++r)
                            Kernels::Detail::run_line(row, samples + 2 * r * plan.columns(),
                                                      scratch.work.data(), FftDirection::Inverse);
                    }
                    for (std::size_t i = 0; i < plan.sample_scalars(); ++i)
                        samples[i] *= scale;
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
