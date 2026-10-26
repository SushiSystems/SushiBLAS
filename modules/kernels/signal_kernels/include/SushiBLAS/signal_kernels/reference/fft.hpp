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
                        const T zr = line[2 * (k % half)];
                        const T zi = line[2 * (k % half) + 1];
                        const T mr = line[2 * ((half - k) % half)];
                        const T mi = -line[2 * ((half - k) % half) + 1];
                        const T even_r = (zr + mr) * T(0.5);
                        const T even_i = (zi + mi) * T(0.5);
                        T odd_r;
                        T odd_i;
                        Kernels::Detail::complex_multiply(zr - mr, zi - mi, T(0), T(-0.5), odd_r,
                                                          odd_i);
                        T turned_r;
                        T turned_i;
                        Kernels::Detail::complex_multiply(roots[2 * k], roots[2 * k + 1], odd_r,
                                                          odd_i, turned_r, turned_i);
                        out[2 * k] = even_r + turned_r;
                        out[2 * k + 1] = even_i + turned_i;
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
                            const bool stored = k < plan.spectrum_columns();
                            const std::size_t source = stored ? k : columns - k;
                            line[2 * k] = in[2 * source];
                            line[2 * k + 1] = stored ? in[2 * source + 1] : -in[2 * source + 1];
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
                        const T mr = in[2 * (half - k)];
                        const T mi = -in[2 * (half - k) + 1];
                        const T even_r = in[2 * k] + mr;
                        const T even_i = in[2 * k + 1] + mi;
                        T odd_r;
                        T odd_i;
                        Kernels::Detail::complex_multiply(in[2 * k] - mr, in[2 * k + 1] - mi,
                                                          roots[2 * k], -roots[2 * k + 1], odd_r,
                                                          odd_i);
                        T turned_r;
                        T turned_i;
                        Kernels::Detail::complex_multiply(T(0), T(1), odd_r, odd_i, turned_r,
                                                          turned_i);
                        line[2 * k] = even_r + turned_r;
                        line[2 * k + 1] = even_i + turned_i;
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
