/**************************************************************************/
/* fft_plan.hpp                                                           */
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
 * @file fft_plan.hpp
 * @brief Declares FftPlan, the tables and scratch layout one batched FFT shape cooks once.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <SushiBLAS/arithmetic/no_contraction.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /** @brief Names the element pairing of a transform's operands. */
        enum class FftPairing : std::uint8_t
        {
            ComplexToComplex, /**< Both sides complex, interleaved real then imaginary. */
            RealToComplex,    /**< Real samples and the columns / 2 + 1 half spectrum per row. */
        };

        /** @brief Names the sign of a transform's exponent. */
        enum class FftDirection : std::uint8_t
        {
            Forward, /**< exp(-2 pi i j k / n). */
            Inverse, /**< exp(+2 pi i j k / n), unscaled. */
        };

        /**
         * @brief Holds the addresses of one cooked line transform on the side that owns them.
         * @see modules/kernels/signal_kernels/README.md
         */
        template<typename T>
        struct FftLineView
        {
                std::size_t length = 0;               /**< Transform length. */
                std::size_t power_size = 0;           /**< Radix-2 length: length or the padding. */
                const std::uint32_t* reversal = nullptr; /**< Bit-reversed index per position. */
                const T* twiddles = nullptr;          /**< exp(-2 pi i k / power_size). */
                const T* chirp = nullptr;             /**< exp(-i pi k^2 / length); null for 2^m. */
                const T* kernel = nullptr;            /**< Transformed conjugate chirp, padded. */

                /** @brief Returns true when the line runs Bluestein's chirp-z transform. */
                [[nodiscard]] bool is_bluestein() const noexcept
                {
                    return chirp != nullptr;
                }
        };

        namespace Detail
        {
            /**
             * @brief Writes a * b into out without the special-value handling of std::complex.
             * @note Each of the four products is rounded before its sum; nothing is fused.
             */
            template<typename T>
            inline void complex_multiply(T ar, T ai, T br, T bi, T& out_r, T& out_i) noexcept
            {
                SB_FP_CONTRACT_OFF
                const T real_real = ar * br;
                const T imag_imag = ai * bi;
                const T real_imag = ar * bi;
                const T imag_real = ai * br;
                out_r = real_real - imag_imag;
                out_i = real_imag + imag_real;
            }

            /**
             * @brief Runs the iterative radix-2 transform of one contiguous interleaved line.
             * @param data line.power_size complex elements, transformed in place.
             */
            template<typename T>
            inline void run_power_of_two_line(const FftLineView<T>& line, T* data,
                                              FftDirection direction) noexcept
            {
                const std::size_t size = line.power_size;
                for (std::size_t i = 0; i < size; ++i)
                {
                    const std::size_t j = line.reversal[i];
                    if (i < j)
                    {
                        std::swap(data[2 * i], data[2 * j]);
                        std::swap(data[2 * i + 1], data[2 * j + 1]);
                    }
                }

                const T sign = direction == FftDirection::Inverse ? T(-1) : T(1);
                for (std::size_t span = 2; span <= size; span <<= 1)
                {
                    const std::size_t half = span / 2;
                    const std::size_t stride = size / span;
                    for (std::size_t start = 0; start < size; start += span)
                        for (std::size_t k = 0; k < half; ++k)
                        {
                            const T wr = line.twiddles[2 * (k * stride)];
                            const T wi = sign * line.twiddles[2 * (k * stride) + 1];
                            T* u = data + 2 * (start + k);
                            T* v = data + 2 * (start + k + half);
                            T vr;
                            T vi;
                            complex_multiply(v[0], v[1], wr, wi, vr, vi);
                            const T ur = u[0];
                            const T ui = u[1];
                            u[0] = ur + vr;
                            u[1] = ui + vi;
                            v[0] = ur - vr;
                            v[1] = ui - vi;
                        }
                }
            }

            /**
             * @brief Transforms one contiguous interleaved line of line.length elements in place.
             * @param work line.power_size complex elements of scratch; unused for a 2^m line.
             */
            template<typename T>
            inline void run_line(const FftLineView<T>& line, T* data, T* work,
                                 FftDirection direction) noexcept
            {
                if (!line.is_bluestein())
                {
                    run_power_of_two_line(line, data, direction);
                    return;
                }

                const bool inverse = direction == FftDirection::Inverse;
                const std::size_t padded = line.power_size;
                for (std::size_t k = 0; k < padded; ++k)
                {
                    if (k >= line.length)
                    {
                        work[2 * k] = T(0);
                        work[2 * k + 1] = T(0);
                        continue;
                    }
                    const T xi = inverse ? -data[2 * k + 1] : data[2 * k + 1];
                    complex_multiply(data[2 * k], xi, line.chirp[2 * k], line.chirp[2 * k + 1],
                                     work[2 * k], work[2 * k + 1]);
                }

                run_power_of_two_line(line, work, FftDirection::Forward);
                for (std::size_t k = 0; k < padded; ++k)
                {
                    const T wr = work[2 * k];
                    const T wi = work[2 * k + 1];
                    complex_multiply(wr, wi, line.kernel[2 * k], line.kernel[2 * k + 1],
                                     work[2 * k], work[2 * k + 1]);
                }
                run_power_of_two_line(line, work, FftDirection::Inverse);

                const T scale = T(1) / static_cast<T>(padded);
                for (std::size_t k = 0; k < line.length; ++k)
                {
                    T yr;
                    T yi;
                    complex_multiply(work[2 * k], work[2 * k + 1], line.chirp[2 * k],
                                     line.chirp[2 * k + 1], yr, yi);
                    yr *= scale;
                    yi *= scale;
                    data[2 * k] = yr;
                    data[2 * k + 1] = inverse ? -yi : yi;
                }
            }
        } // namespace Detail

        /**
         * @brief Holds the cooked tables of one rows by columns transform and its scratch layout.
         * @tparam T float or double.
         * @see modules/kernels/signal_kernels/README.md
         */
        template<typename T>
        class FftPlan final
        {
            public:
                /**
                 * @brief Cooks the tables of a rows by columns transform.
                 * @pre rows and columns are at least 1.
                 */
                FftPlan(std::size_t rows, std::size_t columns, FftPairing pairing)
                    : rows_(rows), columns_(columns), pairing_(pairing),
                      packed_(pairing == FftPairing::RealToComplex && columns % 2 == 0)
                {
                    spectrum_columns_ =
                        pairing_ == FftPairing::RealToComplex ? columns_ / 2 + 1 : columns_;
                    row_line_ = cook_line(packed_ ? columns_ / 2 : columns_);
                    column_line_ = cook_line(rows_ > 1 ? rows_ : 0);
                    if (packed_)
                    {
                        half_roots_ = values_.size();
                        for (std::size_t k = 0; k <= columns_ / 2; ++k)
                            push_unit_root(2 * k, columns_);
                    }
                }

                /** @brief Returns the rows of each transform. */
                [[nodiscard]] std::size_t rows() const noexcept
                {
                    return rows_;
                }

                /** @brief Returns the samples per row. */
                [[nodiscard]] std::size_t columns() const noexcept
                {
                    return columns_;
                }

                /** @brief Returns the complex elements per spectrum row. */
                [[nodiscard]] std::size_t spectrum_columns() const noexcept
                {
                    return spectrum_columns_;
                }

                /** @brief Returns the element pairing of the operands. */
                [[nodiscard]] FftPairing pairing() const noexcept
                {
                    return pairing_;
                }

                /** @brief Returns true when a real row packs into a complex line of half length. */
                [[nodiscard]] bool is_packed() const noexcept
                {
                    return packed_;
                }

                /** @brief Returns the T elements one batch entry of the samples holds. */
                [[nodiscard]] std::size_t sample_scalars() const noexcept
                {
                    return pairing_ == FftPairing::RealToComplex ? rows_ * columns_
                                                                 : 2 * rows_ * columns_;
                }

                /** @brief Returns the T elements one batch entry of the spectrum holds. */
                [[nodiscard]] std::size_t spectrum_scalars() const noexcept
                {
                    return 2 * rows_ * spectrum_columns_;
                }

                /** @brief Returns the cooked T tables, laid out as the views address them. */
                [[nodiscard]] const std::vector<T>& values() const noexcept
                {
                    return values_;
                }

                /** @brief Returns the cooked index tables, laid out as the views address them. */
                [[nodiscard]] const std::vector<std::uint32_t>& indices() const noexcept
                {
                    return indices_;
                }

                /** @brief Returns the row line's view over tables at @p values and @p indices. */
                [[nodiscard]] FftLineView<T> row_line(const T* values,
                                                      const std::uint32_t* indices) const noexcept
                {
                    return view_of(row_line_, values, indices);
                }

                /** @brief Returns the column line's view; its length is 0 for one row. */
                [[nodiscard]] FftLineView<T>
                column_line(const T* values, const std::uint32_t* indices) const noexcept
                {
                    return view_of(column_line_, values, indices);
                }

                /** @brief Returns exp(-2 pi i k / columns) for k up to columns / 2, when packed. */
                [[nodiscard]] const T* half_roots(const T* values) const noexcept
                {
                    return packed_ ? values + half_roots_ : nullptr;
                }

                /** @brief Returns the T elements of real-row scratch @p batch entries need. */
                [[nodiscard]] std::size_t line_scalars(std::size_t batch) const noexcept
                {
                    if (pairing_ != FftPairing::RealToComplex)
                        return 0;
                    return 2 * batch * rows_ * row_line_.length;
                }

                /** @brief Returns the T elements of Bluestein scratch a batch of @p batch needs. */
                [[nodiscard]] std::size_t padded_scalars(std::size_t batch) const noexcept
                {
                    std::size_t largest = 0;
                    if (row_line_.chirp != K_ABSENT)
                        largest = batch * rows_ * row_line_.power_size;
                    if (column_line_.chirp != K_ABSENT)
                        largest = std::max(largest,
                                           batch * spectrum_columns_ * column_line_.power_size);
                    return 2 * largest;
                }

            private:
                /// @brief Marks a table a line does not carry.
                static constexpr std::size_t K_ABSENT = static_cast<std::size_t>(-1);

                /** @brief Holds where one line's tables start in values_ and indices_. */
                struct LineOffsets
                {
                        std::size_t length = 0;       /**< Transform length. */
                        std::size_t power_size = 0;   /**< Radix-2 length. */
                        std::size_t reversal = 0;     /**< Offset in indices_. */
                        std::size_t twiddles = 0;     /**< Offset in values_. */
                        std::size_t chirp = K_ABSENT; /**< Offset in values_, or K_ABSENT. */
                        std::size_t kernel = K_ABSENT; /**< Offset in values_, or K_ABSENT. */
                };

                /** @brief Appends exp(-i pi numerator / denominator), evaluated in double. */
                void push_unit_root(std::size_t numerator, std::size_t denominator)
                {
                    const double angle = -3.14159265358979323846 * static_cast<double>(numerator) /
                                         static_cast<double>(denominator);
                    values_.push_back(static_cast<T>(std::cos(angle)));
                    values_.push_back(static_cast<T>(std::sin(angle)));
                }

                /** @brief Cooks the radix-2 tables of length @p size into @p line. */
                void cook_power_of_two(LineOffsets& line, std::size_t size)
                {
                    line.power_size = size;
                    line.reversal = indices_.size();
                    std::size_t bits = 0;
                    while ((std::size_t{1} << bits) < size)
                        ++bits;
                    for (std::size_t i = 0; i < size; ++i)
                    {
                        std::size_t reversed = 0;
                        for (std::size_t b = 0; b < bits; ++b)
                            reversed |= ((i >> b) & 1u) << (bits - 1 - b);
                        indices_.push_back(static_cast<std::uint32_t>(reversed));
                    }
                    line.twiddles = values_.size();
                    for (std::size_t k = 0; k < size / 2; ++k)
                        push_unit_root(2 * k, size);
                }

                /** @brief Cooks the tables of a line of @p length; 0 makes an empty line. */
                LineOffsets cook_line(std::size_t length)
                {
                    LineOffsets line;
                    line.length = length;
                    if (length == 0)
                        return line;
                    if ((length & (length - 1)) == 0)
                    {
                        cook_power_of_two(line, length);
                        return line;
                    }

                    std::size_t padded = 1;
                    while (padded < 2 * length - 1)
                        padded <<= 1;
                    cook_power_of_two(line, padded);

                    line.chirp = values_.size();
                    for (std::size_t k = 0; k < length; ++k)
                        push_unit_root((k * k) % (2 * length), length);

                    line.kernel = values_.size();
                    values_.resize(values_.size() + 2 * padded, T(0));
                    T* chirp = values_.data() + line.chirp;
                    T* kernel = values_.data() + line.kernel;
                    kernel[0] = chirp[0];
                    kernel[1] = -chirp[1];
                    for (std::size_t k = 1; k < length; ++k)
                    {
                        kernel[2 * k] = chirp[2 * k];
                        kernel[2 * k + 1] = -chirp[2 * k + 1];
                        kernel[2 * (padded - k)] = chirp[2 * k];
                        kernel[2 * (padded - k) + 1] = -chirp[2 * k + 1];
                    }

                    FftLineView<T> radix = view_of(line, values_.data(), indices_.data());
                    radix.chirp = nullptr;
                    Detail::run_power_of_two_line(radix, kernel, FftDirection::Forward);
                    return line;
                }

                /** @brief Returns the view of @p line over tables at @p values and @p indices. */
                static FftLineView<T> view_of(const LineOffsets& line, const T* values,
                                              const std::uint32_t* indices) noexcept
                {
                    FftLineView<T> view;
                    view.length = line.length;
                    view.power_size = line.power_size;
                    if (line.length == 0)
                        return view;
                    view.reversal = indices + line.reversal;
                    view.twiddles = values + line.twiddles;
                    if (line.chirp != K_ABSENT)
                    {
                        view.chirp = values + line.chirp;
                        view.kernel = values + line.kernel;
                    }
                    return view;
                }

                /// @brief Rows of each transform.
                std::size_t rows_ = 1;

                /// @brief Samples per row.
                std::size_t columns_ = 0;

                /// @brief Complex elements per spectrum row.
                std::size_t spectrum_columns_ = 0;

                /// @brief Element pairing of the operands.
                FftPairing pairing_ = FftPairing::ComplexToComplex;

                /// @brief Whether real rows pack into a complex line of half the length.
                bool packed_ = false;

                /// @brief Tables of the row transform, columns or columns / 2 long.
                LineOffsets row_line_;

                /// @brief Tables of the column transform; empty for one row.
                LineOffsets column_line_;

                /// @brief Offset of the half roots in values_, when packed_.
                std::size_t half_roots_ = 0;

                /// @brief Every T table, interleaved complex.
                std::vector<T> values_;

                /// @brief Every bit-reversal table.
                std::vector<std::uint32_t> indices_;
        };
    } // namespace Kernels
} // namespace SushiBLAS
