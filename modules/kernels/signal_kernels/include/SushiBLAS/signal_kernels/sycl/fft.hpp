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
 * @brief Defines the SYCL kernels of the batched forward and inverse FFT.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/launch/submitted_events.hpp>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Holds the device addresses one transform reads its tables and scratch from.
             * @see modules/kernels/signal_kernels/README.md
             */
            template<typename T>
            struct FftDeviceTables
            {
                    FftLineView<T> row;        /**< Row line over the device tables. */
                    FftLineView<T> column;     /**< Column line; length 0 for one row. */
                    const T* half_roots = nullptr; /**< Real packing roots, when packed. */
                    T* lines = nullptr;        /**< Real-row line scratch. */
                    T* spectrum = nullptr;     /**< Copy of the half spectrum, real inverse. */
                    T* padded = nullptr;       /**< Bluestein's padded lines. */
            };

            /**
             * @brief Returns the bytes of device scratch a transform of @p batch entries needs.
             * @note The scratch holds, in order: the T tables, the line scratch, the spectrum copy,
             *       the padded lines and the index tables.
             */
            template<typename T>
            [[nodiscard]] std::size_t fft_scratch_bytes(const FftPlan<T>& plan,
                                                        std::size_t batch) noexcept
            {
                const std::size_t spectrum =
                    plan.pairing() == FftPairing::RealToComplex && plan.rows() > 1
                        ? batch * plan.spectrum_scalars() : 0;
                const std::size_t scalars = plan.values().size() + plan.line_scalars(batch) +
                                            spectrum + plan.padded_scalars(batch);
                return scalars * sizeof(T) + plan.indices().size() * sizeof(std::uint32_t);
            }

            namespace Detail
            {
                /**
                 * @brief Holds how a set of strided lines sits in memory, in complex elements.
                 * @note Line l is line l % per_group of group l / per_group.
                 */
                struct LineLayout
                {
                        std::size_t lines = 0;          /**< Lines transformed together. */
                        std::size_t per_group = 1;      /**< Lines in one evenly strided group. */
                        std::size_t group_stride = 0;   /**< Distance between groups. */
                        std::size_t line_stride = 0;    /**< Distance between lines of a group. */
                        std::size_t element_stride = 1; /**< Distance between a line's elements. */

                        /** @brief Returns the complex offset of element @p k of line @p l. */
                        [[nodiscard]] std::size_t at(std::size_t l, std::size_t k) const noexcept
                        {
                            return (l / per_group) * group_stride + (l % per_group) * line_stride +
                                   k * element_stride;
                        }
                };

                /** @brief Returns the layout of @p lines contiguous lines @p length apart. */
                [[nodiscard]] inline LineLayout contiguous_lines(std::size_t lines,
                                                                 std::size_t length) noexcept
                {
                    LineLayout layout;
                    layout.lines = lines;
                    layout.per_group = lines == 0 ? 1 : lines;
                    layout.line_stride = length;
                    return layout;
                }

                /**
                 * @brief Runs body(i) once for every i below count on the queue.
                 * @param body Device-copyable callable taking std::size_t.
                 */
                template<typename Body>
                sycl::event for_each_index(sycl::queue& q, Device::SubmittedEvents& events,
                                           std::size_t count,
                                           const std::vector<sycl::event>& deps, Body body)
                {
                    if (count == 0)
                    {
                        const sycl::event empty = q.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            h.single_task([] {});
                        });
                        events.append(empty);
                        return empty;
                    }

                    const Device::Grid1D grid = Device::plan_1d(Device::profile_for(q), count);
                    const std::size_t stride = grid.stride();
                    const sycl::event submitted = q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                        {
                            for (std::size_t i = it.get_global_id(0); i < count; i += stride)
                                body(i);
                        });
                    });
                    events.append(submitted);
                    return submitted;
                }

                /** @brief Runs the radix-2 transform of every line of @p layout in place. */
                template<typename T>
                sycl::event run_power_of_two(sycl::queue& q, Device::SubmittedEvents& events,
                                             const FftLineView<T>& line, T* data,
                                             const LineLayout& layout, FftDirection direction,
                                             sycl::event dep)
                {
                    const std::size_t size = line.power_size;
                    const std::uint32_t* reversal = line.reversal;
                    const T* twiddles = line.twiddles;

                    sycl::event last = for_each_index(q, events, layout.lines * size, {dep},
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / size;
                            const std::size_t i = flat % size;
                            const std::size_t j = reversal[i];
                            if (i >= j)
                                return;
                            T* a = data + 2 * layout.at(l, i);
                            T* b = data + 2 * layout.at(l, j);
                            const T ar = a[0];
                            const T ai = a[1];
                            a[0] = b[0];
                            a[1] = b[1];
                            b[0] = ar;
                            b[1] = ai;
                        });

                    const T sign = direction == FftDirection::Inverse ? T(-1) : T(1);
                    const std::size_t butterflies = size / 2;
                    for (std::size_t span = 2; span <= size; span <<= 1)
                    {
                        const std::size_t half = span / 2;
                        const std::size_t stride = size / span;
                        last = for_each_index(q, events, layout.lines * butterflies, {last},
                            [=](std::size_t flat)
                            {
                                const std::size_t l = flat / butterflies;
                                const std::size_t b = flat % butterflies;
                                const std::size_t start = (b / half) * span;
                                const std::size_t k = b % half;
                                const T wr = twiddles[2 * (k * stride)];
                                const T wi = sign * twiddles[2 * (k * stride) + 1];
                                T* u = data + 2 * layout.at(l, start + k);
                                T* v = data + 2 * layout.at(l, start + k + half);
                                T vr;
                                T vi;
                                Kernels::Detail::complex_multiply(v[0], v[1], wr, wi, vr, vi);
                                const T ur = u[0];
                                const T ui = u[1];
                                u[0] = ur + vr;
                                u[1] = ui + vi;
                                v[0] = ur - vr;
                                v[1] = ui - vi;
                            });
                    }
                    return last;
                }

                /**
                 * @brief Transforms every line of @p layout in place, by radix-2 or Bluestein.
                 * @param padded layout.lines * line.power_size complex elements of scratch.
                 */
                template<typename T>
                sycl::event run_lines(sycl::queue& q, Device::SubmittedEvents& events,
                                      const FftLineView<T>& line, T* data,
                                      const LineLayout& layout, T* padded, FftDirection direction,
                                      sycl::event dep)
                {
                    if (line.length == 0 || layout.lines == 0)
                        return dep;
                    if (!line.is_bluestein())
                        return run_power_of_two(q, events, line, data, layout, direction, dep);

                    const bool inverse = direction == FftDirection::Inverse;
                    const std::size_t length = line.length;
                    const std::size_t size = line.power_size;
                    const T* chirp = line.chirp;
                    const T* kernel = line.kernel;
                    const LineLayout work = contiguous_lines(layout.lines, size);

                    sycl::event last = for_each_index(q, events, layout.lines * size, {dep},
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / size;
                            const std::size_t k = flat % size;
                            T* w = padded + 2 * flat;
                            if (k >= length)
                            {
                                w[0] = T(0);
                                w[1] = T(0);
                                return;
                            }
                            const T* x = data + 2 * layout.at(l, k);
                            const T xi = inverse ? -x[1] : x[1];
                            Kernels::Detail::complex_multiply(x[0], xi, chirp[2 * k],
                                                              chirp[2 * k + 1], w[0], w[1]);
                        });

                    last = run_power_of_two(q, events, line, padded, work, FftDirection::Forward, last);
                    last = for_each_index(q, events, layout.lines * size, {last},
                        [=](std::size_t flat)
                        {
                            const std::size_t k = flat % size;
                            T* w = padded + 2 * flat;
                            const T wr = w[0];
                            const T wi = w[1];
                            Kernels::Detail::complex_multiply(wr, wi, kernel[2 * k],
                                                              kernel[2 * k + 1], w[0], w[1]);
                        });
                    last = run_power_of_two(q, events, line, padded, work, FftDirection::Inverse, last);

                    const T scale = T(1) / static_cast<T>(size);
                    return for_each_index(q, events, layout.lines * length, {last},
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / length;
                            const std::size_t k = flat % length;
                            const T* w = padded + 2 * (l * size + k);
                            T yr;
                            T yi;
                            Kernels::Detail::complex_multiply(w[0], w[1], chirp[2 * k],
                                                              chirp[2 * k + 1], yr, yi);
                            yr *= scale;
                            yi *= scale;
                            T* x = data + 2 * layout.at(l, k);
                            x[0] = yr;
                            x[1] = inverse ? -yi : yi;
                        });
                }

                /** @brief Returns the layout of the columns of @p batch spectra. */
                template<typename T>
                [[nodiscard]] LineLayout column_layout(const FftPlan<T>& plan,
                                                       std::size_t batch) noexcept
                {
                    LineLayout layout;
                    layout.lines = batch * plan.spectrum_columns();
                    layout.per_group = plan.spectrum_columns();
                    layout.group_stride = plan.rows() * plan.spectrum_columns();
                    layout.line_stride = 1;
                    layout.element_stride = plan.spectrum_columns();
                    return layout;
                }
            } // namespace Detail

            /**
             * @brief Returns the device views of a plan whose tables were copied to @p scratch.
             * @param scratch fft_scratch_bytes(plan, batch) bytes of device memory.
             */
            template<typename T>
            [[nodiscard]] FftDeviceTables<T> fft_device_tables(const FftPlan<T>& plan,
                                                               std::size_t batch,
                                                               void* scratch) noexcept
            {
                T* values = static_cast<T*>(scratch);
                FftDeviceTables<T> tables;
                tables.lines = values + plan.values().size();
                const std::size_t spectrum =
                    plan.pairing() == FftPairing::RealToComplex && plan.rows() > 1
                        ? batch * plan.spectrum_scalars() : 0;
                tables.spectrum = tables.lines + plan.line_scalars(batch);
                tables.padded = tables.spectrum + spectrum;
                const auto* indices = reinterpret_cast<const std::uint32_t*>(
                    tables.padded + plan.padded_scalars(batch));
                tables.row = plan.row_line(values, indices);
                tables.column = plan.column_line(values, indices);
                tables.half_roots = plan.half_roots(values);
                return tables;
            }

            /**
             * @brief Copies the plan's tables to the head and tail of @p scratch.
             * @pre The plan outlives the returned events.
             * @return The events of every copy, in order.
             */
            template<typename T>
            Device::SubmittedEvents upload_fft_tables(sycl::queue& q, const FftPlan<T>& plan,
                                                      std::size_t batch, void* scratch,
                                                      const std::vector<sycl::event>& deps)
            {
                const FftDeviceTables<T> tables = fft_device_tables(plan, batch, scratch);
                auto* indices =
                    reinterpret_cast<std::uint32_t*>(tables.padded + plan.padded_scalars(batch));
                const sycl::event values = q.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.memcpy(scratch, plan.values().data(), plan.values().size() * sizeof(T));
                });
                Device::SubmittedEvents uploaded{std::vector<sycl::event>{values}};
                if (plan.indices().empty())
                    return uploaded;
                uploaded.append(q.submit([&](sycl::handler& h)
                {
                    h.depends_on(values);
                    h.memcpy(indices, plan.indices().data(),
                             plan.indices().size() * sizeof(std::uint32_t));
                }));
                return uploaded;
            }

            /**
             * @brief Writes the unscaled forward transform of each of @p batch entries.
             * @param tables What fft_device_tables returned over uploaded tables.
             * @param output Shares no byte with @p input.
             * @see modules/kernels/signal_kernels/README.md
             */
            template<typename T>
            Device::SubmittedEvents fft_forward(sycl::queue& q, const FftPlan<T>& plan,
                                                const FftDeviceTables<T>& tables, const T* input,
                                                T* output, std::size_t batch,
                                                const std::vector<sycl::event>& deps)
            {
                const std::size_t rows = plan.rows();
                const std::size_t columns = plan.columns();
                const std::size_t width = plan.spectrum_columns();
                const std::size_t row_lines = batch * rows;
                sycl::event last;
                Device::SubmittedEvents events;

                if (plan.pairing() == FftPairing::ComplexToComplex)
                {
                    last = q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.memcpy(output, input, batch * plan.sample_scalars() * sizeof(T));
                    });
                    events.append(last);
                    last = Detail::run_lines(q, events, tables.row, output,
                                             Detail::contiguous_lines(row_lines, columns),
                                             tables.padded, FftDirection::Forward, last);
                }
                else
                {
                    const std::size_t length = tables.row.length;
                    const bool packed = plan.is_packed();
                    T* lines = tables.lines;
                    last = Detail::for_each_index(q, events, row_lines * length, deps,
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / length;
                            const std::size_t k = flat % length;
                            const T* row = input + l * columns;
                            lines[2 * flat] = packed ? row[2 * k] : row[k];
                            lines[2 * flat + 1] = packed ? row[2 * k + 1] : T(0);
                        });
                    last = Detail::run_lines(q, events, tables.row, lines,
                                             Detail::contiguous_lines(row_lines, length),
                                             tables.padded, FftDirection::Forward, last);

                    const T* roots = tables.half_roots;
                    last = Detail::for_each_index(q, events, row_lines * width, {last},
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / width;
                            const std::size_t k = flat % width;
                            const T* line = lines + 2 * l * length;
                            T* out = output + 2 * flat;
                            if (!packed)
                            {
                                out[0] = line[2 * k];
                                out[1] = line[2 * k + 1];
                                return;
                            }
                            const std::size_t half = length;
                            const T zr = line[2 * (k % half)];
                            const T zi = line[2 * (k % half) + 1];
                            const T mr = line[2 * ((half - k) % half)];
                            const T mi = -line[2 * ((half - k) % half) + 1];
                            const T even_r = (zr + mr) * T(0.5);
                            const T even_i = (zi + mi) * T(0.5);
                            T odd_r;
                            T odd_i;
                            Kernels::Detail::complex_multiply(zr - mr, zi - mi, T(0), T(-0.5),
                                                              odd_r, odd_i);
                            T turned_r;
                            T turned_i;
                            Kernels::Detail::complex_multiply(roots[2 * k], roots[2 * k + 1],
                                                              odd_r, odd_i, turned_r, turned_i);
                            out[0] = even_r + turned_r;
                            out[1] = even_i + turned_i;
                        });
                }

                Detail::run_lines(q, events, tables.column, output,
                                  Detail::column_layout(plan, batch), tables.padded,
                                  FftDirection::Forward, last);
                return events;
            }

            /**
             * @brief Writes the inverse of fft_forward, scaled by 1 / (rows * columns).
             * @param tables What fft_device_tables returned over uploaded tables.
             * @param output Shares no byte with @p input.
             * @see modules/kernels/signal_kernels/README.md
             */
            template<typename T>
            Device::SubmittedEvents fft_inverse(sycl::queue& q, const FftPlan<T>& plan,
                                                const FftDeviceTables<T>& tables, const T* input,
                                                T* output, std::size_t batch,
                                                const std::vector<sycl::event>& deps)
            {
                const std::size_t rows = plan.rows();
                const std::size_t columns = plan.columns();
                const std::size_t width = plan.spectrum_columns();
                const std::size_t row_lines = batch * rows;
                const std::size_t samples = batch * plan.sample_scalars();
                sycl::event last;
                Device::SubmittedEvents events;

                if (plan.pairing() == FftPairing::ComplexToComplex)
                {
                    last = q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        h.memcpy(output, input, samples * sizeof(T));
                    });
                    events.append(last);
                    last = Detail::run_lines(q, events, tables.column, output,
                                             Detail::column_layout(plan, batch), tables.padded,
                                             FftDirection::Inverse, last);
                    last = Detail::run_lines(q, events, tables.row, output,
                                             Detail::contiguous_lines(row_lines, columns),
                                             tables.padded, FftDirection::Inverse, last);
                }
                else
                {
                    const T* spectrum = input;
                    if (rows > 1)
                    {
                        T* copy = tables.spectrum;
                        last = q.submit([&](sycl::handler& h)
                        {
                            h.depends_on(deps);
                            h.memcpy(copy, input, batch * plan.spectrum_scalars() * sizeof(T));
                        });
                        events.append(last);
                        last = Detail::run_lines(q, events, tables.column, copy,
                                                 Detail::column_layout(plan, batch), tables.padded,
                                                 FftDirection::Inverse, last);
                        spectrum = copy;
                    }

                    const std::size_t length = tables.row.length;
                    const bool packed = plan.is_packed();
                    const T* roots = tables.half_roots;
                    // A packed line has the bytes of its real row, so it is built in the output.
                    T* lines = packed ? output : tables.lines;
                    const std::vector<sycl::event> start =
                        rows > 1 ? std::vector<sycl::event>{last} : deps;
                    last = Detail::for_each_index(q, events, row_lines * length, start,
                        [=](std::size_t flat)
                        {
                            const std::size_t l = flat / length;
                            const std::size_t k = flat % length;
                            const T* in = spectrum + 2 * l * width;
                            T* line = lines + 2 * flat;
                            if (!packed)
                            {
                                const bool stored = k < width;
                                const std::size_t source = stored ? k : columns - k;
                                line[0] = in[2 * source];
                                line[1] = stored ? in[2 * source + 1] : -in[2 * source + 1];
                                return;
                            }
                            const std::size_t half = length;
                            const T mr = in[2 * (half - k)];
                            const T mi = -in[2 * (half - k) + 1];
                            const T even_r = in[2 * k] + mr;
                            const T even_i = in[2 * k + 1] + mi;
                            T odd_r;
                            T odd_i;
                            Kernels::Detail::complex_multiply(in[2 * k] - mr, in[2 * k + 1] - mi,
                                                              roots[2 * k], -roots[2 * k + 1],
                                                              odd_r, odd_i);
                            T turned_r;
                            T turned_i;
                            Kernels::Detail::complex_multiply(T(0), T(1), odd_r, odd_i, turned_r,
                                                              turned_i);
                            line[0] = even_r + turned_r;
                            line[1] = even_i + turned_i;
                        });
                    last = Detail::run_lines(q, events, tables.row, lines,
                                             Detail::contiguous_lines(row_lines, length),
                                             tables.padded, FftDirection::Inverse, last);
                    if (!packed)
                    {
                        const T* unpacked = tables.lines;
                        last = Detail::for_each_index(q, events, row_lines * columns, {last},
                            [=](std::size_t flat)
                            {
                                output[flat] = unpacked[2 * flat];
                            });
                    }
                }

                const T scale = T(1) / static_cast<T>(rows * columns);
                Detail::for_each_index(q, events, samples, {last},
                    [=](std::size_t i)
                    {
                        output[i] *= scale;
                    });
                return events;
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
