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
#include <SushiBLAS/signal_kernels/fft_line_index.hpp>

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
                return plan.scratch_bytes(batch);
            }

            namespace Detail
            {
                /// @brief Names the layout of a set of strided lines.
                using LineLayout = Indexing::FftLineLayout;

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
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, size);
                            const std::size_t l = at.line;
                            const std::size_t i = at.element;
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
                        last = for_each_index(q, events, layout.lines * butterflies, {last},
                            [=](std::size_t flat)
                            {
                                const Indexing::FftFlatCoord at =
                                    Indexing::fft_flat_coord(flat, butterflies);
                                const Indexing::ButterflyPositions pair =
                                    Indexing::butterfly_number(at.element, span, size);
                                const T wr = twiddles[2 * pair.twiddle];
                                const T wi = sign * twiddles[2 * pair.twiddle + 1];
                                Kernels::Detail::butterfly(data + 2 * layout.at(at.line, pair.upper),
                                                           data + 2 * layout.at(at.line, pair.lower),
                                                           wr, wi);
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
                    const LineLayout work = Indexing::contiguous_lines(layout.lines, size);

                    sycl::event last = for_each_index(q, events, layout.lines * size, {dep},
                        [=](std::size_t flat)
                        {
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, size);
                            const std::size_t l = at.line;
                            const std::size_t k = at.element;
                            T* w = padded + 2 * flat;
                            if (Indexing::is_padding_lane(k, length))
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
                            const std::size_t k = Indexing::fft_flat_coord(flat, size).element;
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
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, length);
                            const std::size_t l = at.line;
                            const std::size_t k = at.element;
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
                tables.spectrum = tables.lines + plan.line_scalars(batch);
                tables.padded = tables.spectrum + plan.spectrum_copy_scalars(batch);
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
                                             Indexing::contiguous_lines(row_lines, columns),
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
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, length);
                            const std::size_t l = at.line;
                            const std::size_t k = at.element;
                            const T* row = input + l * columns;
                            lines[2 * flat] = packed ? row[2 * k] : row[k];
                            lines[2 * flat + 1] = packed ? row[2 * k + 1] : T(0);
                        });
                    last = Detail::run_lines(q, events, tables.row, lines,
                                             Indexing::contiguous_lines(row_lines, length),
                                             tables.padded, FftDirection::Forward, last);

                    const T* roots = tables.half_roots;
                    last = Detail::for_each_index(q, events, row_lines * width, {last},
                        [=](std::size_t flat)
                        {
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, width);
                            const std::size_t l = at.line;
                            const std::size_t k = at.element;
                            const T* line = lines + 2 * l * length;
                            T* out = output + 2 * flat;
                            if (!packed)
                            {
                                out[0] = line[2 * k];
                                out[1] = line[2 * k + 1];
                                return;
                            }
                            const std::size_t half = length;
                            const std::size_t own = k % half;
                            const std::size_t mirror = Indexing::mirror_bin(k, half);
                            Kernels::Detail::unpack_forward_bin(
                                line[2 * own], line[2 * own + 1], line[2 * mirror],
                                -line[2 * mirror + 1], roots[2 * k], roots[2 * k + 1], out[0],
                                out[1]);
                        });
                }

                Detail::run_lines(q, events, tables.column, output,
                                  Indexing::column_layout(batch, plan.rows(), plan.spectrum_columns()), tables.padded,
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
                                             Indexing::column_layout(batch, plan.rows(), plan.spectrum_columns()), tables.padded,
                                             FftDirection::Inverse, last);
                    last = Detail::run_lines(q, events, tables.row, output,
                                             Indexing::contiguous_lines(row_lines, columns),
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
                                                 Indexing::column_layout(batch, plan.rows(), plan.spectrum_columns()), tables.padded,
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
                            const Indexing::FftFlatCoord at = Indexing::fft_flat_coord(flat, length);
                            const std::size_t l = at.line;
                            const std::size_t k = at.element;
                            const T* in = spectrum + 2 * l * width;
                            T* line = lines + 2 * flat;
                            if (!packed)
                            {
                                const Indexing::HermitianSource from =
                                    Indexing::hermitian_source(k, width, columns);
                                line[0] = in[2 * from.bin];
                                line[1] = from.conjugated ? -in[2 * from.bin + 1]
                                                          : in[2 * from.bin + 1];
                                return;
                            }
                            const std::size_t half = length;
                            Kernels::Detail::pack_inverse_bin(
                                in[2 * k], in[2 * k + 1], in[2 * (half - k)],
                                -in[2 * (half - k) + 1], roots[2 * k], roots[2 * k + 1], line[0],
                                line[1]);
                        });
                    last = Detail::run_lines(q, events, tables.row, lines,
                                             Indexing::contiguous_lines(row_lines, length),
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
