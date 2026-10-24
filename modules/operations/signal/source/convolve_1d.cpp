/**************************************************************************/
/* convolve_1d.cpp                                                        */
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
 * @file convolve_1d.cpp
 * @brief Defines SignalOps::convolve_1d, which records the valid-mode one-dimensional
 * convolution of every row of a signal with one set of taps.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/signal/signal.hpp>
#include <SushiBLAS/signal_kernels/signal.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /**
         * @brief Throws unless a tensor is dense and its last dimension runs along memory.
         * @throws std::invalid_argument on a view, a slice or a column-major tensor of rank 2+.
         * @see modules/operations/signal/README.md
         */
        void require_dense_rows(const Tensor& t, const char* name, const char* operand)
        {
            Dispatch::require_contiguous(t, name, operand);
            Dispatch::require_operand(t.rank < 2 || t.layout == Core::Layout::RowMajor, name,
                                      operand, "is column-major; its rows must be contiguous");
        }
    } // namespace

    TaskHandle SignalOps::convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out)
    {
        static constexpr const char* NAME = "signal.convolve_1d";

        Dispatch::require_dtype(taps, signal.dtype, NAME, "taps");
        Dispatch::require_dtype(out, signal.dtype, NAME, "out");

        Dispatch::require_rank(taps, 1, NAME, "taps");
        Dispatch::require_operand(signal.rank >= 1, NAME, "signal",
                                  "has rank 0; a row needs one dimension");
        Dispatch::require_rank(out, signal.rank, NAME, "out");

        const int64_t signal_length = signal.shape[signal.rank - 1];
        const int64_t tap_count = taps.shape[0];
        Dispatch::require_operand(tap_count >= 1, NAME, "taps",
                                  "holds {} taps; at least 1 is required", tap_count);
        Dispatch::require_operand(signal_length >= tap_count, NAME, "signal",
                                  "has {} samples per row, fewer than the {} taps", signal_length,
                                  tap_count);

        int64_t rows = 1;
        for (int32_t i = 0; i + 1 < signal.rank; ++i)
        {
            Dispatch::require_extent(out.shape[i], signal.shape[i], "rows in a leading dimension",
                                     NAME, "out");
            rows *= signal.shape[i];
        }
        Dispatch::require_extent(out.shape[out.rank - 1], signal_length - tap_count + 1,
                                 "samples per row", NAME, "out");

        Dispatch::require_storage(signal, NAME, "signal");
        Dispatch::require_storage(taps, NAME, "taps");
        Dispatch::require_storage(out, NAME, "out");
        require_dense_rows(signal, NAME, "signal");
        require_dense_rows(taps, NAME, "taps");
        require_dense_rows(out, NAME, "out");
        Dispatch::require_device_dtype(signal, NAME, "signal");
        Dispatch::require_device_dtype(taps, NAME, "taps");
        Dispatch::require_device_dtype(out, NAME, "out");

        // A gather reads its inputs while it writes, so the output may share no byte with them.
        Dispatch::require_operand(!out.region().overlaps(signal.region()), NAME, "out",
                                  "overlaps 'signal' in memory");
        Dispatch::require_operand(!out.region().overlaps(taps.region()), NAME, "out",
                                  "overlaps 'taps' in memory");

        const void* read_signal = signal.data_or_null();
        const void* read_taps = taps.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(signal).reads(taps).writes(out);

        TaskDescription description;
        description.name = NAME;
        description.operation = "signal.convolve_1d"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, signal.dtype, "Signal operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_signal);
                auto tap_values = static_cast<const T*>(read_taps);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::convolve_1d<T>(rows, signal_length, tap_count, source,
                                                           tap_values, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::convolve_1d<T>(q2, rows, signal_length, tap_count,
                                                             source, tap_values, destination,
                                                             deps2);
                    });
            });
    }
} // namespace SushiBLAS
