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
 * @file convolve_1d.cpp
 * @brief Defines SignalOps::convolve_1d, which records the valid-mode one-dimensional
 * convolution of every row of a signal with one set of taps.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/signal/signal.hpp>
#include <SushiBLAS/signal/convolve_operand_check.hpp>
#include <SushiBLAS/signal_kernels/signal.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Holds the operation id, which is also the node's name when a call states none. */
        constexpr const char* K_OPERATION = "signal.convolve_1d";
    } // namespace

    TaskHandle SignalOps::convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out)
    {
        return convolve_1d(signal, taps, out, OperationOptions{K_OPERATION});
    }

    TaskHandle SignalOps::convolve_1d(const Tensor& signal, const Tensor& taps, Tensor& out,
                                      const OperationOptions& options)
    {
        static constexpr const char* NAME = K_OPERATION;

        Dispatch::require_operation_options(options, NAME);
        const std::string label(options.name);

        const int64_t rows = Internal::check_convolve_operands(
            Dispatch::placed(signal), Dispatch::placed(taps), Dispatch::placed(out), NAME);
        const int64_t signal_length = signal.shape[signal.rank - 1];
        const int64_t tap_count = taps.shape[0];

        const void* read_signal = signal.data_or_null();
        const void* read_taps = taps.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(signal).reads(taps).writes(out);

        TaskDescription description;
        description.name = label.c_str();
        description.operation = "signal.convolve_1d"_op;
        description.determinism = options.determinism;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, signal.dtype, "Signal operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_signal);
                auto tap_values = static_cast<const T*>(read_taps);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(deps,
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
