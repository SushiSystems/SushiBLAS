/**************************************************************************/
/* fft.cpp                                                                */
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
 * @file fft.cpp
 * @brief Defines TransformsOps::fft and inverse_fft, which record batched transforms over a plan
 * cooked once at record time.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/signal/transforms.hpp>

#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/signal_kernels/signal.hpp>
#include <SushiBLAS/experimental/dense_operand.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /// @brief The operation id of the forward transform, and its default node name.
        constexpr const char* K_FORWARD = "signal.fft";

        /// @brief The operation id of the inverse transform, and its default node name.
        constexpr const char* K_INVERSE = "signal.inverse_fft";

        /** @brief Holds the plan of whichever scalar the operands carry; the other is null. */
        struct PlanPair
        {
                std::shared_ptr<const Kernels::FftPlan<float>> single; /**< FLOAT32 plan. */
                std::shared_ptr<const Kernels::FftPlan<double>> twin;  /**< FLOAT64 plan. */
        };

        /**
         * @brief Holds whether a task's plan tables are on its device, and the copy that put
         *        them there.
         * @note The scratch lease lives as long as the task, so the tables go up once.
         */
        class TableUpload final
        {
            public:
                /** @brief Holds what a run waits on, and the copies this run submitted. */
                struct Ready
                {
                    std::vector<sycl::event> deps;     /**< The events the transform waits on. */
                    Device::SubmittedEvents submitted; /**< The copies this run made; empty after the first. */
                };

                /**
                 * @brief Returns what a run waits on, copying the tables on the first run.
                 * @param scratch The task's lease; the tables sit at its head.
                 */
                template<typename T>
                Ready ready(sycl::queue& q, const Kernels::FftPlan<T>& plan, std::size_t batch,
                            void* scratch, const std::vector<sycl::event>& deps)
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    if (!uploaded_)
                    {
                        Device::SubmittedEvents copies =
                            Kernels::SYCL::upload_fft_tables(q, plan, batch, scratch, deps);
                        copy_ = copies;
                        uploaded_ = true;
                        return Ready{{copy_}, std::move(copies)};
                    }
                    std::vector<sycl::event> all = deps;
                    all.push_back(copy_);
                    return Ready{std::move(all), Device::SubmittedEvents()};
                }

            private:
                /// @brief Serialises runs of the task that start at once.
                std::mutex mutex_;

                /// @brief Whether the tables have been copied.
                bool uploaded_ = false;

                /// @brief The copy of the tables.
                sycl::event copy_;
        };

        /**
         * @brief Checks one transform's operands and records it.
         * @param direction Forward reads samples and writes a spectrum; Inverse the other way.
         * @throws std::invalid_argument at record time on a mismatched or non-dense operand.
         */
        TaskHandle record_transform(TaskRecorder& recorder, const Tensor& input, Tensor& output,
                                    const FftShape& shape, const OperationOptions& options,
                                    Kernels::FftDirection direction, const char* op)
        {
            Dispatch::require_operation_options(options, op);
            Dispatch::require_operand(shape.rows >= 1 && shape.columns >= 1, op, "shape",
                                      "has {} rows of {} columns; both must be at least 1",
                                      shape.rows, shape.columns);

            const bool forward = direction == Kernels::FftDirection::Forward;
            const bool real = shape.kind == FftKind::RealToComplex;
            const Tensor& samples = forward ? input : output;
            const Tensor& spectrum = forward ? output : input;
            const char* samples_name = forward ? "input" : "output";
            const char* spectrum_name = forward ? "output" : "input";

            Internal::require_complex_dtype(spectrum, op, spectrum_name);
            const Core::DataType scalar = Internal::scalar_dtype_of(spectrum.dtype);
            Dispatch::require_dtype(samples, real ? scalar : spectrum.dtype, op, samples_name);

            const std::size_t spectrum_columns = real ? shape.columns / 2 + 1 : shape.columns;
            Internal::require_dense_elements(samples, shape.batch * shape.rows * shape.columns, op,
                                             samples_name);
            Internal::require_dense_elements(spectrum, shape.batch * shape.rows * spectrum_columns,
                                             op, spectrum_name);
            Dispatch::require_operand(!output.region().overlaps(input.region()), op, "output",
                                      "overlaps 'input' in memory");

            const Kernels::FftPairing pairing = real ? Kernels::FftPairing::RealToComplex
                                                     : Kernels::FftPairing::ComplexToComplex;
            PlanPair plans;
            std::size_t scratch_bytes = 0;
            if (scalar == Core::DataType::Float32)
            {
                plans.single = std::make_shared<const Kernels::FftPlan<float>>(shape.rows,
                                                                               shape.columns,
                                                                               pairing);
                scratch_bytes = Kernels::SYCL::fft_scratch_bytes(*plans.single, shape.batch);
            }
            else
            {
                plans.twin = std::make_shared<const Kernels::FftPlan<double>>(shape.rows,
                                                                              shape.columns,
                                                                              pairing);
                scratch_bytes = Kernels::SYCL::fft_scratch_bytes(*plans.twin, shape.batch);
            }

            OperandSet operands;
            operands.reads(input).writes(output);
            const ScratchBuffer scratch =
                Internal::allocate_blas_scratch(recorder, input, scratch_bytes, operands);

            const std::string label(options.name);
            TaskDescription description;
            description.name = label.c_str();
            description.operation = forward ? "signal.fft"_op : "signal.inverse_fft"_op;
            description.determinism = options.determinism;
            description.shape = TransformShape{static_cast<std::int64_t>(shape.batch),
                                               static_cast<std::int64_t>(shape.rows),
                                               static_cast<std::int64_t>(shape.columns), real,
                                               scalar};

            const auto upload = std::make_shared<TableUpload>();
            const std::size_t batch = shape.batch;
            const void* read_input = input.data_or_null();
            void* write_output = output.data_or_null();

            return Internal::execute_typed_task<Dispatch::Domain::Real>(
                recorder, description, operands, scalar, "Signal operation",
                [=](auto scalar_type, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                    const std::vector<sycl::event>& deps) -> Device::SubmittedEvents
                {
                    using T = decltype(scalar_type);
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                    {
                        const Kernels::FftPlan<T>* plan;
                        if constexpr (std::is_same_v<T, float>)
                            plan = plans.single.get();
                        else
                            plan = plans.twin.get();
                        auto source = static_cast<const T*>(read_input);
                        auto destination = static_cast<T*>(write_output);

                        return dispatcher.run(q, deps,
                            [=]
                            {
                                if (forward)
                                    Kernels::Reference::fft_forward(*plan, source, destination,
                                                                    batch);
                                else
                                    Kernels::Reference::fft_inverse(*plan, source, destination,
                                                                    batch);
                            },
                            [=](sycl::queue& q2,
                                const std::vector<sycl::event>& deps2) -> Device::SubmittedEvents
                            {
                                TableUpload::Ready ready =
                                    upload->ready(q2, *plan, batch, scratch.data(), deps2);
                                const Kernels::SYCL::FftDeviceTables<T> tables =
                                    Kernels::SYCL::fft_device_tables(*plan, batch, scratch.data());
                                Device::SubmittedEvents events = std::move(ready.submitted);
                                events.append(forward
                                    ? Kernels::SYCL::fft_forward(q2, *plan, tables, source,
                                                                 destination, batch, ready.deps)
                                    : Kernels::SYCL::fft_inverse(q2, *plan, tables, source,
                                                                 destination, batch, ready.deps));
                                return events;
                            });
                    }
                    else
                    {
                        // record_transform admits FLOAT32 and FLOAT64 only.
                        SB_THROW_IF(true, "{} does not support dtype HALF.", op);
                        return Device::SubmittedEvents();
                    }
                });
        }
    } // namespace

    TaskHandle TransformsOps::fft(const Tensor& input, Tensor& output, const FftShape& shape)
    {
        return fft(input, output, shape, OperationOptions{K_FORWARD});
    }

    TaskHandle TransformsOps::fft(const Tensor& input, Tensor& output, const FftShape& shape,
                                  const OperationOptions& options)
    {
        return record_transform(recorder_, input, output, shape, options,
                                Kernels::FftDirection::Forward, K_FORWARD);
    }

    TaskHandle TransformsOps::inverse_fft(const Tensor& input, Tensor& output,
                                          const FftShape& shape)
    {
        return inverse_fft(input, output, shape, OperationOptions{K_INVERSE});
    }

    TaskHandle TransformsOps::inverse_fft(const Tensor& input, Tensor& output,
                                          const FftShape& shape, const OperationOptions& options)
    {
        return record_transform(recorder_, input, output, shape, options,
                                Kernels::FftDirection::Inverse, K_INVERSE);
    }
} // namespace SushiBLAS
