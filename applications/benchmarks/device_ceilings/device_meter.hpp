/**************************************************************************/
/* device_meter.hpp                                                       */
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
 * @file device_meter.hpp
 * @brief Declares DeviceMeter, which measures the compute, bandwidth and launch floor of one runtime device.
 * @author Mustafa Garip
 * @see applications/benchmarks/device_ceilings/README.md
 */

#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/api/graph/dispatch_events.hpp>
#include <ceilings_options.hpp>
#include <ceilings_statistics.hpp>
#include <span_observer.hpp>

namespace SushiBLAS
{
    namespace Ceilings
    {
        /** @brief Names the element type a compute ceiling is measured in. */
        enum class ComputeKind : std::uint8_t
        {
            F16, /**< sycl::half. */
            F32, /**< float. */
            F64  /**< double. */
        };

        /** @brief Returns the name a ceilings file gives a compute kind. */
        [[nodiscard]] inline const char* compute_name(ComputeKind kind) noexcept
        {
            switch (kind)
            {
                case ComputeKind::F16:
                    return "f16";
                case ComputeKind::F32:
                    return "f32";
                case ComputeKind::F64:
                    return "f64";
            }
            return "unknown";
        }

        /** @brief Holds one measured rate with its spread and how many runs it is the median of. */
        struct Measurement
        {
            double rate = 0.0;           /**< Operations or bytes per second, from the median span. */
            double spread = 0.0;         /**< (maximum - minimum) / median of the timed spans. */
            std::uint32_t runs = 0;      /**< Timed runs the median is of. */
            bool within_limit = false;   /**< Whether the spread stayed at or below the limit. */
        };

        namespace Detail
        {
            /** @brief Chains eight independent multiply-adds per work-item for @p iterations steps. */
            template<typename T>
            sycl::event fma_chains(sycl::queue& queue, T* sink, std::size_t items,
                                   std::uint64_t iterations, T factor, T term,
                                   const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id)
                    {
                        const T seed = static_cast<T>(id[0] % 5);
                        T a0 = seed * term;
                        T a1 = a0 + term;
                        T a2 = a1 + term;
                        T a3 = a2 + term;
                        T a4 = a3 + term;
                        T a5 = a4 + term;
                        T a6 = a5 + term;
                        T a7 = a6 + term;
                        for (std::uint64_t i = 0; i < iterations; ++i)
                        {
                            a0 = sycl::fma(a0, factor, term);
                            a1 = sycl::fma(a1, factor, term);
                            a2 = sycl::fma(a2, factor, term);
                            a3 = sycl::fma(a3, factor, term);
                            a4 = sycl::fma(a4, factor, term);
                            a5 = sycl::fma(a5, factor, term);
                            a6 = sycl::fma(a6, factor, term);
                            a7 = sycl::fma(a7, factor, term);
                        }
                        sink[id[0]] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                    });
                });
            }

            /** @brief Copies @p count floats from @p source to @p destination in one kernel. */
            inline sycl::event copy_pass(sycl::queue& queue, const float* source, float* destination,
                                         std::size_t count, const std::vector<sycl::event>& deps)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id)
                    {
                        destination[id[0]] = source[id[0]];
                    });
                });
            }
        } // namespace Detail

        /**
         * @brief Measures one device of a runtime through the path its spans come from: a graph run observed on the device clock.
         *
         * Each measurement runs one node on the device, takes the span of the run from a
         * SpanObserver, scales the work until a span lasts at least the minimum, discards the
         * warm-up runs and reports the median of the timed ones with their spread.
         * @see applications/benchmarks/device_ceilings/README.md
         */
        class DeviceMeter final
        {
            public:
                /**
                 * @param runtime Must outlive the meter.
                 * @param options The device and the measurement protocol.
                 * @pre options.device is below the runtime's device count.
                 */
                DeviceMeter(SushiRuntime::Runtime& runtime, const Options& options)
                    : runtime_(runtime), options_(options),
                      queue_(&runtime.advanced().queue(SushiRuntime::DeviceIndex{options.device})),
                      observer_(std::make_shared<SpanObserver>()),
                      handle_(runtime.add_run_observer(observer_))
                {

                }

                DeviceMeter(const DeviceMeter&) = delete;
                DeviceMeter& operator=(const DeviceMeter&) = delete;
                DeviceMeter(DeviceMeter&&) = delete;
                DeviceMeter& operator=(DeviceMeter&&) = delete;
                ~DeviceMeter() = default;

                /** @brief Returns whether the device computes in @p kind. */
                [[nodiscard]] bool supports(ComputeKind kind) const
                {
                    const sycl::device device = queue_->get_device();
                    switch (kind)
                    {
                        case ComputeKind::F16:
                            return device.has(sycl::aspect::fp16);
                        case ComputeKind::F64:
                            return device.has(sycl::aspect::fp64);
                        case ComputeKind::F32:
                            return true;
                    }
                    return false;
                }

                /** @brief Measures the multiply-add rate in @p kind, in floating-point operations per second. */
                [[nodiscard]] Measurement measure_compute(ComputeKind kind)
                {
                    switch (kind)
                    {
                        case ComputeKind::F16:
                            return measure_compute_as<sycl::half>();
                        case ComputeKind::F64:
                            return measure_compute_as<double>();
                        case ComputeKind::F32:
                            break;
                    }
                    return measure_compute_as<float>();
                }

                /** @brief Measures a streaming copy, counting the bytes read plus the bytes written, per second. */
                [[nodiscard]] Measurement measure_bandwidth()
                {
                    const std::size_t count = options_.stream_mib * 1024U * 1024U / sizeof(float);
                    float* source = sycl::malloc_device<float>(count, *queue_);
                    float* destination = sycl::malloc_device<float>(count, *queue_);
                    if (source == nullptr || destination == nullptr)
                    {
                        sycl::free(source, *queue_);
                        sycl::free(destination, *queue_);
                        return Measurement{};
                    }
                    queue_->fill(source, 1.0f, count).wait();

                    const auto run = [&](std::uint64_t passes)
                    {
                        return run_span([=](sycl::queue& queue,
                                            const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            sycl::event last;
                            std::vector<sycl::event> previous = deps;
                            for (std::uint64_t pass = 0; pass < passes; ++pass)
                            {
                                last = Detail::copy_pass(queue, source, destination, count, previous);
                                SushiRuntime::note_dispatch_event(last);
                                previous.assign(1, last);
                            }
                            return last;
                        });
                    };
                    const double bytes_per_pass = 2.0 * static_cast<double>(count) * sizeof(float);
                    const Measurement measured = measure(run, bytes_per_pass);
                    sycl::free(source, *queue_);
                    sycl::free(destination, *queue_);
                    return measured;
                }

                /** @brief Measures the device time of an empty kernel, in nanoseconds: the median of many runs. */
                [[nodiscard]] double measure_launch_floor_ns()
                {
                    const auto run = [&]()
                    {
                        return run_span([](sycl::queue& queue,
                                           const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            return queue.submit([&](sycl::handler& h)
                            {
                                h.depends_on(deps);
                                h.single_task([]() {});
                            });
                        });
                    };
                    for (std::uint32_t i = 0; i < options_.warmup; ++i)
                        static_cast<void>(run());

                    std::vector<double> spans;
                    const std::uint32_t samples = options_.runs * 10U;
                    spans.reserve(samples);
                    for (std::uint32_t i = 0; i < samples; ++i)
                        spans.push_back(static_cast<double>(run()));
                    return summarize(std::move(spans)).median;
                }

            private:
                /**
                 * @brief Measures the multiply-add rate in the element type T.
                 * @tparam T sycl::half, float or double.
                 */
                template<typename T>
                [[nodiscard]] Measurement measure_compute_as()
                {
                    const std::size_t compute_units =
                        runtime_.advanced().devices()[options_.device].compute_units;
                    const std::size_t items = (compute_units == 0 ? 1 : compute_units) * 4096U;
                    T* sink = sycl::malloc_device<T>(items, *queue_);
                    if (sink == nullptr)
                        return Measurement{};

                    const auto run = [&](std::uint64_t iterations)
                    {
                        return run_span([=](sycl::queue& queue,
                                            const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            return Detail::fma_chains<T>(queue, sink, items, iterations,
                                                         static_cast<T>(0.5), static_cast<T>(0.25),
                                                         deps);
                        });
                    };
                    constexpr double K_FLOPS_PER_ITERATION_PER_ITEM = 16.0;
                    const Measurement measured =
                        measure(run, static_cast<double>(items) * K_FLOPS_PER_ITERATION_PER_ITEM);
                    sycl::free(sink, *queue_);
                    return measured;
                }

                /**
                 * @brief Runs one node on the device and returns the span of the run in nanoseconds.
                 * @param body Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
                 */
                template<typename Body>
                [[nodiscard]] std::uint64_t run_span(Body body)
                {
                    SushiRuntime::Graph graph = runtime_.graph();
                    graph.recorder().task(SushiRuntime::Access(), std::move(body),
                                          SushiRuntime::On(SushiRuntime::DeviceIndex{options_.device}));
                    graph.run().value();
                    return observer_->take_span_ns();
                }

                /**
                 * @brief Scales the work of @p run until a span lasts at least the minimum, then
                 *        times the protocol's runs and reports the rate.
                 * @param run Callable `(std::uint64_t scale) -> std::uint64_t` span in nanoseconds.
                 * @param work_per_scale Operations or bytes one unit of scale performs.
                 */
                template<typename Run>
                [[nodiscard]] Measurement measure(Run run, double work_per_scale)
                {
                    const double minimum_ns = options_.min_span_ms * 1.0e6;
                    std::uint64_t scale = 16;
                    for (int round = 0; round < 24; ++round)
                    {
                        const double span = static_cast<double>(run(scale));
                        if (span >= minimum_ns * 1.2)
                            break;
                        const double wanted = span > 0.0 ? minimum_ns * 1.5 / span : 2.0;
                        scale = static_cast<std::uint64_t>(
                            static_cast<double>(scale) * std::max(2.0, wanted));
                    }

                    Measurement best;
                    for (int attempt = 0; attempt < 3; ++attempt)
                    {
                        for (std::uint32_t i = 0; i < options_.warmup; ++i)
                            static_cast<void>(run(scale));

                        std::vector<double> spans;
                        spans.reserve(options_.runs);
                        for (std::uint32_t i = 0; i < options_.runs; ++i)
                            spans.push_back(static_cast<double>(run(scale)));

                        const RunStatistics statistics = summarize(spans);
                        if (statistics.median <= 0.0)
                            continue;
                        Measurement current;
                        current.rate = work_per_scale * static_cast<double>(scale) /
                                       (statistics.median * 1.0e-9);
                        current.spread = statistics.spread;
                        current.runs = options_.runs;
                        current.within_limit = statistics.spread <= options_.spread_limit;
                        if (best.runs == 0 || current.spread < best.spread)
                            best = current;
                        if (current.within_limit)
                            break;
                    }
                    return best;
                }

                /// @brief The runtime whose graphs the meter runs.
                SushiRuntime::Runtime& runtime_;

                /// @brief The protocol and the device.
                Options options_;

                /// @brief The queue of the measured device, which owns the buffers the kernels use.
                sycl::queue* queue_;

                /// @brief Times each run on the device clock.
                std::shared_ptr<SpanObserver> observer_;

                /// @brief Removes the observer when the meter goes.
                SushiRuntime::RunObserverHandle handle_;
        };
    } // namespace Ceilings
} // namespace SushiBLAS
