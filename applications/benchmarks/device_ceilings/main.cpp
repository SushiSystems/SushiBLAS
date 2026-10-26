/**************************************************************************/
/* main.cpp                                                               */
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
 * @file main.cpp
 * @brief Measures the compute, bandwidth and launch floor of one device and writes its ceilings file.
 * @author Mustafa Garip
 * @see applications/benchmarks/device_ceilings/README.md
 */

#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <exception>
#include <filesystem>
#include <sycl/sycl.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/profiling/ceilings.hpp>
#include <SushiBLAS/profiling/ceilings_file.hpp>
#include <ceilings_options.hpp>
#include <device_meter.hpp>

namespace
{
    using SushiBLAS::Ceilings::ComputeKind;
    using SushiBLAS::Ceilings::Measurement;

    /** @brief Prints one result as `<name> <value> <unit>`. */
    void print_result(const std::string& name, double value, const char* unit)
    {
        std::printf("%s %.6g %s\n", name.c_str(), value, unit);
    }

    /** @brief Prints one measurement and its spread, and warns on stderr when the spread broke the limit. */
    void print_measurement(const std::string& name, const Measurement& measurement,
                           const char* unit, double spread_limit)
    {
        print_result(name, measurement.rate, unit);
        print_result(name + ".spread", measurement.spread, "fraction");
        if (!measurement.within_limit)
            std::fprintf(stderr, "warning: %s spread %.3f is above the limit %.3f; the median is "
                                 "kept, the spread is in the file\n",
                         name.c_str(), measurement.spread, spread_limit);
    }

    /** @brief Returns the build type this program was compiled as. */
    const char* build_type() noexcept
    {
#if defined(NDEBUG)
        return "release";
#else
        return "debug";
#endif
    }

    /** @brief Measures the device the options name, prints the results and writes the file. */
    int run(const SushiBLAS::Ceilings::Options& options)
    {
        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        const auto& devices = runtime.advanced().devices();
        if (options.device >= devices.size())
        {
            std::fprintf(stderr, "--device %zu names no device; the runtime drives %zu\n",
                         options.device, devices.size());
            return 2;
        }

        const SushiRuntime::DeviceDescription& description = devices[options.device];
        sycl::queue& queue = runtime.advanced().queue(SushiRuntime::DeviceIndex{options.device});
        const std::string driver = queue.get_device().get_info<sycl::info::device::driver_version>();

        std::printf("device %s\n", description.name.c_str());
        std::printf("driver %s\n", driver.c_str());
        std::printf("build %s sushiblas %s\n", build_type(), SUSHIBLAS_VERSION_STRING);
        std::printf("protocol warmup %u runs %u min_span_ms %g spread_limit %g\n", options.warmup,
                    options.runs, options.min_span_ms, options.spread_limit);

        SushiBLAS::Ceilings::DeviceMeter meter(runtime, options);
        SushiBLAS::Profiling::DeviceCeilings ceilings;
        ceilings.device_name = description.name;
        ceilings.backend = queue.get_device().get_platform().get_info<sycl::info::platform::name>();
        ceilings.driver = driver;
        ceilings.smoke_only = description.kind == SushiRuntime::DeviceKind::CPU;
        ceilings.method = {options.warmup, options.runs, options.min_span_ms, options.spread_limit};

        for (const ComputeKind kind : {ComputeKind::F32, ComputeKind::F64, ComputeKind::F16})
        {
            const std::string name = std::string("device_ceilings.") +
                                     SushiBLAS::Ceilings::compute_name(kind);
            if (!meter.supports(kind))
            {
                std::printf("%s unsupported\n", name.c_str());
                continue;
            }
            const Measurement measurement = meter.measure_compute(kind);
            if (measurement.runs == 0)
            {
                std::fprintf(stderr, "error: %s could not be measured\n", name.c_str());
                continue;
            }
            print_measurement(name, measurement, "FLOP/s", options.spread_limit);
            ceilings.compute.push_back({SushiBLAS::Ceilings::compute_name(kind), measurement.rate,
                                        measurement.spread, measurement.runs});
        }

        const Measurement bandwidth = meter.measure_bandwidth();
        if (bandwidth.runs == 0)
        {
            std::fprintf(stderr, "error: the bandwidth could not be measured\n");
            return 1;
        }
        print_measurement("device_ceilings.bandwidth", bandwidth, "B/s", options.spread_limit);
        ceilings.bandwidth = {bandwidth.rate, bandwidth.spread, bandwidth.runs};

        ceilings.launch_floor_ns = meter.measure_launch_floor_ns();
        print_result("device_ceilings.launch_floor", ceilings.launch_floor_ns, "ns");

        if (ceilings.compute.empty())
        {
            std::fprintf(stderr, "error: no compute ceiling could be measured; no file written\n");
            return 1;
        }

        const std::filesystem::path path = std::filesystem::path(options.out) /
            SushiBLAS::Profiling::ceilings_file_name(ceilings.device_name);
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file << SushiBLAS::Profiling::write_ceilings(ceilings);
        file.close();
        if (!file)
        {
            std::fprintf(stderr, "error: %s could not be written\n", path.string().c_str());
            return 1;
        }
        std::printf("wrote %s%s\n", path.string().c_str(),
                    ceilings.smoke_only ? " (smoke only: a CPU device)" : "");
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    const SushiBLAS::Ceilings::ParseResult parsed = SushiBLAS::Ceilings::parse_options(argc, argv);
    if (parsed.help)
    {
        std::fputs(SushiBLAS::Ceilings::K_USAGE.data(), stdout);
        return 0;
    }
    if (!parsed.ok)
    {
        std::fprintf(stderr, "error: %s\n%s", parsed.message.c_str(),
                     SushiBLAS::Ceilings::K_USAGE.data());
        return 2;
    }

    try
    {
        return run(parsed.options);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
