/**************************************************************************/
/* ceilings_file.cpp                                                      */
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
 * @file ceilings_file.cpp
 * @brief Defines the reader and the writer of the ceilings file, format version 1.
 * @author Mustafa Garip
 * @see docs/reference/CEILINGS_FORMAT.md
 */

#include <SushiBLAS/profiling/ceilings_file.hpp>

#include <cmath>
#include <cctype>
#include <string>
#include <utility>
#include <SushiProfiler/json/json_field_reader.hpp>
#include <SushiProfiler/json/json_reader.hpp>
#include <SushiProfiler/json/json_value.hpp>
#include <SushiProfiler/json/json_writer.hpp>

namespace SushiBLAS
{
    namespace Profiling
    {
        namespace
        {
            using SushiProfiler::JsonFieldReader;
            using SushiProfiler::JsonValue;

            /** @brief Returns whether a dtype is one the program measures. */
            bool is_measured_dtype(const std::string& dtype)
            {
                return dtype == "f16" || dtype == "f32" || dtype == "f64";
            }

            /** @brief Returns whether a number is finite and above zero. */
            bool is_positive(double value)
            {
                return std::isfinite(value) && value > 0.0;
            }

            /** @brief Reads the `device` object. */
            bool read_device(JsonFieldReader& reader, const JsonValue& root, DeviceCeilings& out)
            {
                const JsonFieldReader::Segment segment(reader, "device");
                const JsonValue& device = JsonFieldReader::member(root, "device");
                if (!reader.object(device, {"name", "backend", "driver"}))
                    return false;
                if (!reader.text(device, "name", out.device_name) ||
                    !reader.text(device, "backend", out.backend) ||
                    !reader.text(device, "driver", out.driver))
                    return false;
                return !out.device_name.empty() || reader.fail("name is empty");
            }

            /** @brief Reads the `method` object. */
            bool read_method(JsonFieldReader& reader, const JsonValue& root, CeilingsMethod& out)
            {
                const JsonFieldReader::Segment segment(reader, "method");
                const JsonValue& method = JsonFieldReader::member(root, "method");
                if (!reader.object(method, {"warmup_runs", "runs", "min_span_ms", "spread_limit"}))
                    return false;
                if (!reader.whole(method, "warmup_runs", out.warmup_runs) ||
                    !reader.whole(method, "runs", out.runs) ||
                    !reader.real(method, "min_span_ms", out.min_span_ms) ||
                    !reader.real(method, "spread_limit", out.spread_limit))
                    return false;
                if (out.runs == 0)
                    return reader.fail("runs is 0; a ceiling is the median of at least one run");
                return true;
            }

            /** @brief Reads one entry of `compute`. */
            bool read_compute_entry(JsonFieldReader& reader, const JsonValue& entry,
                                    DeviceCeilings& out)
            {
                if (!reader.object(entry, {"dtype", "flops_per_s", "spread", "runs"}))
                    return false;
                ComputeCeiling ceiling;
                if (!reader.text(entry, "dtype", ceiling.dtype) ||
                    !reader.real(entry, "flops_per_s", ceiling.flops_per_second) ||
                    !reader.real(entry, "spread", ceiling.spread) ||
                    !reader.whole(entry, "runs", ceiling.runs))
                    return false;
                if (!is_measured_dtype(ceiling.dtype))
                    return reader.fail("dtype is not f16, f32 or f64");
                if (!is_positive(ceiling.flops_per_second))
                    return reader.fail("flops_per_s is not a positive number");
                for (const ComputeCeiling& earlier : out.compute)
                {
                    if (earlier.dtype == ceiling.dtype)
                        return reader.fail("dtype appears twice");
                }
                out.compute.push_back(std::move(ceiling));
                return true;
            }

            /** @brief Reads the `bandwidth` object. */
            bool read_bandwidth(JsonFieldReader& reader, const JsonValue& root,
                                BandwidthCeiling& out)
            {
                const JsonFieldReader::Segment segment(reader, "bandwidth");
                const JsonValue& bandwidth = JsonFieldReader::member(root, "bandwidth");
                if (!reader.object(bandwidth, {"bytes_per_s", "spread", "runs"}))
                    return false;
                if (!reader.real(bandwidth, "bytes_per_s", out.bytes_per_second) ||
                    !reader.real(bandwidth, "spread", out.spread) ||
                    !reader.whole(bandwidth, "runs", out.runs))
                    return false;
                return is_positive(out.bytes_per_second) ||
                       reader.fail("bytes_per_s is not a positive number");
            }

            /** @brief Reads every member of a document whose format and version are known. */
            bool read_members(JsonFieldReader& reader, const JsonValue& root, DeviceCeilings& out)
            {
                if (!reader.object(root, {"format", "version", "device", "smoke_only", "method",
                                          "compute", "bandwidth", "launch_floor_ns"}))
                    return false;
                if (!read_device(reader, root, out) || !reader.flag(root, "smoke_only", out.smoke_only) ||
                    !read_method(reader, root, out.method))
                    return false;
                const auto each_entry = [&](const JsonValue& entry)
                {
                    return read_compute_entry(reader, entry, out);
                };
                if (!reader.each(root, "compute", each_entry))
                    return false;
                if (out.compute.empty())
                {
                    const JsonFieldReader::Segment segment(reader, "compute");
                    return reader.fail("lists no dtype");
                }
                if (!read_bandwidth(reader, root, out.bandwidth) ||
                    !reader.real(root, "launch_floor_ns", out.launch_floor_ns))
                    return false;
                if (!std::isfinite(out.launch_floor_ns) || out.launch_floor_ns < 0.0)
                {
                    const JsonFieldReader::Segment segment(reader, "launch_floor_ns");
                    return reader.fail("is not a number of nanoseconds that is zero or more");
                }
                return true;
            }
        } // namespace

        CeilingsReadResult read_ceilings(std::string_view text)
        {
            CeilingsReadResult result;
            const SushiProfiler::JsonReadResult parsed = SushiProfiler::read_json(text);
            if (!parsed.ok())
            {
                result.status = CeilingsReadStatus::NotJson;
                result.message = "line " + std::to_string(parsed.error.line) + ", column " +
                                 std::to_string(parsed.error.column) + ": " + parsed.error.message;
                return result;
            }

            const JsonValue& root = *parsed.value;
            const JsonValue* format = root.find("format");
            if (format == nullptr || format->as_string() == nullptr ||
                *format->as_string() != K_CEILINGS_FORMAT_NAME)
            {
                result.status = CeilingsReadStatus::WrongFormat;
                result.message = "the document has no `format` member equal to \"" +
                                 std::string(K_CEILINGS_FORMAT_NAME) + "\"";
                return result;
            }

            const JsonValue* version = root.find("version");
            std::uint64_t number = 0;
            if (version == nullptr || !version->to_uint64(number) ||
                number != K_CEILINGS_FORMAT_VERSION)
            {
                result.status = CeilingsReadStatus::UnknownVersion;
                result.message = "the file's `version` is not " +
                                 std::to_string(K_CEILINGS_FORMAT_VERSION);
                return result;
            }

            JsonFieldReader reader("ceilings file");
            if (!read_members(reader, root, result.ceilings))
            {
                result.status = CeilingsReadStatus::Invalid;
                result.message = reader.error();
                result.ceilings = DeviceCeilings{};
            }
            return result;
        }

        std::string write_ceilings(const DeviceCeilings& ceilings)
        {
            SushiProfiler::JsonWriter writer(true);
            writer.begin_object();
            writer.key("format");
            writer.string(K_CEILINGS_FORMAT_NAME);
            writer.key("version");
            writer.unsigned_integer(K_CEILINGS_FORMAT_VERSION);

            writer.key("device");
            writer.begin_object();
            writer.key("name");
            writer.string(ceilings.device_name);
            writer.key("backend");
            writer.string(ceilings.backend);
            writer.key("driver");
            writer.string(ceilings.driver);
            writer.end_object();

            writer.key("smoke_only");
            writer.boolean(ceilings.smoke_only);

            writer.key("method");
            writer.begin_object();
            writer.key("warmup_runs");
            writer.unsigned_integer(ceilings.method.warmup_runs);
            writer.key("runs");
            writer.unsigned_integer(ceilings.method.runs);
            writer.key("min_span_ms");
            writer.real(ceilings.method.min_span_ms);
            writer.key("spread_limit");
            writer.real(ceilings.method.spread_limit);
            writer.end_object();

            writer.key("compute");
            writer.begin_array();
            for (const ComputeCeiling& ceiling : ceilings.compute)
            {
                writer.begin_object();
                writer.key("dtype");
                writer.string(ceiling.dtype);
                writer.key("flops_per_s");
                writer.real(ceiling.flops_per_second);
                writer.key("spread");
                writer.real(ceiling.spread);
                writer.key("runs");
                writer.unsigned_integer(ceiling.runs);
                writer.end_object();
            }
            writer.end_array();

            writer.key("bandwidth");
            writer.begin_object();
            writer.key("bytes_per_s");
            writer.real(ceilings.bandwidth.bytes_per_second);
            writer.key("spread");
            writer.real(ceilings.bandwidth.spread);
            writer.key("runs");
            writer.unsigned_integer(ceilings.bandwidth.runs);
            writer.end_object();

            writer.key("launch_floor_ns");
            writer.real(ceilings.launch_floor_ns);
            writer.end_object();
            return writer.take();
        }

        std::string ceilings_file_name(std::string_view device_name)
        {
            std::string stem;
            bool separated = true;
            for (const char character : device_name)
            {
                const unsigned char byte = static_cast<unsigned char>(character);
                if (std::isalnum(byte) != 0)
                {
                    stem += static_cast<char>(std::tolower(byte));
                    separated = false;
                }
                else if (!separated)
                {
                    stem += '_';
                    separated = true;
                }
            }
            while (!stem.empty() && stem.back() == '_')
                stem.pop_back();
            if (stem.empty())
                stem = "device";
            return stem + ".ceilings.json";
        }
    } // namespace Profiling
} // namespace SushiBLAS
