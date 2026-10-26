/**************************************************************************/
/* ceilings_options.hpp                                                   */
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
 * @file ceilings_options.hpp
 * @brief Declares the options of the device_ceilings program and the parse of its command line.
 * @author Mustafa Garip
 * @see applications/benchmarks/device_ceilings/README.md
 */

#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace SushiBLAS
{
    namespace Ceilings
    {
        /** @brief Holds what the command line of device_ceilings asked for. */
        struct Options
        {
            std::size_t device = 0;          /**< Index of the runtime device to measure. */
            std::string out = ".";           /**< Folder the ceilings file is written into. */
            std::uint32_t runs = 5;          /**< Timed runs per ceiling; the ceiling is their median. */
            std::uint32_t warmup = 2;        /**< Runs made and discarded before the timed ones. */
            double min_span_ms = 10.0;       /**< The shortest device span a timed run may have. */
            double spread_limit = 0.05;      /**< The spread above which a measurement is repeated. */
            std::size_t stream_mib = 256;    /**< Size of each buffer of the bandwidth kernel, in MiB. */
        };

        /** @brief Holds the options parsed from a command line, or why it was refused. */
        struct ParseResult
        {
            bool ok = true;       /**< False when the command line was refused or help was asked for. */
            bool help = false;    /**< True when `--help` was given. */
            std::string message;  /**< Why the line was refused; empty when ok. */
            Options options;      /**< The options, with a default for each one not given. */
        };

        /** @brief Names every option and what it takes, for `--help` and for a refusal. */
        inline constexpr std::string_view K_USAGE =
            "usage: device_ceilings [--device INDEX] [--out FOLDER] [--runs N] [--warmup N]\n"
            "                       [--min-span-ms MS] [--spread-limit FRACTION] [--stream-mib MIB]\n";

        namespace Detail
        {
            /** @brief Reads a whole number of at least @p minimum from @p text; false when it is not one. */
            inline bool whole(const char* text, std::uint64_t minimum, std::uint64_t& out)
            {
                char* end = nullptr;
                const unsigned long long value = std::strtoull(text, &end, 10);
                if (end == text || *end != '\0' || value < minimum || text[0] == '-')
                    return false;
                out = value;
                return true;
            }

            /** @brief Reads a positive real from @p text; false when it is not one. */
            inline bool positive_real(const char* text, double& out)
            {
                char* end = nullptr;
                const double value = std::strtod(text, &end);
                if (end == text || *end != '\0' || !(value > 0.0))
                    return false;
                out = value;
                return true;
            }
        } // namespace Detail

        /**
         * @brief Parses the arguments after the program name.
         * @param argc Count of arguments including the program name.
         * @param argv The arguments; each option takes the next one as its value.
         * @return The options, or ok false with the reason in message.
         */
        [[nodiscard]] inline ParseResult parse_options(int argc, const char* const* argv)
        {
            ParseResult result;
            for (int i = 1; i < argc; ++i)
            {
                const std::string_view name = argv[i];
                if (name == "--help")
                {
                    result.ok = false;
                    result.help = true;
                    return result;
                }
                if (i + 1 >= argc)
                {
                    result.ok = false;
                    result.message = std::string(name) + " needs a value";
                    return result;
                }
                const char* value = argv[++i];
                std::uint64_t whole = 0;
                bool accepted = true;
                if (name == "--device")
                {
                    accepted = Detail::whole(value, 0, whole);
                    result.options.device = static_cast<std::size_t>(whole);
                }
                else if (name == "--out")
                {
                    accepted = value[0] != '\0';
                    result.options.out = value;
                }
                else if (name == "--runs")
                {
                    accepted = Detail::whole(value, 1, whole);
                    result.options.runs = static_cast<std::uint32_t>(whole);
                }
                else if (name == "--warmup")
                {
                    accepted = Detail::whole(value, 0, whole);
                    result.options.warmup = static_cast<std::uint32_t>(whole);
                }
                else if (name == "--min-span-ms")
                {
                    accepted = Detail::positive_real(value, result.options.min_span_ms);
                }
                else if (name == "--spread-limit")
                {
                    accepted = Detail::positive_real(value, result.options.spread_limit);
                }
                else if (name == "--stream-mib")
                {
                    accepted = Detail::whole(value, 1, whole);
                    result.options.stream_mib = static_cast<std::size_t>(whole);
                }
                else
                {
                    result.ok = false;
                    result.message = "unknown option " + std::string(name);
                    return result;
                }

                if (!accepted)
                {
                    result.ok = false;
                    result.message = std::string(name) + " does not accept '" + value + "'";
                    return result;
                }
            }
            return result;
        }
    } // namespace Ceilings
} // namespace SushiBLAS
