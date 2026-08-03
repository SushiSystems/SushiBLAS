/**************************************************************************/
/* format.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <string>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string_view>
#include <type_traits>

/**
 * @file format.hpp
 * @brief The `std::format` stand-in the `SB_LOG_*` / `SB_THROW_IF` macros substitute
 * their `{}` placeholders with.
 *
 * `std::format` is C++20 and this library is C++17, so the placeholder substitution the
 * logger has always done has to be spelled without it. This is that spelling, in its own
 * header rather than inside logger.hpp, because "turn a format string and some values
 * into a string" is not logging - it is the one thing logging happens to need, and
 * keeping it separate is what lets it be read, tested and eventually **deleted** on its
 * own. When the whole stack can assume C++20 again, this file goes away and
 * @ref SushiBLAS::Core::Detail::format becomes a call to `std::vformat`; nothing else
 * moves.
 *
 * SushiRuntime carries the same stand-in (`SushiRuntime::Core::Detail::sr_format`), and
 * this is deliberately *not* a call into it: that one lives in SushiRuntime's own
 * `Detail` namespace inside its logger header - internal by this stack's own convention,
 * not a utility it publishes - so reaching into it would couple SushiBLAS's diagnostics
 * to another repo's private implementation detail. The approach is mirrored; the code is
 * not shared.
 *
 * **What it supports, exactly.** Positional `{}` fields filled left to right, `{{` and
 * `}}` for literal braces, and the replacement-field spec subset
 * `{:[0][width][.precision][type]}` where `type` is one of `f`, `e` or `g`. That is a
 * strict subset of `std::format`'s grammar chosen to cover what this codebase actually
 * writes (`{}` everywhere, `{:.4f}` on a handful of log lines), so every existing format
 * string means the same thing it meant under `std::format`. Anything richer - explicit
 * argument indices, fill/align, `+`, `#`, chrono and range specs - is **not** supported
 * and is ignored rather than diagnosed.
 *
 * **What it gives up, and why that is acceptable.** `std::format` with a literal format
 * string is `consteval`-checked: a placeholder count that disagrees with the argument
 * count is a compile error. C++17 has no way to recover that check for a string that
 * arrives through a variadic macro, so a mismatch degrades here to a best-effort message
 * - a surplus placeholder is dropped, a surplus argument is unused - exactly as it does
 * in SushiRuntime today. Substituting a message is never allowed to be the reason a
 * program fails, which is why the failure mode is "a slightly worse diagnostic" rather
 * than a throw.
 */

namespace SushiBLAS
{
    namespace Core
    {
        namespace Detail
        {
            /**
             * @brief The parsed contents of one `{:...}` replacement field.
             *
             * A plain aggregate holding the subset of `std::format`'s spec grammar this
             * header implements; @ref parse_format_spec fills it and
             * @ref apply_format_spec turns it into stream state.
             */
            struct FormatSpec
            {
                /** @brief Minimum field width, or 0 for "as wide as the value needs". */
                int width = 0;

                /** @brief Digits after the decimal point, or -1 when unspecified. */
                int precision = -1;

                /** @brief Whether the field pads with '0' rather than ' '. */
                bool zero_fill = false;

                /** @brief The presentation type character (`f`, `e`, `g`), or '\0'. */
                char type = '\0';
            };

            /**
             * @brief Parses the text between the braces of a replacement field.
             *
             * @param field The characters between `{` and `}` - `""` for a bare `{}`, or
             *        `":.4f"` and the like for a specced field. Anything not matching
             *        `:[0][width][.precision][type]` is ignored, leaving @p spec at its
             *        defaults, which renders the value exactly as a bare `{}` would.
             * @param spec The spec to populate.
             */
            inline void parse_format_spec(std::string_view field, FormatSpec& spec) noexcept
            {
                // No leading ':' means this is not a spec at all (an explicit argument
                // index, say). Unsupported, so it renders as a plain field.
                if (field.empty() || field.front() != ':') return;

                std::size_t i = 1;

                if (i < field.size() && field[i] == '0')
                {
                    spec.zero_fill = true;
                    ++i;
                }

                while (i < field.size() && field[i] >= '0' && field[i] <= '9')
                {
                    spec.width = (spec.width * 10) + (field[i] - '0');
                    ++i;
                }

                if (i < field.size() && field[i] == '.')
                {
                    ++i;
                    spec.precision = 0;
                    while (i < field.size() && field[i] >= '0' && field[i] <= '9')
                    {
                        spec.precision = (spec.precision * 10) + (field[i] - '0');
                        ++i;
                    }
                }

                if (i < field.size()) spec.type = field[i];
            }

            /**
             * @brief Puts @p os into the state @p spec describes, for exactly one insertion.
             *
             * `std::boolalpha` is unconditional rather than part of the spec: `std::format`
             * renders `true` and `false` by name, `operator<<` renders them as 1 and 0 by
             * default, and a log line that used to read "contiguous: true" must not start
             * reading "contiguous: 1" because the formatter changed.
             *
             * @param os The stream about to receive the value.
             * @param spec The parsed field spec.
             */
            inline void apply_format_spec(std::ostream& os, const FormatSpec& spec)
            {
                os << std::boolalpha;

                if (spec.zero_fill) os << std::setfill('0') << std::internal;
                if (spec.width > 0) os << std::setw(spec.width);
                if (spec.precision >= 0) os << std::setprecision(spec.precision);

                switch (spec.type)
                {
                    case 'f': os << std::fixed; break;
                    case 'e': os << std::scientific; break;
                    case 'g': os.unsetf(std::ios_base::floatfield); break;
                    default: break;
                }
            }

            /**
             * @brief One argument to @ref format, type-erased down to "how to print it".
             *
             * A pointer to the value plus a function pointer that streams it, which is the
             * same shape `std::vformat` uses and is here for the same reason: it keeps the
             * substitution loop a single non-template function. The alternative - a
             * recursive variadic template - instantiates a fresh loop for every distinct
             * argument list, and this library has 240-odd `SB_LOG_*` / `SB_THROW_IF` call
             * sites, so that cost lands on every build.
             *
             * @warning Non-owning. A `FormatArg` is only valid for the full-expression that
             *          created it, which is all @ref format needs and all it is used for.
             */
            class FormatArg
            {
                public:
                    /**
                     * @brief Binds to @p value without copying it.
                     * @tparam T The argument's type, deduced.
                     * @param value The value to print later; must outlive this FormatArg.
                     */
                    template<typename T,
                             typename = std::enable_if_t<
                                 !std::is_same<std::decay_t<T>, FormatArg>::value>>
                    explicit FormatArg(const T& value) noexcept
                        : value_(static_cast<const void*>(&value)), write_(&write_as<T>) {}

                    /**
                     * @brief Streams the bound value into @p os under @p spec.
                     *
                     * Saves and restores @p os's formatting state around the insertion, so a
                     * `{:.4f}` field cannot leak `std::fixed` onto the fields that follow it.
                     *
                     * @param os The destination stream.
                     * @param spec The field spec to render under.
                     */
                    void write(std::ostream& os, const FormatSpec& spec) const
                    {
                        const std::ios_base::fmtflags flags = os.flags();
                        const std::streamsize precision = os.precision();
                        const char fill = os.fill();

                        write_(os, value_, spec);

                        os.fill(fill);
                        os.precision(precision);
                        os.flags(flags);
                    }

                private:
                    /**
                     * @brief The streaming step, instantiated once per distinct argument type.
                     * @tparam T The bound value's type.
                     * @param os The destination stream.
                     * @param value The bound value, erased.
                     * @param spec The field spec to render under.
                     */
                    template<typename T>
                    static void write_as(std::ostream& os, const void* value, const FormatSpec& spec)
                    {
                        apply_format_spec(os, spec);
                        os << *static_cast<const T*>(value);
                    }

                    const void* value_;
                    void (*write_)(std::ostream&, const void*, const FormatSpec&);
            };

            /**
             * @brief Substitutes @p args into @p fmt's replacement fields, left to right.
             *
             * The type-erased half of @ref format, so the parsing loop is compiled once for
             * the whole library rather than once per argument list.
             *
             * @param fmt The format string.
             * @param args The arguments, in order; may be null when @p arg_count is 0.
             * @param arg_count How many arguments @p args holds.
             * @return The substituted string.
             */
            inline std::string vformat(std::string_view fmt, const FormatArg* args, std::size_t arg_count)
            {
                std::ostringstream os;
                std::size_t next = 0;

                for (std::size_t i = 0; i < fmt.size(); ++i)
                {
                    const char c = fmt[i];

                    if (c == '{' && i + 1 < fmt.size() && fmt[i + 1] == '{')
                    {
                        os << '{';
                        ++i;
                        continue;
                    }
                    if (c == '}' && i + 1 < fmt.size() && fmt[i + 1] == '}')
                    {
                        os << '}';
                        ++i;
                        continue;
                    }
                    if (c != '{')
                    {
                        os << c;
                        continue;
                    }

                    const std::size_t close = fmt.find('}', i + 1);

                    // An unterminated field is a typo in the format string, not a reason to
                    // lose the message: emit the rest of it verbatim so the author can see
                    // what they wrote.
                    if (close == std::string_view::npos)
                    {
                        os << fmt.substr(i);
                        break;
                    }

                    FormatSpec spec;
                    parse_format_spec(fmt.substr(i + 1, close - (i + 1)), spec);

                    // A field with no argument left to fill it is dropped. See the header
                    // comment: formatting a message never fails the program.
                    if (next < arg_count) args[next++].write(os, spec);

                    i = close;
                }

                return os.str();
            }

            /**
             * @brief Substitutes nothing; the no-argument case of @ref format.
             *
             * Still goes through @ref vformat rather than returning @p fmt directly, so
             * `{{` and `}}` mean the same thing whether or not the call has arguments.
             *
             * @param fmt The format string.
             * @return @p fmt with its brace escapes resolved.
             */
            inline std::string format(std::string_view fmt)
            {
                return vformat(fmt, nullptr, 0);
            }

            /**
             * @brief Substitutes @p first and @p rest into @p fmt's `{}` fields, in order.
             *
             * Spelled with a leading named parameter rather than a bare pack so the
             * argument array below is never zero-sized; the no-argument case is the
             * overload above.
             *
             * @tparam First The first argument's type, deduced.
             * @tparam Rest The remaining argument types, deduced.
             * @param fmt The format string.
             * @param first The first value to substitute.
             * @param rest The remaining values to substitute.
             * @return The substituted string.
             */
            template<typename First, typename... Rest>
            inline std::string format(std::string_view fmt, const First& first, const Rest&... rest)
            {
                const FormatArg args[] = { FormatArg(first), FormatArg(rest)... };
                return vformat(fmt, args, 1 + sizeof...(Rest));
            }

        } // namespace Detail
    } // namespace Core
} // namespace SushiBLAS
