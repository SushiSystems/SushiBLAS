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
 * @file format.hpp
 * @brief Declares the C++17 stand-in for std::format that log and error messages use.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string_view>
#include <type_traits>

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

                /** @brief The alignment character (`<`, `>`, `^`), or '\0' for the default. */
                char align = '\0';

                /** @brief The explicit padding character, or '\0' when none was given. */
                char fill = '\0';

                /** @brief The presentation type character (`f`, `e`, `g`, `d`, `o`, `x`, `X`), or '\0'. */
                char type = '\0';
            };

            /**
             * @brief Reports whether @p c is one of `std::format`'s alignment characters.
             * @param c The character to test.
             * @return True for `<`, `>` and `^`.
             */
            inline bool is_align_char(char c) noexcept
            {
                return c == '<' || c == '>' || c == '^';
            }

            /**
             * @brief Parses the text between the braces of a replacement field into a FormatSpec.
             * @param field Text outside `:[[fill]align][0][width][.precision][type]` leaves spec at its defaults.
             */
            inline void parse_format_spec(std::string_view field, FormatSpec& spec) noexcept
            {
                // A field without a leading ':', such as an argument index, is unsupported and renders plain; see modules/foundation/core/README.md.
                if (field.empty() || field.front() != ':')
                    return;

                std::size_t i = 1;

                // Fill-then-align is tested first, so a '0' fill is not a flag; see modules/foundation/core/README.md.
                if (i + 1 < field.size() && is_align_char(field[i + 1]))
                {
                    spec.fill = field[i];
                    spec.align = field[i + 1];
                    i += 2;
                }
                else if (i < field.size() && is_align_char(field[i]))
                {
                    spec.align = field[i];
                    ++i;
                }

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

                if (i < field.size())
                    spec.type = field[i];
            }

            /**
             * @brief Puts a stream into the state a FormatSpec describes, for one insertion.
             * @see modules/foundation/core/README.md
             */
            inline void apply_format_spec(std::ostream& os, const FormatSpec& spec)
            {
                os << std::boolalpha;

                if (spec.align == '<')
                    os << std::left;
                else if (spec.align == '>')
                    os << std::right;

                // An explicit fill wins; '0' pads only without an alignment; see modules/foundation/core/README.md.
                if (spec.fill != '\0')
                    os << std::setfill(spec.fill);
                else if (spec.zero_fill && spec.align == '\0')
                    os << std::setfill('0') << std::internal;

                if (spec.width > 0)
                    os << std::setw(spec.width);
                if (spec.precision >= 0)
                    os << std::setprecision(spec.precision);

                switch (spec.type)
                {
                    case 'f': os << std::fixed; break;
                    case 'e': os << std::scientific; break;
                    case 'g': os.unsetf(std::ios_base::floatfield); break;
                    case 'd': os << std::dec; break;
                    case 'o': os << std::oct; break;
                    case 'x': os << std::hex << std::nouppercase; break;
                    case 'X': os << std::hex << std::uppercase; break;
                    default: break;
                }
            }

            /**
             * @brief Holds one type-erased argument to format: a pointer to the value and a function that streams it.
             * @warning Non-owning; valid only for the full-expression that created it.
             * @see modules/foundation/core/README.md
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
                     * @brief Streams the bound value under a FormatSpec and restores the stream's formatting state.
                     * @see modules/foundation/core/README.md
                     */
                    void write(std::ostream& os, const FormatSpec& spec) const
                    {
                        if (spec.align == '^' && spec.width > 0)
                        {
                            write_centred(os, spec);
                            return;
                        }

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
                     * @brief Streams the bound value centred in the spec's width, with the odd padding on the right.
                     * @see modules/foundation/core/README.md
                     */
                    void write_centred(std::ostream& os, const FormatSpec& spec) const
                    {
                        FormatSpec unpadded = spec;
                        unpadded.width = 0;
                        unpadded.align = '\0';
                        unpadded.fill = '\0';
                        unpadded.zero_fill = false;

                        std::ostringstream rendered;
                        write_(rendered, value_, unpadded);
                        const std::string text = rendered.str();

                        const std::size_t width = static_cast<std::size_t>(spec.width);
                        if (text.size() >= width)
                        {
                            os << text;
                            return;
                        }

                        const std::size_t padding = width - text.size();
                        const char fill = (spec.fill != '\0') ? spec.fill : ' ';
                        os << std::string(padding / 2, fill) << text
                           << std::string(padding - (padding / 2), fill);
                    }

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
             * @brief Substitutes the arguments into the format string's replacement fields, left to right.
             * @param args May be null when arg_count is 0.
             * @see modules/foundation/core/README.md
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

                    // An unterminated field emits the remaining text verbatim; see modules/foundation/core/README.md.
                    if (close == std::string_view::npos)
                    {
                        os << fmt.substr(i);
                        break;
                    }

                    FormatSpec spec;
                    parse_format_spec(fmt.substr(i + 1, close - (i + 1)), spec);

                    // A field with no argument left to fill it is dropped; formatting never fails the program; see modules/foundation/core/README.md.
                    if (next < arg_count)
                        args[next++].write(os, spec);

                    i = close;
                }

                return os.str();
            }

            /**
             * @brief Resolves the brace escapes of a format string that takes no arguments.
             * @see modules/foundation/core/README.md
             */
            inline std::string format(std::string_view fmt)
            {
                return vformat(fmt, nullptr, 0);
            }

            /**
             * @brief Substitutes first and rest into the `{}` fields of fmt, in order.
             * @see modules/foundation/core/README.md
             */
            template<typename First, typename... Rest>
            inline std::string format(std::string_view fmt, const First& first, const Rest&... rest)
            {
                const FormatArg args[] = { FormatArg(first), FormatArg(rest)... };
                return vformat(fmt, args, 1 + sizeof...(Rest));
            }

        } // namespace Detail

        /**
         * @brief Substitutes the arguments into the replacement fields of fmt, left to right.
         * @see modules/foundation/core/README.md
         */
        template<typename... Args>
        inline std::string format(std::string_view fmt, const Args&... args)
        {
            return Detail::format(fmt, args...);
        }

    } // namespace Core
} // namespace SushiBLAS
