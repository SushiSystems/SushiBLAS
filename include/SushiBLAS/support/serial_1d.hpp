/**************************************************************************/
/* serial_1d.hpp                                                          */
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
 * @file serial_1d.hpp
 * @brief Declares the serial host loops that are the reference path of the per-element launch shapes.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace SushiBLAS
{
    namespace Serial
    {
        namespace Detail
        {
            /** @brief Calls `op(src[0][i], ..., src[N-1][i])` and returns what it returns. */
            template<typename Func, typename Src, std::size_t... K>
            inline decltype(auto) call_values(const Func& op, const Src& src, std::size_t i,
                                              std::index_sequence<K...>)
            {
                return op(src[K][i]...);
            }

            /** @brief Calls `op(dst[0][i], ..., dst[N-1][i])`, passing every element as an lvalue. */
            template<typename Func, typename Dst, std::size_t... K>
            inline void call_refs(const Func& op, const Dst& dst, std::size_t i,
                                  std::index_sequence<K...>)
            {
                op(dst[K][i]...);
            }
        } // namespace Detail

        /**
         * @brief Runs `out[i] = op(in0[i], ..., inM[i])` for i from 0 to n - 1 on the calling thread.
         * @param op The functor Launch::map would launch; it is called once per element, in index order.
         * @param out May alias an input at the same index only.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename Func, typename T, typename... Ins>
        inline void map(std::size_t n, const Func& op, T* out, const Ins*... ins)
        {
            static_assert(sizeof...(Ins) >= 1, "An elementwise map needs at least one input array.");
            static_assert((std::is_same_v<T, Ins> && ...),
                          "Every operand of an elementwise map must share one element type.");

            const std::array<const T*, sizeof...(Ins)> in{ins...};
            for (std::size_t i = 0; i < n; ++i)
                out[i] = Detail::call_values(op, in, i, std::make_index_sequence<sizeof...(Ins)>{});
        }

        /**
         * @brief Runs `op(t0[i], ..., tN-1[i])` for i from 0 to n - 1 on the calling thread, every slot writable.
         * @param ptrs The arrays to update, in the order op expects; they must not overlap each other.
         * @param op The functor Launch::zip_inplace would launch.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename T, std::size_t N, typename Func>
        inline void zip_inplace(std::size_t n, const std::array<T*, N>& ptrs, const Func& op)
        {
            static_assert(N >= 1, "A fused in-place update needs at least one operand.");

            for (std::size_t i = 0; i < n; ++i)
                Detail::call_refs(op, ptrs, i, std::make_index_sequence<N>{});
        }

        /**
         * @brief Runs `op(i)` for i from 0 to n - 1 on the calling thread.
         * @param op The functor Launch::indexed would launch.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename Func>
        inline void indexed(std::size_t n, const Func& op)
        {
            for (std::size_t i = 0; i < n; ++i)
                op(i);
        }
    } // namespace Serial
} // namespace SushiBLAS
