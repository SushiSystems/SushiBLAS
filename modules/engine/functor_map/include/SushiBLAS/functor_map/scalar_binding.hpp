/**************************************************************************/
/* scalar_binding.hpp                                                     */
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
 * @file scalar_binding.hpp
 * @brief Declares the binding that narrows an operation's scalar parameters on the host before its functor reaches a kernel.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <type_traits>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds an element functor with its N scalar parameters narrowed to T; this is
         *        what a kernel captures.
         * @tparam Fn Callable `(elements..., T scalars...)` that captures nothing wider than T.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename T, std::size_t N>
        class NarrowedOp final
        {
            public:
                /** @brief Stores the functor and the narrowed scalars. */
                NarrowedOp(Fn fn, const std::array<T, N>& scalars)
                    : fn_(std::move(fn)), scalars_(scalars)
                {

                }

                /** @brief Calls the functor with the elements, then the scalars in bound order. */
                template<typename... Elements>
                decltype(auto) operator()(Elements&&... elements) const
                {
                    return call(std::make_index_sequence<N>{}, std::forward<Elements>(elements)...);
                }

            private:
                /** @brief Expands the scalars after the elements. */
                template<std::size_t... K, typename... Elements>
                decltype(auto) call(std::index_sequence<K...>, Elements&&... elements) const
                {
                    return fn_(std::forward<Elements>(elements)..., scalars_[K]...);
                }

                /// @brief The element functor.
                Fn fn_;

                /// @brief The scalar parameters in the element type.
                std::array<T, N> scalars_;
        };

        /**
         * @brief Holds an element functor with N scalar parameters not yet narrowed; it stays on
         *        the host.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, std::size_t N>
        class ScalarBound final
        {
            public:
                /** @brief Stores the functor and the scalars as given. */
                ScalarBound(Fn fn, const std::array<ElementScalar, N>& scalars)
                    : fn_(std::move(fn)), scalars_(scalars)
                {

                }

                /**
                 * @brief Returns the functor with every scalar narrowed to T.
                 * @tparam T sycl::half, float, double, int32_t or int64_t.
                 */
                template<typename T>
                [[nodiscard]] NarrowedOp<Fn, T, N> narrow() const
                {
                    return narrow_each<T>(std::make_index_sequence<N>{});
                }

            private:
                /** @brief Narrows scalar K for every K. */
                template<typename T, std::size_t... K>
                [[nodiscard]] NarrowedOp<Fn, T, N> narrow_each(std::index_sequence<K...>) const
                {
                    return NarrowedOp<Fn, T, N>(
                        fn_, std::array<T, N>{scalars_[K].template as<T>()...});
                }

                /// @brief The element functor.
                Fn fn_;

                /// @brief The scalar parameters, each held in every element type.
                std::array<ElementScalar, N> scalars_;
        };

        /**
         * @brief Binds scalar parameters to an element functor for a seam to narrow per element type.
         * @param fn Callable `(elements..., auto scalars...)`; it takes the scalars after the elements.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename... Scalars>
        [[nodiscard]] inline ScalarBound<Fn, sizeof...(Scalars)> bind_scalars(Fn fn,
                                                                             Scalars... scalars)
        {
            static_assert(sizeof...(Scalars) >= 1, "bind_scalars needs at least one scalar.");
            static_assert((std::is_same_v<Scalars, ElementScalar> && ...),
                          "bind_scalars takes its scalars as Dispatch::ElementScalar values.");

            return ScalarBound<Fn, sizeof...(Scalars)>(
                std::move(fn), std::array<ElementScalar, sizeof...(Scalars)>{scalars...});
        }

        namespace Detail
        {
            /** @brief Reports false: an op without `narrow<T>()` has nothing to narrow. */
            template<typename Op, typename T, typename = void>
            struct IsNarrowable : std::false_type
            {

            };

            /** @brief Reports true: an op with `narrow<T>()` narrows itself, as a ScalarBound does. */
            template<typename Op, typename T>
            struct IsNarrowable<Op, T,
                                std::void_t<decltype(std::declval<const Op&>().template narrow<T>())>>
                : std::true_type
            {

            };

            /** @brief Returns a plain functor unchanged; it has nothing to narrow. */
            template<typename T, typename Op>
            [[nodiscard]] inline std::enable_if_t<!IsNarrowable<Op, T>::value, const Op&>
            narrow_for(ElementTag<T>, const Op& op) noexcept
            {
                return op;
            }

            /** @brief Returns an op that narrows itself, such as a bound functor, narrowed to the tag's type. */
            template<typename T, typename Op>
            [[nodiscard]] inline std::enable_if_t<
                IsNarrowable<Op, T>::value,
                decltype(std::declval<const Op&>().template narrow<T>())>
            narrow_for(ElementTag<T>, const Op& op)
            {
                return op.template narrow<T>();
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
