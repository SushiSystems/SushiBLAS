/**************************************************************************/
/* serial_reduce.hpp                                                      */
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
 * @file serial_reduce.hpp
 * @brief Declares the serial host folds that are the reference path of the fixed-order reductions.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/reduce_axis.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Serial
    {
        /**
         * @brief Folds values on the host in a Reduce accumulator, in the type the SYCL fold of A uses.
         * @tparam Accum The accumulator the SYCL path folds in.
         * @tparam A The type the result is stored in.
         * @see include/SushiBLAS/support/README.md
         */
        template<template<typename> class Accum, typename A>
        class Fold final
        {
            public:
                /** @brief The type a value is folded in. */
                using Wide = Reduce::AccumulationOf<A>;

                /** @brief Folds one value. */
                void accumulate(Wide x) noexcept
                {
                    accumulator_.accumulate(x);
                }

                /** @brief Returns the fold of every value so far, or the accumulator's identity. */
                [[nodiscard]] Wide value() const noexcept
                {
                    return accumulator_.value();
                }

            private:
                /// @brief The accumulator, walked by one thread in index order.
                Accum<Wide> accumulator_{};
        };

        /**
         * @brief Folds a sum on the host in the reference accumulator, not in the SYCL path's type.
         * @tparam A The type the result is stored in; a floating-point type.
         * @see include/SushiBLAS/support/README.md
         */
        template<typename A>
        class Fold<Reduce::CompensatedSum, A> final
        {
            public:
                /** @brief The type a term is added in: double. */
                using Wide = Kernels::ReferenceAccumulator<A>;

                /** @brief Adds one term. */
                void accumulate(Wide x) noexcept
                {
                    sum_.add(x);
                }

                /** @brief Returns the sum of every term so far, or zero. */
                [[nodiscard]] Wide value() const noexcept
                {
                    return sum_.value();
                }

            private:
                /// @brief The running sum of a reference kernel.
                Kernels::ReferenceSum<A> sum_{};
        };

        /**
         * @brief Reduces `in[0..n)` to `*out` as `finalize(fold(transform(in[i])), n)`, in index order.
         * @param transform The callable Reduce::reduce would apply to each element.
         * @param finalize Callable `(Wide value, std::size_t n)`, applied once; converted to A on store.
         * @see include/SushiBLAS/support/README.md
         */
        template<template<typename> class Accum, typename T, typename A,
                 typename Transform, typename Finalize>
        inline void reduce(const T* in, A* out, std::size_t n, const Transform& transform,
                           const Finalize& finalize)
        {
            using Wide = typename Fold<Accum, A>::Wide;

            Fold<Accum, A> fold;
            for (std::size_t i = 0; i < n; ++i)
                fold.accumulate(static_cast<Wide>(transform(in[i])));

            *out = static_cast<A>(finalize(fold.value(), n));
        }

        /**
         * @brief Reduces indices `[0..n)` to `*out` as `finalize(fold(read(i)), n)`, in index order.
         * @param read The callable Reduce::reduce_indexed would read element i through.
         * @param finalize Callable `(W value, std::size_t n)`, applied once; converted to Out on store.
         * @see include/SushiBLAS/support/README.md
         */
        template<template<typename> class Accum, typename Read, typename Out, typename Finalize>
        inline void reduce_indexed(std::size_t n, const Read& read, Out* out,
                                   const Finalize& finalize)
        {
            using W = decltype(read(std::size_t{0}));

            Accum<W> accumulator;
            for (std::size_t i = 0; i < n; ++i)
                accumulator.accumulate(read(i));

            *out = static_cast<Out>(finalize(accumulator.value(), n));
        }

        /**
         * @brief Reduces each segment of span to one element of out, segment by segment, in index order.
         * @param transform The callable Reduce::reduce_axis would apply to each element.
         * @param finalize Callable `(Wide value, std::size_t n)`, applied once per segment.
         * @see include/SushiBLAS/support/README.md
         */
        template<template<typename> class Accum, typename T, typename Transform, typename Finalize>
        inline void reduce_axis(const T* in, T* out, const Reduce::AxisSpan& span,
                                const Transform& transform, const Finalize& finalize)
        {
            using Wide = typename Fold<Accum, T>::Wide;

            const std::size_t total = span.total_segments();
            for (std::size_t s = 0; s < total; ++s)
            {
                const std::size_t s2 = s / span.segments;
                const std::size_t s1 = s % span.segments;
                const T* base = in + s2 * span.segment_stride2 + s1 * span.segment_stride;

                Fold<Accum, T> fold;
                for (std::size_t k = 0; k < span.extent; ++k)
                    fold.accumulate(static_cast<Wide>(transform(base[k * span.element_stride])));

                out[s * span.out_stride] = static_cast<T>(finalize(fold.value(), span.extent));
            }
        }
    } // namespace Serial
} // namespace SushiBLAS
