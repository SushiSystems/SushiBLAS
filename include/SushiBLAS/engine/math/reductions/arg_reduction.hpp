/**************************************************************************/
/* arg_reduction.hpp                                                      */
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
 * @file arg_reduction.hpp
 * @brief Declares the dispatcher of an index reduction through the Reduce arg accumulators.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /** @brief Reads element i of an array as a (value, index) candidate. */
        template<typename T>
        struct IndexedRead
        {
            /** @brief The array read. */
            const T* in;

            /** @brief Returns the candidate of element i. */
            [[nodiscard]] Reduce::IndexedValue<T> operator()(std::size_t i) const noexcept
            {
                return Reduce::IndexedValue<T>{in[i], static_cast<std::int64_t>(i)};
            }
        };

        /** @brief Finalizes a winning candidate into its index. */
        struct WinningIndex
        {
            /** @brief Returns the index of best. */
            template<typename P>
            [[nodiscard]] std::int64_t operator()(P best, std::size_t) const noexcept
            {
                return best.index;
            }
        };

        /**
         * @brief Finds the index an arg accumulator elects over a whole tensor.
         * @tparam ArgAccum Reduce::RunningArgMax or Reduce::RunningArgMin.
         */
        template<template<typename> class ArgAccum>
        class ArgReductionDispatch final
        {
            public:
                /** @brief Submits the reduction and returns its event. */
                template<typename T>
                [[nodiscard]] sycl::event operator()(
                    sycl::queue& q, const T* in, int64_t* out, int64_t size,
                    ReductionScratch<Reduce::IndexedValue<T>> scratch,
                    const std::vector<sycl::event>& deps) const
                {
                    return Reduce::reduce_indexed<ArgAccum>(
                        q, static_cast<std::size_t>(size), IndexedRead<T>{in}, out,
                        scratch.data, scratch.capacity, WinningIndex{}, deps);
                }
        };
    } // namespace Dispatch
} // namespace SushiBLAS
