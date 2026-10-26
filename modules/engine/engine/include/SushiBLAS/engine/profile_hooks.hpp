/**************************************************************************/
/* profile_hooks.hpp                                                      */
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
 * @file profile_hooks.hpp
 * @brief Declares the zones and counters of the recorder and the scratch shelf, profiler-free.
 * @author Mustafa Garip
 * @see modules/engine/engine/README.md
 */

#pragma once

#include <mutex>
#include <type_traits>

namespace SushiBLAS
{
    namespace Internal
    {
        /** @brief Names which way a scratch lease was served. */
        enum class ScratchLeaseKind
        {
            PoolHit,  /**< An idle block of the size class was reused. */
            PoolMiss, /**< A new block of the size class was allocated. */
            Unpooled  /**< The size is above the largest class; the block is not kept. */
        };

        /**
         * @brief Runs @p body inside the `blas.record` zone, holding @p mutex for its whole run.
         *
         * A wait on the mutex is recorded when it was contended. Defined in `engine_profile.cpp`,
         * so the includers of the recorder do not see a profiler header.
         */
        void run_recording(std::mutex& mutex, void (*body)(void*), void* context);

        /** @brief Runs @p body inside the `blas.scratch.lease` zone. */
        void run_scratch_lease(void (*body)(void*), void* context);

        /** @brief Adds one lease of @p kind to its profiler counter. */
        void count_scratch_lease(ScratchLeaseKind kind) noexcept;

        /**
         * @brief Registers the observer that turns a kernel's variant note into an instant.
         *
         * Idempotent. The engine calls it on construction.
         */
        void install_profile_hooks() noexcept;

        /** @brief Runs the callable @p body in the `blas.record` zone under @p mutex. */
        template<typename Body>
        void in_recording(std::mutex& mutex, Body&& body)
        {
            using Plain = std::remove_reference_t<Body>;
            run_recording(
                mutex, [](void* context) { (*static_cast<Plain*>(context))(); },
                const_cast<std::remove_const_t<Plain>*>(&body));
        }

        /** @brief Runs the callable @p body in the `blas.scratch.lease` zone. */
        template<typename Body>
        void in_scratch_lease(Body&& body)
        {
            using Plain = std::remove_reference_t<Body>;
            run_scratch_lease(
                [](void* context) { (*static_cast<Plain*>(context))(); },
                const_cast<std::remove_const_t<Plain>*>(&body));
        }
    } // namespace Internal
} // namespace SushiBLAS
