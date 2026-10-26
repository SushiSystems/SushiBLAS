/**************************************************************************/
/* engine_profile.cpp                                                     */
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
 * @file engine_profile.cpp
 * @brief Defines the profiling zones, counters and the kernel variant observer of the engine.
 * @author Mustafa Garip
 * @see modules/engine/engine/README.md
 */

#include <SushiBLAS/engine/profile_hooks.hpp>

#include <SushiBLAS/launch/kernel_variant.hpp>
#include <SushiBLAS/profile/profile.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        void run_recording(std::mutex& mutex, void (*body)(void*), void* context)
        {
            SB_ZONE_FINE("blas.record");

            // The count before, the recording and the sequence number are one step.
            SB_LOCK_WAIT("blas.recording_lock", mutex);
            body(context);
        }

        void run_scratch_lease(void (*body)(void*), void* context)
        {
            SB_ZONE_FINE("blas.scratch.lease");
            body(context);
        }

        void count_scratch_lease(ScratchLeaseKind kind) noexcept
        {
            switch (kind)
            {
                case ScratchLeaseKind::PoolHit:
                    SB_COUNTER("blas.scratch.pool_hit", "blocks", Add, 1.0);
                    return;
                case ScratchLeaseKind::PoolMiss:
                    SB_COUNTER("blas.scratch.pool_miss", "blocks", Add, 1.0);
                    return;
                case ScratchLeaseKind::Unpooled:
                    SB_COUNTER("blas.scratch.unpooled", "blocks", Add, 1.0);
                    return;
            }
        }

        namespace
        {
            /** @brief Writes the instant that names the variant a kernel launched. */
            void note_variant(Device::KernelVariant variant) noexcept
            {
                switch (variant)
                {
                    case Device::KernelVariant::GemmLocalTiled:
                        SB_INSTANT("blas.gemm.local_tiled");
                        return;
                    case Device::KernelVariant::GemmJointMatrix:
                        SB_INSTANT("blas.gemm.joint_matrix");
                        return;
                    case Device::KernelVariant::GemmDepthSplit:
                        SB_INSTANT("blas.gemm.depth_split");
                        return;
                    case Device::KernelVariant::GemmNoLocal:
                        SB_INSTANT("blas.gemm.no_local");
                        return;
                    case Device::KernelVariant::GemmPacked:
                        SB_INSTANT("blas.gemm.packed");
                        return;
                }
            }
        } // namespace

        void install_profile_hooks() noexcept
        {
            Device::set_kernel_variant_observer(&note_variant);
        }
    } // namespace Internal
} // namespace SushiBLAS
