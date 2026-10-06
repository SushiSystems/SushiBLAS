/**************************************************************************/
/* launch_1d.hpp                                                          */
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
 * @file launch_1d.hpp
 * @brief Declares the launch primitives every per-element kernel shares: map, zip_inplace and indexed.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/device_policy.hpp>

namespace SushiBLAS
{
    namespace Launch
    {
        /**
         * @brief Holds the `sycl::vec` width the library considers for element type T; 1 for `std::complex`.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T>
        inline constexpr int K_VECTOR_WIDTH =
            (Core::is_complex_v<T> || sizeof(T) >= Device::K_VECTOR_TRANSACTION_BYTES)
                ? 1
                : static_cast<int>(Device::K_VECTOR_TRANSACTION_BYTES / sizeof(T));

        /**
         * @brief Returns whether p can be the base address of a 128-bit access.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline bool is_vector_aligned(const void* p) noexcept
        {
            return reinterpret_cast<std::uintptr_t>(p) % Device::K_VECTOR_TRANSACTION_BYTES == 0;
        }

        /**
         * @brief Whether every pointer given can be the base address of a 128-bit access.
         *
         * @tparam Ptrs The pointee types, deduced.
         * @param ps The pointers to test.
         * @return True only if all of @p ps satisfy @ref is_vector_aligned.
         */
        template<typename... Ptrs>
        [[nodiscard]] inline bool vector_aligned(const Ptrs*... ps) noexcept
        {
            return (is_vector_aligned(ps) && ...);
        }

        /**
         * @brief Reinterprets a dtype-erased pointer group as a typed one.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T, std::size_t N>
        [[nodiscard]] inline std::array<T*, N> typed_ptrs(const std::array<void*, N>& raw) noexcept
        {
            std::array<T*, N> out{};
            for (std::size_t k = 0; k < N; ++k) out[k] = static_cast<T*>(raw[k]);
            return out;
        }

        /**
         * @struct Shape1D
         * @brief How one per-element kernel walks its elements.
         */
        struct Shape1D
        {
            /** @brief The nd_range to launch and the stride its loop advances by. */
            Device::Grid1D grid{};

            /**
             * @brief How many whole 128-bit accesses the wide pass covers.
             *
             * Zero means "no wide pass": the scalar loop starts at element 0 and
             * covers everything. That is how the scalar and vectorised shapes stay a
             * single kernel with a single set of loop bounds, rather than two kernels
             * or a divergent branch.
             */
            std::size_t vector_steps = 0;

            /**
             * @brief Holds whether the launch is a flat `range<1>(n)` instead of a grid-stride loop.
             * @see include/SushiBLAS/engine/README.md
             */
            bool flat = false;
        };

        /**
         * @brief Turns a device profile and an access width into a launch geometry.
         * @param width Elements per access: 1 for scalar, the `sycl::vec` width for a 128-bit path.
         * @return The geometry; Shape1D::vector_steps is 0 when width is 1.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline Shape1D plan_for(const Device::Profile& p, std::size_t n,
                                              std::size_t width) noexcept
        {
            Shape1D s;
            s.vector_steps = (width > 1) ? n / width : std::size_t{0};
            s.grid = Device::plan_1d(p, n, width);

            // n == 0 stays on the grid-stride path, which is never empty; see include/SushiBLAS/engine/README.md.
            s.flat = p.is_cpu && width == 1 && n > 0;
            return s;
        }

        /**
         * @brief Decides the launch geometry for n elements of type T.
         * @param aligned Whether every operand satisfies is_vector_aligned.
         * @return The geometry; Shape1D::vector_steps is 0 when the wide path is not taken.
         */
        template<typename T>
        [[nodiscard]] inline Shape1D plan(const sycl::queue& q, std::size_t n, bool aligned)
        {
            const Device::Profile& p = Device::profile_for(q);
            constexpr std::size_t width = static_cast<std::size_t>(K_VECTOR_WIDTH<T>);

            // The wide path needs the compile-time and device widths to agree; see include/SushiBLAS/engine/README.md.
            const bool vectorize = (width > 1) && aligned &&
                                   (Device::vector_width(p, sizeof(T)) == width);

            return plan_for(p, n, vectorize ? width : std::size_t{1});
        }

        /**
         * @brief Decides the launch geometry for n indices, with no wide path.
         * @return The geometry, always with Shape1D::vector_steps == 0.
         * @see include/SushiBLAS/engine/README.md
         */
        [[nodiscard]] inline Shape1D plan_scalar(const sycl::queue& q, std::size_t n)
        {
            return plan_for(Device::profile_for(q), n, std::size_t{1});
        }

        namespace Detail
        {
            /**
             * @brief Loads one whole vector from p at wide index v with a direct aligned access.
             * @pre p satisfies is_vector_aligned.
             * @see include/SushiBLAS/engine/README.md
             */
            template<typename T, int W>
            inline sycl::vec<T, W> load_wide(const T* p, std::size_t v)
            {
                using vec_t = sycl::vec<T, W>;
                static_assert(sizeof(vec_t) == sizeof(T) * static_cast<std::size_t>(W),
                              "sycl::vec must be exactly its elements; a padded vec would stride wrongly.");
                static_assert(alignof(vec_t) == sizeof(T) * static_cast<std::size_t>(W),
                              "sycl::vec must be aligned to its own width for this access to be the wide one.");
                return reinterpret_cast<const vec_t*>(p)[v];
            }

            /** @brief One whole-vector store to @p p at wide index @p v. */
            template<typename T, int W>
            inline void store_wide(T* p, std::size_t v, const sycl::vec<T, W>& r)
            {
                using vec_t = sycl::vec<T, W>;
                static_assert(sizeof(vec_t) == sizeof(T) * static_cast<std::size_t>(W),
                              "sycl::vec must be exactly its elements; a padded vec would stride wrongly.");
                static_assert(alignof(vec_t) == sizeof(T) * static_cast<std::size_t>(W),
                              "sycl::vec must be aligned to its own width for this access to be the wide one.");
                reinterpret_cast<vec_t*>(p)[v] = r;
            }

            /** @brief Fills @p regs with one wide load from each pointer in @p src. */
            template<typename T, int W, std::size_t N, typename Ptrs, std::size_t... K>
            inline void load_all(sycl::vec<T, W> (&regs)[N], const Ptrs& src, std::size_t v,
                                 std::index_sequence<K...>)
            {
                ((regs[K] = load_wide<T, W>(src[K], v)), ...);
            }

            /** @brief Writes each register in @p regs back through the matching pointer in @p dst. */
            template<typename T, int W, std::size_t N, typename Ptrs, std::size_t... K>
            inline void store_all(const Ptrs& dst, const sycl::vec<T, W> (&regs)[N], std::size_t v,
                                  std::index_sequence<K...>)
            {
                (store_wide<T, W>(dst[K], v, regs[K]), ...);
            }

            /** @brief Copies `src[K][i]` into `lane[K]` for every K. */
            template<typename Lane, typename Src, typename Idx, std::size_t... K>
            inline void gather(Lane& lane, const Src& src, Idx i, std::index_sequence<K...>)
            {
                ((lane[K] = src[K][i]), ...);
            }

            /** @brief Copies `lane[K]` back into `dst[K][i]` for every K. */
            template<typename Lane, typename Dst, typename Idx, std::size_t... K>
            inline void scatter(const Lane& lane, Dst& dst, Idx i, std::index_sequence<K...>)
            {
                ((dst[K][i] = lane[K]), ...);
            }

            /** @brief Calls `op(src[0][i], ..., src[N-1][i])` and returns what it returns. */
            template<typename Func, typename Src, typename Idx, std::size_t... K>
            inline auto apply_values(Func& op, const Src& src, Idx i, std::index_sequence<K...>)
            {
                return op(src[K][i]...);
            }

            /** @brief Calls `op(lane[0], ..., lane[N-1])`, passing every slot as an lvalue. */
            template<typename Func, typename Lane, std::size_t... K>
            inline void apply_refs(Func& op, Lane& lane, std::index_sequence<K...>)
            {
                op(lane[K]...);
            }
        } // namespace Detail

        /**
         * @brief Launches `out[i] = op(in0[i], ..., inM[i])` over n elements.
         * @param aligned Whether out and every input satisfy is_vector_aligned.
         * @param op Device-copyable, captures by value, holds no host pointer, allocates and throws nothing.
         * @param out May alias an input at the same index only.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename Func, typename T, typename... Ins>
        inline void map(sycl::handler& h, const sycl::queue& q, std::size_t n, bool aligned,
                        Func op, T* out, const Ins*... ins)
        {
            static_assert(sizeof...(Ins) >= 1, "An elementwise map needs at least one input array.");
            static_assert((std::is_same_v<T, Ins> && ...),
                          "Every operand of an elementwise map must share one element type.");

            constexpr int width = K_VECTOR_WIDTH<T>;
            constexpr std::size_t arity = sizeof...(Ins);
            using seq_t = std::make_index_sequence<arity>;

            const std::array<const T*, arity> in{ins...};
            const Shape1D shape = plan<T>(q, n, aligned);
            const std::size_t stride = shape.grid.stride();
            const std::size_t steps = shape.vector_steps;
            const std::size_t tail = steps * static_cast<std::size_t>(width);

            if (shape.flat)
            {
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    out[i] = Detail::apply_values(op, in, i, seq_t{});
                });
                return;
            }

            h.parallel_for(shape.grid.nd_range(), [=](sycl::nd_item<1> it)
            {
                const std::size_t gid = it.get_global_id(0);

                if constexpr (width > 1)
                {
                    for (std::size_t v = gid; v < steps; v += stride)
                    {
                        sycl::vec<T, width> lanes[arity];
                        Detail::load_all<T, width>(lanes, in, v, seq_t{});

                        sycl::vec<T, width> result;
                        for (int c = 0; c < width; ++c)
                            result[c] = Detail::apply_values(op, lanes, c, seq_t{});

                        Detail::store_wide<T, width>(out, v, result);
                    }
                }

                // Covers everything without a wide pass, and only the sub-vector remainder after one; see include/SushiBLAS/engine/README.md.
                for (std::size_t i = tail + gid; i < n; i += stride)
                    out[i] = Detail::apply_values(op, in, i, seq_t{});
            });
        }

        /**
         * @brief Launches `op(t0[i], ..., tN-1[i])` over n elements with every slot writable.
         * @param ptrs The arrays to update, in the order op expects; they must not overlap each other.
         * @param op Callable `(auto&...) -> void` under the same device-kernel constraints as map.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename T, std::size_t N, typename Func>
        inline void zip_inplace(sycl::handler& h, const sycl::queue& q, std::size_t n, bool aligned,
                                const std::array<T*, N>& ptrs, Func op)
        {
            static_assert(N >= 1, "A fused in-place update needs at least one operand.");

            constexpr int width = K_VECTOR_WIDTH<T>;
            using seq_t = std::make_index_sequence<N>;

            const std::array<T*, N> p = ptrs;
            const Shape1D shape = plan<T>(q, n, aligned);
            const std::size_t stride = shape.grid.stride();
            const std::size_t steps = shape.vector_steps;
            const std::size_t tail = steps * static_cast<std::size_t>(width);

            if (shape.flat)
            {
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                {
                    const std::size_t i = idx[0];
                    T lane[N];
                    Detail::gather(lane, p, i, seq_t{});
                    Detail::apply_refs(op, lane, seq_t{});
                    Detail::scatter(lane, p, i, seq_t{});
                });
                return;
            }

            h.parallel_for(shape.grid.nd_range(), [=](sycl::nd_item<1> it)
            {
                const std::size_t gid = it.get_global_id(0);

                if constexpr (width > 1)
                {
                    for (std::size_t v = gid; v < steps; v += stride)
                    {
                        sycl::vec<T, width> regs[N];
                        Detail::load_all<T, width>(regs, p, v, seq_t{});

                        for (int c = 0; c < width; ++c)
                        {
                            T lane[N];
                            Detail::gather(lane, regs, c, seq_t{});
                            Detail::apply_refs(op, lane, seq_t{});
                            Detail::scatter(lane, regs, c, seq_t{});
                        }

                        Detail::store_all<T, width>(p, regs, v, seq_t{});
                    }
                }

                for (std::size_t i = tail + gid; i < n; i += stride)
                {
                    T lane[N];
                    Detail::gather(lane, p, i, seq_t{});
                    Detail::apply_refs(op, lane, seq_t{});
                    Detail::scatter(lane, p, i, seq_t{});
                }
            });
        }

        /**
         * @brief Launches `op(i)` once for each index in `[0, n)`, with no wide path.
         * @param op Callable `(std::size_t i) -> void` under the same device-kernel constraints as map.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename Func>
        inline void indexed(sycl::handler& h, const sycl::queue& q, std::size_t n, Func op)
        {
            const Shape1D shape = plan_scalar(q, n);
            const std::size_t stride = shape.grid.stride();

            if (shape.flat)
            {
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                {
                    op(static_cast<std::size_t>(idx[0]));
                });
                return;
            }

            h.parallel_for(shape.grid.nd_range(), [=](sycl::nd_item<1> it)
            {
                for (std::size_t i = it.get_global_id(0); i < n; i += stride) op(i);
            });
        }

    } // namespace Launch
} // namespace SushiBLAS
