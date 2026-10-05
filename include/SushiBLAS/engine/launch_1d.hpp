/****************************************************************/
/* launch_1d.hpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file launch_1d.hpp
 * @brief The launch geometry every per-element kernel in this library shares.
 *
 * Every elementwise, non-linear and logic op used to launch with a bare
 * `sycl::range<1>(n)`: one work-item per element, no work-group size, no
 * vectorisation, and the runtime free to pick whatever geometry it liked. That
 * is the shape a device is least likely to want. This header replaces it with
 * three primitives - @ref Launch::map, @ref Launch::zip_inplace and
 * @ref Launch::indexed - that own the geometry once, so tuning it is one edit
 * rather than fifty-five.
 *
 * @ref Launch::map and @ref Launch::zip_inplace serve the kernels whose memory
 * access pattern this header can see: contiguous, one element per index, so it
 * may vectorise them. @ref Launch::indexed serves the ones whose pattern is
 * their own business - a Level 1 vector walked with a BLAS `incx`, a GEMV output
 * row that reduces across a whole matrix row, an RNG fill whose value is a
 * function of the index rather than of any input array. Those get the device
 * split and nothing else, which is the part that was missing: before this they
 * were the launch sites still spelling `sycl::range<1>(n)` by hand and therefore
 * still getting the runtime's default geometry on a GPU.
 *
 * All three launch a **grid-stride** `nd_range` planned by
 * @ref SushiBLAS::Device::plan_1d: a grid sized to what the device can hold
 * resident, walked with a `for (i = gid; i < n; i += stride)` loop. Two things
 * follow. The `i < n` test *is* the bounds check, so there is no padded,
 * half-idle tail work-group; and the launch stops growing once the machine is
 * full, which is what keeps a large tensor from paying to schedule a grid the
 * device cannot run at once.
 *
 * Where @ref SushiBLAS::Device::vector_width says a 128-bit access is worth
 * issuing, the loop is preceded by a wide pass that loads and stores
 * `sycl::vec<T, W>` but **calls the caller's functor once per scalar
 * component**. That is deliberate: the functor contract stays scalar, so every
 * existing caller - including a downstream consumer's own fused lambda - keeps
 * working unchanged and never has to know whether it was vectorised.
 *
 * **Determinism.** Both the grid and the scalar/wide split are pure functions
 * of the device, the element type, `n`, and the operands' alignment, none of
 * which vary between two runs of the same binary on the same device. Every
 * element is written exactly once, by exactly one work-item, from the same
 * inputs, in the same arithmetic. So the byte-for-byte reproducibility
 * `sushiruntime_deterministic_fp` promises survives this header intact - which
 * matters, because a training framework above compares checkpoints for
 * bit-equality.
 */

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
         * @brief The `sycl::vec` width this library will consider for element type @p T.
         *
         * The compile-time companion to @ref SushiBLAS::Device::vector_width, which
         * answers the same question at runtime but cannot be used to *name* a type.
         * They must agree before a wide path is taken, and @ref plan checks exactly
         * that - if the device policy ever returns a width this constant does not
         * predict, the launch falls back to scalar rather than issuing an access of
         * the wrong shape.
         *
         * 1 for `std::complex` on purpose. There is no `sycl::vec<std::complex<T>, W>`,
         * and reinterpreting a complex array as 2N reals would hand the functor halves
         * of numbers rather than numbers.
         *
         * @tparam T The element type.
         */
        template<typename T>
        inline constexpr int K_VECTOR_WIDTH =
            (Core::is_complex_v<T> || sizeof(T) >= Device::K_VECTOR_TRANSACTION_BYTES)
                ? 1
                : static_cast<int>(Device::K_VECTOR_TRANSACTION_BYTES / sizeof(T));

        /**
         * @brief Whether @p p can be the base address of a 128-bit access.
         *
         * Asked on the **host**, at task-record time, against the pointer the kernel
         * will actually be handed. A `Tensor` view is only guaranteed to be aligned to
         * its own element type (see `Tensor::storage_offset`), so an odd
         * `storage_offset` into a shared arena produces a perfectly valid tensor whose
         * data pointer is not 16-byte aligned. Assuming otherwise and issuing a wide
         * load anyway is a fault on some devices and silently slow on the rest.
         *
         * @param p The pointer to test.
         * @return True when @p p is a multiple of the wide transaction width.
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
         *
         * The dispatch seam carries operand addresses as `void*` because the element
         * type is a runtime `Core::DataType`; each arm of the dtype switch needs them
         * typed. This is that cast, once, instead of once per operand per arm.
         *
         * @tparam T The element type to view the pointers as.
         * @tparam N How many pointers there are.
         * @param raw The dtype-erased addresses.
         * @return The same addresses, typed.
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
             * @brief Launch as a flat `range<1>(n)` instead of a grid-stride loop.
             *
             * The one place this header takes the device split seriously for its own
             * launches rather than only for GEMM's. A grid-stride kernel exists to let
             * a GPU size its grid to what it can hold resident and walk the rest; a CPU
             * backend has nothing to hold resident, already iterates work-items in a
             * loop of its own, and charges real `nd_range` setup for the privilege.
             *
             * Measured on this machine over 200 chained adds at 1M elements: the
             * grid-stride shape was ~19% slower than the flat one it replaced. Below
             * 64K the two are within run-to-run noise. So the flat shape is not merely
             * simpler on a CPU, it is faster, and `device_policy.hpp` says in its own
             * opening paragraph that these two machines want opposite shapes - it would
             * be strange to measure exactly that and then ship one shape for both.
             *
             * Only ever set when the wide path is off, which on a CPU is always
             * (`Device::vector_width` returns 1 there by design), so the flat kernel is
             * the plain scalar body and nothing is lost by skipping the stride loop.
             */
            bool flat = false;
        };

        /**
         * @brief Turns a device and a chosen access width into a launch geometry.
         *
         * The one place the flat/grid-stride decision and the wide-pass arithmetic are
         * written down. @ref plan and @ref plan_scalar differ only in how they arrive at
         * @p width, so neither of them restates any of this.
         *
         * @param p The target device's profile.
         * @param n How many elements the kernel must cover.
         * @param width Elements per access: 1 for the scalar shape, the `sycl::vec`
         *        width when a 128-bit path was chosen.
         * @return The geometry; @ref Shape1D::vector_steps is 0 when @p width is 1.
         */
        [[nodiscard]] inline Shape1D plan_for(const Device::Profile& p, std::size_t n,
                                              std::size_t width) noexcept
        {
            Shape1D s;
            s.vector_steps = (width > 1) ? n / width : std::size_t{0};
            s.grid = Device::plan_1d(p, n, width);

            // n == 0 stays on the grid-stride path, whose geometry is never empty, so
            // no caller has to think about whether this backend accepts a zero range.
            s.flat = p.is_cpu && width == 1 && n > 0;
            return s;
        }

        /**
         * @brief Decides the launch geometry for @p n elements of type @p T.
         *
         * @tparam T The element type.
         * @param q The queue the kernel will be submitted to.
         * @param n How many elements the kernel must cover.
         * @param aligned Whether every operand satisfies @ref is_vector_aligned.
         * @return The geometry; @ref Shape1D::vector_steps is 0 when the wide path is
         *         not taken.
         */
        template<typename T>
        [[nodiscard]] inline Shape1D plan(const sycl::queue& q, std::size_t n, bool aligned)
        {
            const Device::Profile& p = Device::profile_for(q);
            constexpr std::size_t width = static_cast<std::size_t>(K_VECTOR_WIDTH<T>);

            // Both conditions, not either: the compile-time width names the type, the
            // runtime one carries the policy (it returns 1 on a CPU deliberately - see
            // device_policy.hpp - because the OpenCL CPU backend already vectorises
            // across work-items and a hand-written sycl::vec fights it for the lanes).
            const bool vectorize = (width > 1) && aligned &&
                                   (Device::vector_width(p, sizeof(T)) == width);

            return plan_for(p, n, vectorize ? width : std::size_t{1});
        }

        /**
         * @brief Decides the launch geometry for @p n indices, with no wide path.
         *
         * What @ref indexed plans with. There is no element type in the signature
         * because there is no element the launch knows about: the caller's body decides
         * what it touches and how, so the only question left is the device one - flat on
         * a CPU, grid-stride on a GPU.
         *
         * @param q The queue the kernel will be submitted to.
         * @param n How many indices the kernel must cover.
         * @return The geometry, always with @ref Shape1D::vector_steps == 0.
         */
        [[nodiscard]] inline Shape1D plan_scalar(const sycl::queue& q, std::size_t n)
        {
            return plan_for(Device::profile_for(q), n, std::size_t{1});
        }

        namespace Detail
        {
            /**
             * @brief One whole-vector load from @p p at wide index @p v.
             *
             * A direct aligned access rather than `sycl::vec::load`, which the spec
             * defines as @p W separate scalar reads and which therefore only becomes a
             * single transaction if the optimiser both fuses them and proves the
             * alignment. Here the alignment is already known - checked on the host by
             * @ref is_vector_aligned - so the load can be spelled as what it is. The
             * static_asserts pin the layout that makes it legal: `sycl::vec<T, W>` is a
             * `T[W]` aligned to its own size, so it is layout-compatible with @p W
             * consecutive `T`.
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
         * @brief Launches `out[i] = op(in0[i], ..., inM[i])` over @p n elements.
         *
         * The read-many/write-one shape: every elementwise, non-linear and logic
         * executor in the library is one call to this per dtype arm.
         *
         * @p out may alias any input - the in-place unary executors rely on it. Each
         * index is handled by exactly one work-item, which loads before it stores, so
         * aliasing at the *same* index is well-defined; overlapping at *different*
         * indices is not, and no elementwise op in this library does it.
         *
         * @tparam Func The functor type, deduced.
         * @tparam T The element type, deduced from @p out.
         * @tparam Ins The input element types, deduced; all must equal @p T.
         * @param h The command group handler to record into.
         * @param q The queue being submitted to, used only to reach the device profile.
         * @param n How many elements to cover.
         * @param aligned Whether @p out and every input satisfy @ref is_vector_aligned.
         * @param op Callable `(auto...) -> auto`, invoked once per element on device.
         * @param out The destination array.
         * @param ins The source arrays.
         *
         * @warning @p op is inlined into a SYCL device kernel: it must be
         *          device-copyable, capture by value only, hold no host pointers,
         *          allocate nothing and throw nothing.
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

                // Covers everything when there was no wide pass (tail == 0), and only
                // the sub-vector remainder when there was.
                for (std::size_t i = tail + gid; i < n; i += stride)
                    out[i] = Detail::apply_values(op, in, i, seq_t{});
            });
        }

        /**
         * @brief Launches `op(t0[i], ..., tN-1[i])` over @p n elements, every slot writable.
         *
         * The all-read-write shape, and the one that lets a whole optimiser step
         * collapse into a single kernel: the functor takes lvalue references, so it can
         * read and update any operand it likes in one pass over memory.
         *
         * Each element index is staged into registers, handed to @p op, and written
         * back - so a functor that touches an operand repeatedly pays one load and one
         * store for it, not one per mention, and the scalar and wide paths have
         * identical semantics. The consequence to know about: the @p ptrs arrays must
         * not overlap each other, because a write through one staged slot is not
         * visible to another slot that happens to name the same address.
         *
         * @tparam T The element type.
         * @tparam N How many arrays are being updated.
         * @tparam Func The functor type, deduced.
         * @param h The command group handler to record into.
         * @param q The queue being submitted to, used only to reach the device profile.
         * @param n How many elements to cover.
         * @param aligned Whether every pointer in @p ptrs satisfies @ref is_vector_aligned.
         * @param ptrs The arrays to update, in the order @p op expects them.
         * @param op Callable `(auto&...) -> void`, invoked once per element on device.
         *
         * @warning Same device-kernel constraints as @ref map.
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
         * @brief Launches `op(i)` once for each index in `[0, n)`.
         *
         * The shape for a kernel that owns its own memory access pattern and therefore
         * cannot be spelled as `out[i] = op(in[i]...)`:
         *
         * - a Level 1 BLAS vector, whose `incx`/`incy` mean index `i` is at `i * incx`
         *   rather than at `i`;
         * - a GEMV or SYMV output element, which is a reduction across a whole matrix
         *   row and reads `in_len` values to write one;
         * - a counter-based RNG fill, whose value at `i` is a function of `i` and no
         *   input array at all.
         *
         * What it provides is exactly the device split - a flat `range<1>` on a CPU, a
         * grid-stride `nd_range` on a GPU - taken from the same @ref plan_for that
         * @ref map and @ref zip_inplace use, so there is one geometry in this library
         * rather than one for the kernels that fit `map` and the runtime's default for
         * the rest.
         *
         * What it deliberately does **not** provide is the 128-bit wide path. That pass
         * loads `sycl::vec<T, W>` from a *contiguous* array, which is a claim about the
         * caller's access pattern that this primitive, by construction, cannot make.
         *
         * **Determinism.** Every index is visited exactly once by exactly one work-item,
         * and which work-item that is comes only from the geometry - a pure function of
         * the device and @p n. A body that writes `out[i]` from index `i` alone is
         * therefore as reproducible here as it was under a flat launch. A body that
         * accumulates across work-items would not be, and is not what this is for; see
         * <SushiBLAS/engine/reduce_1d.hpp> for that.
         *
         * @tparam Func The functor type, deduced.
         * @param h The command group handler to record into.
         * @param q The queue being submitted to, used only to reach the device profile.
         * @param n How many indices to cover.
         * @param op Callable `(std::size_t i) -> void`, invoked once per index on device.
         *
         * @warning @p op is inlined into a SYCL device kernel: it must be
         *          device-copyable, capture by value only, hold no host pointers,
         *          allocate nothing and throw nothing.
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
