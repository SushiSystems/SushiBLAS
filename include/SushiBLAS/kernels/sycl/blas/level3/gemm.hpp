/**************************************************************************/
/* gemm.hpp                                                               */
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

// Tiling/blocking algorithm ported from portBLAS's gemm_local.hpp and
// gemm_no_local.hpp (see NOTICE.md) into SushiBLAS's flat template-function
// convention: upstream's three-level-hierarchy Tile<> and expression-tree
// executor collapse here into two fixed-shape kernels selected at launch from
// a Device::Profile. See NOTICE.md's "Deviations" note for what this port
// still drops from upstream and why none of it is a correctness concern.
//
// Why two kernels and not one. On a device with real scratchpad memory,
// staging an A/B tile there and reading it back item_rows + item_cols times
// per K-step is the whole reason a tiled GEMM is fast. On a CPU backend the
// "local memory" is the same cache the global pointer already reads through,
// so the staging copies the data twice and every group_barrier is a real
// rendezvous between OS threads buying nothing. portBLAS ships exactly this
// split; so does this file. The fact table both variants dispatch on lives in
// <SushiBLAS/engine/device_policy.hpp> and knows nothing about GEMM - tile
// sizes are chosen here, because only this file knows what its accumulator
// costs in registers.

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                * @struct OperandStrides
                * @brief How far one row step and one column step move in an operand.
                *
                * The point of carrying these instead of a (layout, transposed) pair is
                * that the four combinations of the two are only ever two strides, and
                * resolving them once on the host removes a branch from the innermost
                * loop of every kernel below without adding a single template
                * instantiation.
                */
                struct OperandStrides
                {
                    /** @brief Elements to advance to move down one row. */
                    int64_t row = 1;

                    /** @brief Elements to advance to move right one column. */
                    int64_t col = 1;
                };

                /**
                * @brief Resolves a (layout, transpose) pair into a row/column stride pair.
                *
                * The full table, verified element for element against
                * `Kernels::Reference::gemm` (which is this operation's specification):
                *
                * | layout       | transposed | row stride | col stride |
                * |--------------|------------|------------|------------|
                * | ROW_MAJOR    | false      | `ld`       | 1          |
                * | ROW_MAJOR    | true       | 1          | `ld`       |
                * | COLUMN_MAJOR | false      | 1          | `ld`       |
                * | COLUMN_MAJOR | true       | `ld`       | 1          |
                *
                * Transposing an operand and flipping its layout are the same operation,
                * so the table is just the XOR of the two flags - which is the sanity
                * check worth remembering: ROW_MAJOR+transposed and
                * COLUMN_MAJOR+untransposed are literally the same addressing, and so
                * are ROW_MAJOR+untransposed and COLUMN_MAJOR+transposed.
                *
                * @param layout The layout every operand of this call is stored in.
                * @param transposed Whether this operand is used transposed.
                * @param ld The operand's leading dimension.
                * @return The strides to index it with as `p[row * s.row + col * s.col]`.
                */
                [[nodiscard]] inline constexpr OperandStrides strides_for(Core::Layout layout,
                                                                          bool transposed,
                                                                          int64_t ld) noexcept
                {
                    const bool column_indexed =
                        transposed != (layout == Core::Layout::COLUMN_MAJOR);
                    return column_indexed ? OperandStrides{1, ld} : OperandStrides{ld, 1};
                }

                /**
                * @brief One operand element, or zero for coordinates outside the matrix.
                *
                * The zero is what lets both kernels below accumulate over a full padded
                * tile instead of clamping the loop to a ragged remainder: a zero
                * contributes nothing to the running sum, so the arithmetic stays exact.
                *
                * @tparam T The element type.
                * @param p Base pointer of the operand.
                * @param row Row coordinate in the logical (possibly transposed) matrix.
                * @param col Column coordinate in the logical matrix.
                * @param rows Logical row count.
                * @param cols Logical column count.
                * @param s The operand's strides, from @ref strides_for.
                * @return The element, or `T(0)` when out of range.
                */
                template<typename T>
                [[nodiscard]] inline T load_operand(const T* p, int64_t row, int64_t col,
                                                    int64_t rows, int64_t cols,
                                                    OperandStrides s) noexcept
                {
                    if (row >= rows || col >= cols) return T(0);
                    return p[row * s.row + col * s.col];
                }

                /**
                * @brief Accumulator byte budget per work-item for the LOCAL_TILED variant.
                *
                * 256 B is what portBLAS's default `Tile<8,8,8,8>` spends on a float
                * accumulator, and it is the number the register block below is derived
                * from rather than a per-type list of specialisations. Deriving it from
                * `sizeof(T)` is the point: the previous fixed 4x4 block penalised
                * `float` - the training path - to accommodate `complex<double>`, a type
                * almost nobody calls.
                */
                inline constexpr std::size_t K_LOCAL_ACC_BYTES = 256;

                /**
                * @brief Accumulator byte budget per work-item for the NO_LOCAL variant.
                *
                * A quarter of the local-tiled budget, because the constraint on a CPU
                * backend is not the same one. The OpenCL CPU runtime vectorises *across*
                * work-items, packing neighbouring items into SIMD lanes, so every scalar
                * in a work-item's accumulator array is really a whole vector register.
                * A 4x4 float block is already 16 of them - the entire AVX2 register file
                * - before the operand registers are counted. Wider spills to stack, and
                * a spilled accumulator is slower than a smaller one that stays resident.
                */
                inline constexpr std::size_t K_NO_LOCAL_ACC_BYTES = 64;

                /**
                * @brief The largest square register block of @p T that fits a byte budget.
                *
                * @tparam T The accumulator element type.
                * @param budget_bytes Bytes of private memory the block may occupy.
                * @param cap Hard upper bound on the block edge, whatever the budget allows.
                * @return A power-of-two edge length in [1, @p cap].
                */
                template<typename T>
                [[nodiscard]] inline constexpr int register_block_dim(std::size_t budget_bytes,
                                                                      int cap) noexcept
                {
                    int dim = 1;
                    while (dim * 2 <= cap &&
                           static_cast<std::size_t>(dim * 2) *
                           static_cast<std::size_t>(dim * 2) * sizeof(T) <= budget_bytes)
                        dim *= 2;
                    return dim;
                }

                /**
                * @struct GemmLocalTile
                * @brief Compile-time geometry of the LOCAL_TILED variant, for element type T.
                *
                * Everything here is `constexpr` because the accumulator is a fixed-size
                * private array and the local tiles are fixed-size local allocations;
                * that is exactly why the two variants are two instantiations selected by
                * a plain runtime `if`, rather than one function threaded with
                * `if constexpr`.
                *
                * The register block comes out as 8x8 for `half` and `float` and 4x4 for
                * `double`, `complex<float>` and `complex<double>` - see
                * @ref K_LOCAL_ACC_BYTES.
                */
                template<typename T>
                struct GemmLocalTile
                {
                    /** @brief Work-items down the row axis of a work-group. */
                    static constexpr int wg_rows = 8;

                    /** @brief Work-items across the column axis of a work-group. */
                    static constexpr int wg_cols = 8;

                    /** @brief Work-items per work-group; 64 is accepted by every device. */
                    static constexpr int wg_items = wg_rows * wg_cols;

                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int item_rows = register_block_dim<T>(K_LOCAL_ACC_BYTES, 8);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int item_cols = item_rows;

                    /** @brief Rows of C covered by one work-group. */
                    static constexpr int block_m = wg_rows * item_rows;

                    /** @brief Columns of C covered by one work-group. */
                    static constexpr int block_n = wg_cols * item_cols;

                    /** @brief K-extent staged into local memory per iteration. */
                    static constexpr int tile_k = 16;

                    /**
                    * @brief Leading dimension of the A tile, padded off a bank multiple.
                    *
                    * The +1 is not cargo cult; the arithmetic is worked out on the read
                    * site in @ref gemm_local_tiled.
                    */
                    static constexpr int tile_a_ld = tile_k + 1;

                    /**
                    * @brief Local memory one work-group needs, in bytes.
                    *
                    * A `local_accessor` larger than the device's `local_mem_size` is a
                    * runtime launch failure, not a compile error, so @ref gemm checks
                    * this against the profile and falls back rather than crashing.
                    *
                    * @return Bytes of work-group local memory required.
                    */
                    [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
                    {
                        return (static_cast<std::size_t>(block_m) * tile_a_ld +
                                static_cast<std::size_t>(tile_k) * block_n) * sizeof(T);
                    }
                };

                /**
                * @struct GemmNoLocalTile
                * @brief Compile-time geometry of the NO_LOCAL variant, for element type T.
                *
                * Only the register block is fixed at compile time. The work-group shape
                * is not: with no local memory and no barrier there is nothing whose size
                * depends on it, so it is taken from `Device::work_group_size` at launch
                * and stays a runtime value.
                *
                * Comes out as 4x4 for `half` and `float` and 2x2 for the 8- and 16-byte
                * types - see @ref K_NO_LOCAL_ACC_BYTES for why smaller is right here.
                */
                template<typename T>
                struct GemmNoLocalTile
                {
                    /** @brief Rows of C accumulated privately by one work-item. */
                    static constexpr int item_rows = register_block_dim<T>(K_NO_LOCAL_ACC_BYTES, 4);

                    /** @brief Columns of C accumulated privately by one work-item. */
                    static constexpr int item_cols = item_rows;
                };

                /**
                * @struct GroupShape
                * @brief A two-dimensional work-group shape, in work-items.
                */
                struct GroupShape
                {
                    /** @brief Work-items down the row axis. */
                    std::size_t rows = 1;

                    /** @brief Work-items across the column axis. */
                    std::size_t cols = 1;
                };

                /**
                * @brief Splits a work-item budget into the squarest power-of-two 2-D shape.
                *
                * Square-ish on purpose: a work-group covering a square-ish block of C
                * reads the fewest operand elements per element of C it produces, which
                * is the whole arithmetic-intensity argument for register blocking
                * applied one level up.
                *
                * @param budget Work-items the device will accept in one group.
                * @return A shape whose product is at most @p budget and never zero.
                */
                [[nodiscard]] inline GroupShape group_shape(std::size_t budget) noexcept
                {
                    std::size_t side = 1;
                    while ((side * 2) * (side * 2) <= budget) side *= 2;

                    // A budget that is not a perfect square leaves room for a second
                    // row of sub-groups; taking it beats leaving half the group idle.
                    std::size_t rows = side;
                    if (rows * 2 * side <= budget) rows *= 2;

                    return GroupShape{rows, side};
                }

                /**
                * @brief Group count covering @p extent in steps of @p block, at least one.
                *
                * The clamp matters for the legal degenerate cases: BLAS permits m or n
                * of zero, and a zero-sized `nd_range` is a launch failure. One group
                * that stores nothing is the cheap, correct answer.
                *
                * @param extent Rows or columns of C to cover.
                * @param block Rows or columns one work-group covers.
                * @return The number of work-groups, never zero.
                */
                [[nodiscard]] inline constexpr int64_t group_count(int64_t extent,
                                                                   int64_t block) noexcept
                {
                    const int64_t count = (extent + block - 1) / block;
                    return count > 0 ? count : int64_t{1};
                }

                /**
                * @brief GEMM for devices with real scratchpad memory: cooperative A/B
                * tiling through local memory, one barrier each side of the compute phase.
                *
                * Callers normally reach this through @ref gemm, which picks between this
                * and @ref gemm_no_local from the device profile and guarantees the local
                * allocation fits. Called directly, those are the caller's obligations.
                *
                * @tparam T The element type.
                * @tparam Epilogue Functor applied per element before the store; see @ref gemm.
                * @param queue The queue to submit to.
                * @param m Rows of C.
                * @param n Columns of C.
                * @param k Shared inner dimension.
                * @param alpha Scale applied to the product.
                * @param a Base pointer of A.
                * @param a_strides A's row/column strides, from @ref strides_for.
                * @param str_a Element stride between consecutive A matrices in the batch.
                * @param b Base pointer of B.
                * @param b_strides B's row/column strides.
                * @param str_b Element stride between consecutive B matrices in the batch.
                * @param beta Scale applied to the incoming C; exactly zero skips reading C.
                * @param c Base pointer of C.
                * @param c_strides C's row/column strides.
                * @param str_c Element stride between consecutive C matrices in the batch.
                * @param batch_size Number of matrices in the batch.
                * @param deps Events this submission must depend on.
                * @param epilogue The per-element epilogue functor.
                * @return The submitted work's event.
                */
                template<typename T, typename Epilogue>
                [[nodiscard]] sycl::event gemm_local_tiled(
                    sycl::queue& queue,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, OperandStrides a_strides, int64_t str_a,
                    const T* b, OperandStrides b_strides, int64_t str_b,
                    T beta, T* c, OperandStrides c_strides, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using Cfg = GemmLocalTile<T>;

                    SB_ASSERT(Device::profile_for(queue).max_work_group_size >=
                                  static_cast<std::size_t>(Cfg::wg_items),
                              "gemm_local_tiled launched on a device that cannot host its work-group");
                    SB_ASSERT(Device::profile_for(queue).local_mem_bytes >= Cfg::local_mem_bytes(),
                              "gemm_local_tiled launched on a device that cannot host its local tile");

                    const int64_t m_groups = group_count(m, Cfg::block_m);
                    const int64_t n_groups = group_count(n, Cfg::block_n);

                    const sycl::range<3> global(static_cast<std::size_t>(batch_size),
                                                static_cast<std::size_t>(m_groups * Cfg::wg_rows),
                                                static_cast<std::size_t>(n_groups * Cfg::wg_cols));
                    const sycl::range<3> local(1, static_cast<std::size_t>(Cfg::wg_rows),
                                                  static_cast<std::size_t>(Cfg::wg_cols));

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        sycl::local_accessor<T, 1> tile_a(
                            sycl::range<1>(static_cast<std::size_t>(Cfg::block_m) * Cfg::tile_a_ld), h);
                        sycl::local_accessor<T, 1> tile_b(
                            sycl::range<1>(static_cast<std::size_t>(Cfg::tile_k) * Cfg::block_n), h);

                        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                        {
                            // 64-bit only where a value can genuinely exceed 2^31: the
                            // global coordinates, the strides, and the batch offsets.
                            // Everything scoped to a tile is 32-bit, because 64-bit
                            // integer arithmetic is markedly slower on a GPU and a tile
                            // index never comes close to needing it.
                            const int64_t batch_id = static_cast<int64_t>(item.get_group(0));
                            const int64_t row0 = static_cast<int64_t>(item.get_group(1)) * Cfg::block_m;
                            const int64_t col0 = static_cast<int64_t>(item.get_group(2)) * Cfg::block_n;
                            const int local_row = static_cast<int>(item.get_local_id(1));
                            const int local_col = static_cast<int>(item.get_local_id(2));
                            const int local_linear = local_row * Cfg::wg_cols + local_col;

                            const T* A = a + batch_id * str_a;
                            const T* B = b + batch_id * str_b;

                            // Only the output is __restrict__-qualified, and that is
                            // deliberate. BLAS forbids C from overlapping A or B, so
                            // promising the compiler the store target is unaliased is
                            // sound. `a` and `b` are deliberately NOT qualified: a caller
                            // computing C = A * A passes one pointer for both, which is
                            // legal, and __restrict__ on both would make it undefined
                            // behaviour. Do not "fix" this by adding it.
                            T* __restrict__ C = c + batch_id * str_c;

                            T acc[Cfg::item_rows][Cfg::item_cols];
                            for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                    acc[ii][jj] = T(0);

                            // No double buffering, deliberately. Alternating between two
                            // local buffers would elide one of the two barriers below,
                            // but the win over a correctly bank-padded single-buffered
                            // kernel is modest, and this project has no GPU to measure it
                            // on - the only reachable device is an OpenCL CPU, which
                            // takes the no-local variant and has no barriers to elide.
                            // Restructuring a barrier sequence that cannot be verified is
                            // the wrong risk to take; an honest omission beats an
                            // unmeasurable optimisation.
                            for (int64_t k0 = 0; k0 < k; k0 += Cfg::tile_k)
                            {
                                // Cooperative stage: the whole group fills block_m x tile_k
                                // of A and tile_k x block_n of B, each work-item taking a
                                // grid-strided share.
                                for (int idx = local_linear;
                                     idx < Cfg::block_m * Cfg::tile_k;
                                     idx += Cfg::wg_items)
                                {
                                    const int r = idx / Cfg::tile_k;
                                    const int kk = idx % Cfg::tile_k;
                                    tile_a[r * Cfg::tile_a_ld + kk] =
                                        load_operand(A, row0 + r, k0 + kk, m, k, a_strides);
                                }
                                for (int idx = local_linear;
                                     idx < Cfg::tile_k * Cfg::block_n;
                                     idx += Cfg::wg_items)
                                {
                                    const int kk = idx / Cfg::block_n;
                                    const int cc = idx % Cfg::block_n;
                                    tile_b[idx] =
                                        load_operand(B, k0 + kk, col0 + cc, k, n, b_strides);
                                }

                                sycl::group_barrier(item.get_group());

                                // Out-of-range elements staged above are zero, so summing
                                // the full tile_k width - rather than clamping to the true
                                // remainder on the last tile - is still exact.
                                for (int kk = 0; kk < Cfg::tile_k; ++kk)
                                {
                                    // Bank-conflict arithmetic for the A read, worked out
                                    // for float (32 banks of 4 bytes) at this file's
                                    // widest configuration: item_rows = item_cols = 8,
                                    // wg_rows = wg_cols = 8, tile_k = 16, tile_a_ld = 17.
                                    //
                                    // A work-group's linear local id is
                                    // local_row * wg_cols + local_col, so a 32-lane
                                    // sub-group spans 32 / wg_cols = 4 distinct local_row
                                    // values. The 8 lanes sharing a local_row read the
                                    // identical address and are served by a broadcast, so
                                    // only those 4 matter.
                                    //
                                    //   tile_a_ld = 16 (unpadded): stride 16 floats,
                                    //   16 mod 32 == 16, banks {0, 16, 0, 16} - 2-way.
                                    //
                                    //   Contiguous ownership (the shape this kernel
                                    //   started from): stride item_rows * tile_k
                                    //   == 8 * 16 == 128, and 128 mod 32 == 0, so all 4
                                    //   land on one bank - 4-way.
                                    //
                                    //   tile_a_ld = 17 (this code): stride 17 floats and
                                    //   gcd(17, 32) == 1, so the 4 lanes take banks
                                    //   {0, 17, 2, 19} - all distinct, conflict-free. It
                                    //   holds on AMD's 64-lane sub-groups too, where the 8
                                    //   distinct local_row give {0,17,2,19,4,21,6,23}.
                                    //
                                    // The B read needs no padding: the strided ownership
                                    // below puts neighbouring local_col on neighbouring
                                    // floats, so a sub-group's 8 distinct local_col cover
                                    // 8 consecutive banks. Contiguous ownership would have
                                    // spaced them item_cols == 8 apart - banks
                                    // {0, 8, 16, 24} twice, a 2-way conflict.
                                    //
                                    // Register-block ownership is therefore a strided
                                    // lattice, not a contiguous block: work-item
                                    // (local_row, local_col) owns tile rows
                                    // local_row + ii*wg_rows and tile columns
                                    // local_col + jj*wg_cols. Neighbouring work-items touch
                                    // neighbouring elements, which is what makes these
                                    // reads conflict-free and the C store below coalesced.
                                    T reg_a[Cfg::item_rows];
                                    T reg_b[Cfg::item_cols];

                                    for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                        reg_a[ii] = tile_a[(local_row + ii * Cfg::wg_rows) *
                                                           Cfg::tile_a_ld + kk];
                                    for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                        reg_b[jj] = tile_b[kk * Cfg::block_n +
                                                           (local_col + jj * Cfg::wg_cols)];

                                    for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                        for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                            acc[ii][jj] += reg_a[ii] * reg_b[jj];
                                }

                                sycl::group_barrier(item.get_group());
                            }

                            for (int ii = 0; ii < Cfg::item_rows; ++ii)
                            {
                                const int64_t global_row = row0 + local_row + ii * Cfg::wg_rows;
                                if (global_row >= m) continue;

                                for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                {
                                    const int64_t global_col = col0 + local_col + jj * Cfg::wg_cols;
                                    if (global_col >= n) continue;

                                    const int64_t c_idx = global_row * c_strides.row +
                                                          global_col * c_strides.col;

                                    // beta == 0 must not read the (possibly uninitialised)
                                    // prior C value, matching standard BLAS semantics.
                                    const T value = (beta == T(0))
                                        ? alpha * acc[ii][jj]
                                        : alpha * acc[ii][jj] + beta * C[c_idx];

                                    C[c_idx] = epilogue(value, global_row, global_col);
                                }
                            }
                        });
                    });
                }

                /**
                * @brief GEMM for devices without usable scratchpad memory: no
                * `local_accessor`, no barrier, operands read straight from global memory.
                *
                * This is the variant a CPU backend wants. Its "local memory" is the same
                * cache the global pointer already reads through, so the tiling the other
                * variant does would copy every element twice and pay a real
                * thread rendezvous per K-step for the privilege. Reuse still happens -
                * an item_rows x item_cols register block reads (item_rows + item_cols)
                * operand elements per (item_rows * item_cols) FMAs - it just happens out
                * of cache instead of out of scratchpad.
                *
                * @tparam T The element type.
                * @tparam Epilogue Functor applied per element before the store; see @ref gemm.
                * @param queue The queue to submit to.
                * @param profile The target device's profile, for the work-group shape.
                * @param m Rows of C.
                * @param n Columns of C.
                * @param k Shared inner dimension.
                * @param alpha Scale applied to the product.
                * @param a Base pointer of A.
                * @param a_strides A's row/column strides, from @ref strides_for.
                * @param str_a Element stride between consecutive A matrices in the batch.
                * @param b Base pointer of B.
                * @param b_strides B's row/column strides.
                * @param str_b Element stride between consecutive B matrices in the batch.
                * @param beta Scale applied to the incoming C; exactly zero skips reading C.
                * @param c Base pointer of C.
                * @param c_strides C's row/column strides.
                * @param str_c Element stride between consecutive C matrices in the batch.
                * @param batch_size Number of matrices in the batch.
                * @param deps Events this submission must depend on.
                * @param epilogue The per-element epilogue functor.
                * @return The submitted work's event.
                */
                template<typename T, typename Epilogue>
                [[nodiscard]] sycl::event gemm_no_local(
                    sycl::queue& queue, const Device::Profile& profile,
                    int64_t m, int64_t n, int64_t k,
                    T alpha, const T* a, OperandStrides a_strides, int64_t str_a,
                    const T* b, OperandStrides b_strides, int64_t str_b,
                    T beta, T* c, OperandStrides c_strides, int64_t str_c,
                    int64_t batch_size, const std::vector<sycl::event>& deps,
                    Epilogue epilogue)
                {
                    using Cfg = GemmNoLocalTile<T>;

                    const GroupShape group = group_shape(Device::work_group_size(profile));
                    const int wg_rows = static_cast<int>(group.rows);
                    const int wg_cols = static_cast<int>(group.cols);

                    const int64_t block_m = static_cast<int64_t>(wg_rows) * Cfg::item_rows;
                    const int64_t block_n = static_cast<int64_t>(wg_cols) * Cfg::item_cols;

                    const int64_t m_groups = group_count(m, block_m);
                    const int64_t n_groups = group_count(n, block_n);

                    const sycl::range<3> global(static_cast<std::size_t>(batch_size),
                                                static_cast<std::size_t>(m_groups) * group.rows,
                                                static_cast<std::size_t>(n_groups) * group.cols);
                    const sycl::range<3> local(1, group.rows, group.cols);

                    return queue.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);

                        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item)
                        {
                            const int64_t batch_id = static_cast<int64_t>(item.get_group(0));

                            // Same strided-lattice ownership as the local-tiled variant:
                            // this work-item's rows are row0 + ii*wg_rows and its columns
                            // col0 + jj*wg_cols. The reason is different here - there is
                            // no local memory to conflict on - but the payoff is the same
                            // and matters more. A CPU backend vectorises across
                            // neighbouring work-items, so neighbouring items reading and
                            // writing neighbouring addresses is what turns the B load and
                            // the C store into contiguous vector accesses instead of a
                            // gather and a scatter.
                            const int64_t row0 =
                                static_cast<int64_t>(item.get_group(1)) * block_m +
                                static_cast<int64_t>(item.get_local_id(1));
                            const int64_t col0 =
                                static_cast<int64_t>(item.get_group(2)) * block_n +
                                static_cast<int64_t>(item.get_local_id(2));

                            const T* A = a + batch_id * str_a;
                            const T* B = b + batch_id * str_b;

                            // Output only - see the note in gemm_local_tiled for why `a`
                            // and `b` must not be __restrict__-qualified.
                            T* __restrict__ C = c + batch_id * str_c;

                            T acc[Cfg::item_rows][Cfg::item_cols];
                            for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                    acc[ii][jj] = T(0);

                            for (int64_t kk = 0; kk < k; ++kk)
                            {
                                T reg_a[Cfg::item_rows];
                                T reg_b[Cfg::item_cols];

                                // load_operand's zero-fill doubles as the tail guard: an
                                // out-of-range row or column contributes nothing, so no
                                // ragged-remainder loop is needed.
                                for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                    reg_a[ii] = load_operand(A, row0 + ii * wg_rows, kk,
                                                             m, k, a_strides);
                                for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                    reg_b[jj] = load_operand(B, kk, col0 + jj * wg_cols,
                                                             k, n, b_strides);

                                for (int ii = 0; ii < Cfg::item_rows; ++ii)
                                    for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                        acc[ii][jj] += reg_a[ii] * reg_b[jj];
                            }

                            for (int ii = 0; ii < Cfg::item_rows; ++ii)
                            {
                                const int64_t global_row = row0 + ii * wg_rows;
                                if (global_row >= m) continue;

                                for (int jj = 0; jj < Cfg::item_cols; ++jj)
                                {
                                    const int64_t global_col = col0 + jj * wg_cols;
                                    if (global_col >= n) continue;

                                    const int64_t c_idx = global_row * c_strides.row +
                                                          global_col * c_strides.col;

                                    // beta == 0 must not read the (possibly uninitialised)
                                    // prior C value, matching standard BLAS semantics.
                                    const T value = (beta == T(0))
                                        ? alpha * acc[ii][jj]
                                        : alpha * acc[ii][jj] + beta * C[c_idx];

                                    C[c_idx] = epilogue(value, global_row, global_col);
                                }
                            }
                        });
                    });
                }
            } // namespace Detail

            /**
            * @brief SYCL GEMM: `C = Epilogue(alpha * op(A) * op(B) + beta * C)`.
            *
            * Resolves the four (layout, transpose) combinations into plain row/column
            * strides on the host - see @ref Detail::strides_for - so the innermost loop
            * carries no layout branch and no extra template instantiation, then picks
            * one of two kernel shapes from the device profile:
            *
            * - **LOCAL_TILED** when the device has genuine scratchpad memory and the
            *   tile fits it. Cooperative A/B staging, bank-padded, barrier-synchronised.
            * - **NO_LOCAL** otherwise, and always on a CPU backend. No `local_accessor`,
            *   no barrier, a smaller register block and the work-group size the profile
            *   asks for.
            *
            * The choice is logged at `SB_LOG_INFO` so it is observable rather than
            * inferred, and a device that declares scratchpad memory but cannot fit the
            * tile or host the work-group falls back with a warning instead of failing
            * the launch.
            *
            * @tparam T The element type.
            * @tparam Epilogue A functor applied to each element of C immediately before
            *         it is stored: `T operator()(T value, int64_t row, int64_t col) const`.
            *         `value` is the fully formed `alpha * op(A) * op(B) + beta * C`
            *         element and `row`/`col` are its coordinates in C. The coordinates
            *         are the point: without them an epilogue can only do elementwise
            *         work, and with them a bias broadcast (`value + bias[col]`) becomes
            *         expressible. This is the library's highest-value fusion hook - it
            *         lets the layer above fold a bias-add and an activation into GEMM
            *         instead of paying two more kernel launches. Defaults to
            *         @ref SushiBLAS::Kernels::GemmIdentityEpilogue, which costs nothing.
            *
            *         Reachable from outside this library through
            *         @ref SushiBLAS::Dispatch::execute_gemm, which is the supported
            *         seam; this entry point is the kernel underneath it.
            *
            *         @warning An **activation** epilogue used in a *forward training*
            *         pass is silently wrong unless it also writes its pre-activation
            *         input to an auxiliary buffer, or is invertible. The backward pass
            *         has to differentiate the activation at its input; if the only thing
            *         that reached memory is the activation's output, that input no
            *         longer exists and the gradient is computed at the wrong point.
            *         Nothing throws, nothing produces NaN - the model simply trains
            *         towards the wrong place. This is exactly why cuBLASLt exposes
            *         `GELU_AUX_BIAS`, which spills the pre-activation, and not merely
            *         `GELU_BIAS`. Inference-only fusion carries no such constraint, and
            *         neither does a bias-only epilogue, which is invertible.
            *
            * @param queue The queue to submit to.
            * @param layout The layout all three operands are stored in.
            * @param transA Whether A is used transposed.
            * @param transB Whether B is used transposed.
            * @param m Rows of C.
            * @param n Columns of C.
            * @param k Shared inner dimension.
            * @param alpha Scale applied to the product.
            * @param a Base pointer of A.
            * @param lda A's leading dimension.
            * @param str_a Element stride between consecutive A matrices in the batch.
            * @param b Base pointer of B.
            * @param ldb B's leading dimension.
            * @param str_b Element stride between consecutive B matrices in the batch.
            * @param beta Scale applied to the incoming C; exactly zero skips reading C.
            * @param c Base pointer of C. Must not overlap A or B.
            * @param ldc C's leading dimension.
            * @param str_c Element stride between consecutive C matrices in the batch.
            * @param batch_size Number of matrices in the batch.
            * @param deps Events this submission must depend on.
            * @param epilogue The per-element epilogue functor.
            * @return The submitted work's event.
            */
            template<typename T, typename Epilogue = GemmIdentityEpilogue>
            [[nodiscard]] sycl::event gemm(sycl::queue& queue, Core::Layout layout,
                            bool transA, bool transB,
                            int64_t m, int64_t n, int64_t k,
                            T alpha, const T* a, int64_t lda, int64_t str_a,
                            const T* b, int64_t ldb, int64_t str_b,
                            T beta, T* c, int64_t ldc, int64_t str_c,
                            int64_t batch_size, const std::vector<sycl::event>& deps,
                            Epilogue epilogue = Epilogue{})
            {
                const Detail::OperandStrides a_strides = Detail::strides_for(layout, transA, lda);
                const Detail::OperandStrides b_strides = Detail::strides_for(layout, transB, ldb);

                // C is never transposed in BLAS, so it only ever sees the layout.
                const Detail::OperandStrides c_strides = Detail::strides_for(layout, false, ldc);

                using LocalCfg = Detail::GemmLocalTile<T>;
                constexpr std::size_t local_bytes = LocalCfg::local_mem_bytes();
                constexpr std::size_t local_items = static_cast<std::size_t>(LocalCfg::wg_items);

                const Device::Profile& profile = Device::profile_for(queue);
                const bool tile_fits = profile.local_mem_bytes >= local_bytes;
                const bool group_fits = profile.max_work_group_size >= local_items;

                // Both variants are named from this one function, so both are
                // instantiated and both reach the device image whichever branch a given
                // machine takes at runtime - there is no dead-stripping hazard to guard
                // against, and no configuration in which one of them stops compiling
                // unnoticed.
                if (profile.has_dedicated_local_memory && tile_fits && group_fits)
                {
                    SB_LOG_INFO("SYCL GEMM [local_tiled]: batch={} {}x{}x{} block={}x{} tile_k={} local={}B",
                                batch_size, m, n, k,
                                LocalCfg::block_m, LocalCfg::block_n, LocalCfg::tile_k, local_bytes);

                    return Detail::gemm_local_tiled<T>(
                        queue, m, n, k,
                        alpha, a, a_strides, str_a,
                        b, b_strides, str_b,
                        beta, c, c_strides, str_c,
                        batch_size, deps, epilogue);
                }

                if (profile.has_dedicated_local_memory)
                    SB_LOG_WARN("SYCL GEMM: local tile wants {} B and {} work-items, device offers "
                                "{} B and {}; falling back to the no-local variant.",
                                local_bytes, local_items,
                                profile.local_mem_bytes, profile.max_work_group_size);

                SB_LOG_INFO("SYCL GEMM [no_local]: batch={} {}x{}x{} item_block={}x{}",
                            batch_size, m, n, k,
                            Detail::GemmNoLocalTile<T>::item_rows,
                            Detail::GemmNoLocalTile<T>::item_cols);

                return Detail::gemm_no_local<T>(
                    queue, profile, m, n, k,
                    alpha, a, a_strides, str_a,
                    b, b_strides, str_b,
                    beta, c, c_strides, str_c,
                    batch_size, deps, epilogue);
            }

        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
