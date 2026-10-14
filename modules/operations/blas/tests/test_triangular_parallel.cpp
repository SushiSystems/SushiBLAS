/**************************************************************************/
/* test_triangular_parallel.cpp                                           */
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
 * @file test_triangular_parallel.cpp
 * @brief Tests the blocked TRSV, TBSV, TPSV and TRSM kernels and the parallel TRMV kernel
 *        against exact integer systems, a double-precision truth and each other.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tpsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/trsm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include "test_utils.hpp"
#include "blas_truth.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the seed every case of this file draws its operands from.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Holds the order of the small systems: ten blocks of 7 with a short last one.
            constexpr int64_t K_ORDER = 67;

            /// @brief Holds the right-hand-side vectors of the small systems.
            constexpr int64_t K_VECTORS = 5;

            /// @brief Holds the block widths compared with one block of the whole order.
            constexpr int64_t K_BLOCK_WIDTHS[] = {1, 2, 7, 64, 1000};

            /// @brief Holds the block widths the exact systems are solved at.
            constexpr int64_t K_EXACT_WIDTHS[] = {1, 7, K_ORDER};

            /// @brief Holds the block width of the batched and the gated solve: 19 launches.
            constexpr int64_t K_NARROW_WIDTH = 7;

            /// @brief Holds the matrices of one batched solve.
            constexpr int64_t K_BATCH = 3;

            /// @brief Holds the order at which a device of several compute units blocks a solve.
            constexpr int64_t K_LARGE_ORDER = 1100;

            /// @brief Holds the right-hand-side vectors of the large exact TRSM.
            constexpr int64_t K_LARGE_VECTORS = 3;

            /// @brief Holds the order of the systems whose residual is bounded.
            constexpr int64_t K_RESIDUAL_ORDER = 300;

            /// @brief Holds the right-hand-side vectors of the systems whose residual is bounded.
            constexpr int64_t K_RESIDUAL_VECTORS = 4;

            /// @brief Holds the extent of the empty systems' other axis.
            constexpr int64_t K_EMPTY_EXTENT = 3;

            /// @brief Holds the orders of the exact TRMV cases.
            constexpr int64_t K_PRODUCT_ORDERS[] = {1, 2, 70};

            /// @brief Holds the increments of x in the exact TRMV cases.
            constexpr int64_t K_INCREMENTS[] = {1, 3};

            /// @brief Holds the order of the TRMV measured against the truth.
            constexpr int64_t K_PRODUCT_BOUND_ORDER = 1500;

            /// @brief Holds the order of the TRMV whose snapshot fits a pooled block.
            constexpr int64_t K_POOLED_ORDER = 64;

            /// @brief Holds the work-group width the TRMV kernel is called with directly.
            constexpr std::size_t K_GROUP_WIDTH = 8;

            /// @brief Holds the two row shapes of the TRMV kernel: per work-item and per group.
            constexpr std::size_t K_GROUP_WIDTHS[] = {0, K_GROUP_WIDTH};

            /// @brief Holds the scale of the right-hand side in the rounded cases.
            constexpr double K_ROUNDED_ALPHA = 1.5;

            /// @brief Holds the scale of the right-hand side in the exact cases.
            constexpr double K_EXACT_ALPHA = 2.0;

            /**
             * @brief Holds the value written between the elements of a strided x and into B
             *        where a solve must write nothing.
             */
            constexpr double K_SENTINEL = 77.0;

            /// @brief Holds the fractions a draw can take on each side of zero: 2^20.
            constexpr double K_FRACTION_STEPS = 1048576.0;

            /// @brief Holds the roundings a row product makes whatever its length.
            constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Holds the roundings a substitution step makes whatever its length.
            constexpr std::size_t K_STEP_ROUNDINGS = 6;

            /// @brief Holds the roundings counted for a compensated sum of any length.
            constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

            /// @brief Holds the order of the HALF system of test_blas_half_accumulation.cpp.
            constexpr int64_t K_HALF_ORDER = 64;

            /// @brief Holds the right-hand-side vectors of the HALF system.
            constexpr int64_t K_HALF_VECTORS = 3;

            /// @brief Holds the block widths the HALF system is solved at.
            constexpr int64_t K_HALF_WIDTHS[] = {8, K_HALF_ORDER};

            /// @brief Holds the off-diagonals of the small band systems: none, a few, every one.
            constexpr int64_t K_BAND_WIDTHS[] = {0, 3, K_ORDER - 1};

            /// @brief Holds the off-diagonals at which a device of several units blocks a band.
            constexpr int64_t K_LARGE_BAND = 400;

            /// @brief Holds the two modes a public call is run in.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /** @brief Holds a queued event that stays incomplete until the gate is opened. */
            class DependencyGate final
            {
                public:
                    /** @brief Submits a host task to queue that blocks until open() is called. */
                    explicit DependencyGate(sycl::queue& queue)
                        : event_(queue.submit(
                              [opened = opened_.get_future().share()](sycl::handler& h)
                              {
                                  h.host_task([opened]() { opened.wait(); });
                              }))
                    {

                    }

                    DependencyGate(const DependencyGate&) = delete;
                    DependencyGate& operator=(const DependencyGate&) = delete;
                    DependencyGate(DependencyGate&&) = delete;
                    DependencyGate& operator=(DependencyGate&&) = delete;

                    /** @brief Opens the gate, so that a test leaving early blocks no host task. */
                    ~DependencyGate()
                    {
                        open();
                    }

                    /** @brief Returns the event a kernel is handed as its dependency. */
                    [[nodiscard]] const sycl::event& event() const noexcept { return event_; }

                    /** @brief Lets the host task finish; a second call does nothing. */
                    void open()
                    {
                        if (is_open_)
                            return;

                        is_open_ = true;
                        opened_.set_value();
                    }

                private:
                    /// @brief Fulfilled by open(); declared first, because event_ is built from it.
                    std::promise<void> opened_;

                    /// @brief The event of the blocked host task.
                    sycl::event event_;

                    /// @brief Whether open() has fulfilled the promise.
                    bool is_open_ = false;
            };

            /** @brief Returns whether every command behind an event has finished. */
            [[nodiscard]] bool is_complete(const sycl::event& event)
            {
                return event.get_info<sycl::info::event::command_execution_status>() ==
                       sycl::info::event_command_status::complete;
            }

            /** @brief Holds the flags of one triangular call. */
            struct TriangularFlags
            {
                Core::Layout layout = Core::Layout::RowMajor;     /**< The memory order. */
                Core::Side side = Core::Side::Left;                /**< The side of op(A). */
                Core::Uplo uplo = Core::Uplo::Lower;               /**< The triangle read. */
                Core::Transpose trans = Core::Transpose::NoTrans; /**< The op applied to A. */
                Core::Diag diag = Core::Diag::NonUnit;            /**< The diagonal read. */
            };

            /** @brief Returns the flags of a call as one line of a failure message. */
            [[nodiscard]] std::string describe(const TriangularFlags& flags)
            {
                return std::string(flags.layout == Core::Layout::RowMajor ? "ROW_MAJOR"
                                                                           : "COLUMN_MAJOR") +
                       " " + Core::to_string(flags.side) + " " + Core::to_string(flags.uplo) +
                       " " + Core::to_string(flags.trans) + " " + Core::to_string(flags.diag);
            }

            /**
             * @brief Returns every flag combination over the given sides: both layouts, both
             *        triangles, both diagonals, and each op the element type tells apart.
             * @tparam T The element type; CONJ_TRANS is listed for a complex one only.
             */
            template<typename T>
            [[nodiscard]] std::vector<TriangularFlags> flags_over(
                std::initializer_list<Core::Side> sides)
            {
                std::vector<Core::Transpose> transposes = {Core::Transpose::NoTrans,
                                                           Core::Transpose::Trans};
                if (Core::is_complex_v<T>)
                    transposes.push_back(Core::Transpose::ConjTrans);

                std::vector<TriangularFlags> all;
                for (const Core::Layout layout :
                     {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                    for (const Core::Side side : sides)
                        for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                            for (const Core::Transpose trans : transposes)
                                for (const Core::Diag diag :
                                     {Core::Diag::NonUnit, Core::Diag::Unit})
                                    all.push_back(
                                        TriangularFlags{layout, side, uplo, trans, diag});
                return all;
            }

            /** @brief Returns a real value as an element of T with a zero imaginary part. */
            template<typename T>
            [[nodiscard]] T element_of(double value) noexcept
            {
                return static_cast<T>(static_cast<Kernels::RealType<T>>(value));
            }

            /** @brief Returns the element no kernel may read: NaN in every component. */
            template<typename T>
            [[nodiscard]] T poison() noexcept
            {
                using R = Kernels::RealType<T>;
                const R nan = std::numeric_limits<R>::quiet_NaN();
                if constexpr (Core::is_complex_v<T>)
                    return T(nan, nan);
                else
                    return nan;
            }

            /** @brief Draws integers and binary fractions from one generator with a fixed seed. */
            class Draws final
            {
                public:
                    /** @brief Seeds the generator. */
                    explicit Draws(std::uint32_t seed)
                        : generator_(seed)
                    {

                    }

                    /** @brief Returns an integer from low to high, both included. */
                    [[nodiscard]] int integer(int low, int high)
                    {
                        const std::uint32_t span = static_cast<std::uint32_t>(high - low + 1);
                        return low + static_cast<int>(next() % span);
                    }

                    /** @brief Returns a multiple of 2^-20 in [-1, 1), which every type holds. */
                    [[nodiscard]] double fraction()
                    {
                        return static_cast<double>(next() >> 11) / K_FRACTION_STEPS - 1.0;
                    }

                    /** @brief Returns an integer of T, a Gaussian integer for a complex T. */
                    template<typename T>
                    [[nodiscard]] T integer_element(int low, int high)
                    {
                        using R = Kernels::RealType<T>;
                        if constexpr (Core::is_complex_v<T>)
                        {
                            const R real = static_cast<R>(integer(low, high));
                            const R imaginary = static_cast<R>(integer(low, high));
                            return T(real, imaginary);
                        }
                        else
                            return static_cast<R>(integer(low, high));
                    }

                    /** @brief Returns a fraction of T; a complex T takes the real part first. */
                    template<typename T>
                    [[nodiscard]] T fraction_element()
                    {
                        using R = Kernels::RealType<T>;
                        if constexpr (Core::is_complex_v<T>)
                        {
                            const R real = static_cast<R>(fraction());
                            const R imaginary = static_cast<R>(fraction());
                            return T(real, imaginary);
                        }
                        else
                            return static_cast<R>(fraction());
                    }

                private:
                    /** @brief Returns the next 32 bits of the generator. */
                    [[nodiscard]] std::uint32_t next()
                    {
                        return static_cast<std::uint32_t>(generator_());
                    }

                    /// @brief The generator every draw advances.
                    std::mt19937 generator_;
            };

            /** @brief Returns count integers of T from low to high. */
            template<typename T>
            [[nodiscard]] std::vector<T> integer_values(std::size_t count, int low, int high,
                                                        Draws& draws)
            {
                std::vector<T> values(count);
                for (T& value : values)
                    value = draws.integer_element<T>(low, high);
                return values;
            }

            /** @brief Returns count fractions of T in [-1, 1). */
            template<typename T>
            [[nodiscard]] std::vector<T> fraction_values(std::size_t count, Draws& draws)
            {
                std::vector<T> values(count);
                for (T& value : values)
                    value = draws.fraction_element<T>();
                return values;
            }

            /** @brief Returns 1, 2, -2 or 4, or 2i or -2i if complex: an exact divisor. */
            template<typename T>
            [[nodiscard]] T exact_pivot(Draws& draws)
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    const T pivots[] = {T(1), T(2), T(-2), T(4), T(0, 2), T(0, -2)};
                    return pivots[draws.integer(0, 5)];
                }
                else
                {
                    const T pivots[] = {T(1), T(2), T(-2), T(4)};
                    return pivots[draws.integer(0, 3)];
                }
            }

            /**
             * @brief Returns a full square matrix of integers from -2 to 2 around a diagonal of
             *        exact pivots, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> exact_matrix(int64_t order, Draws& draws)
            {
                std::vector<T> full(static_cast<std::size_t>(order * order));
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        full[static_cast<std::size_t>(r * order + c)] =
                            (r == c) ? exact_pivot<T>(draws) : draws.integer_element<T>(-2, 2);
                return full;
            }

            /**
             * @brief Returns a full square matrix of fractions divided by the order around a
             *        diagonal of a fraction plus 2, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> dominant_matrix(int64_t order, Draws& draws)
            {
                std::vector<T> full(static_cast<std::size_t>(order * order));
                for (int64_t r = 0; r < order; ++r)
                {
                    for (int64_t c = 0; c < order; ++c)
                    {
                        const T value = draws.fraction_element<T>();
                        full[static_cast<std::size_t>(r * order + c)] =
                            (r == c) ? value + element_of<T>(2.0)
                                     : value / element_of<T>(static_cast<double>(order));
                    }
                }
                return full;
            }

            /** @brief Returns a full matrix with zero wherever a column is over k from its row. */
            template<typename T>
            [[nodiscard]] std::vector<T> banded(std::vector<T> full, int64_t order, int64_t k)
            {
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (r - c > k || c - r > k)
                            full[static_cast<std::size_t>(r * order + c)] = element_of<T>(0.0);
                return full;
            }

            /** @brief Returns whether a triangular call reads element (r, c) of A. */
            [[nodiscard]] bool is_read(const TriangularFlags& flags, int64_t r, int64_t c) noexcept
            {
                if (r == c)
                    return flags.diag != Core::Diag::Unit;
                return flags.uplo == Core::Uplo::Upper ? r < c : r > c;
            }

            /**
             * @brief Returns the packed storage of the triangle a call reads, poisoned elsewhere.
             * @param full order * order elements, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> packed_storage(const TriangularFlags& flags, int64_t order,
                                                        const std::vector<T>& full)
            {
                const Kernels::PackedAccess stored = Kernels::packed_access(
                    flags.layout, flags.uplo, Core::Transpose::NoTrans, order);

                std::vector<T> packed(static_cast<std::size_t>(order * (order + 1) / 2),
                                      poison<T>());
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (is_read(flags, r, c))
                            packed[static_cast<std::size_t>(stored.index(r, c))] =
                                full[static_cast<std::size_t>(r * order + c)];
                return packed;
            }

            /**
             * @brief Returns the band storage of the triangle a call reads, k + 1 elements per
             *        line, poisoned elsewhere.
             * @param full order * order elements, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> band_storage(const TriangularFlags& flags, int64_t order,
                                                      int64_t k, const std::vector<T>& full)
            {
                const Kernels::BandAccess stored = Kernels::triangle_band_access(
                    flags.layout, flags.uplo, Core::Transpose::NoTrans, k, k + 1);

                std::vector<T> band(static_cast<std::size_t>(order * (k + 1)), poison<T>());
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (is_read(flags, r, c) && stored.holds(r, c))
                            band[static_cast<std::size_t>(stored.index(r, c))] =
                                full[static_cast<std::size_t>(r * order + c)];
                return band;
            }

            /** @brief Returns every element of values times factor. */
            template<typename T>
            [[nodiscard]] std::vector<T> scaled(const std::vector<T>& values, T factor)
            {
                std::vector<T> out(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = factor * values[i];
                return out;
            }

            /** @brief Returns values spread increment apart, with gap in every element between. */
            template<typename T>
            [[nodiscard]] std::vector<T> spaced(const std::vector<T>& values, int64_t increment,
                                                T gap)
            {
                if (values.empty())
                    return {};

                const std::size_t step = static_cast<std::size_t>(increment);
                std::vector<T> out((values.size() - 1) * step + 1, gap);
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i * step] = values[i];
                return out;
            }

            /** @brief Returns the elements of values from first up to, and not including, last. */
            template<typename T>
            [[nodiscard]] std::vector<T> slice_of(const std::vector<T>& values, std::size_t first,
                                                  std::size_t last)
            {
                return std::vector<T>(values.begin() + static_cast<std::ptrdiff_t>(first),
                                      values.begin() + static_cast<std::ptrdiff_t>(last));
            }

            /** @brief Returns the offset of element (r, c) of a dense rows x cols matrix. */
            [[nodiscard]] constexpr std::size_t offset_of(Core::Layout layout, int64_t rows,
                                                          int64_t cols, int64_t r,
                                                          int64_t c) noexcept
            {
                return static_cast<std::size_t>(layout == Core::Layout::RowMajor ? r * cols + c
                                                                                  : c * rows + r);
            }

            /** @brief Returns extent as a leading dimension: 1 when the axis is empty. */
            [[nodiscard]] constexpr int64_t leading(int64_t extent) noexcept
            {
                return extent > 0 ? extent : 1;
            }

            /**
             * @brief Holds a triangular call on the host as steps and vectors: op(A) * X = B on
             *        the left, X * op(A) = B on the right.
             * @tparam T The element type.
             */
            template<typename T>
            class HostSolve final
            {
                public:
                    /**
                     * @brief Binds the flags, the extents and a full square matrix.
                     * @param full order * order elements, row by row; the call reads one triangle.
                     */
                    HostSolve(const TriangularFlags& flags, int64_t order, int64_t vectors,
                              std::vector<T> full)
                        : flags_(flags), order_(order), vectors_(vectors), full_(std::move(full))
                    {

                    }

                    /** @brief Returns the flags of the call. */
                    [[nodiscard]] const TriangularFlags& flags() const noexcept { return flags_; }

                    /** @brief Returns the steps of one substitution: the order of A. */
                    [[nodiscard]] int64_t order() const noexcept { return order_; }

                    /** @brief Returns the independent vectors of B. */
                    [[nodiscard]] int64_t vectors() const noexcept { return vectors_; }

                    /** @brief Returns the rows of B. */
                    [[nodiscard]] int64_t rows() const noexcept
                    {
                        return is_left() ? order_ : vectors_;
                    }

                    /** @brief Returns the columns of B. */
                    [[nodiscard]] int64_t cols() const noexcept
                    {
                        return is_left() ? vectors_ : order_;
                    }

                    /** @brief Returns the elements of B. */
                    [[nodiscard]] std::size_t elements() const noexcept
                    {
                        return static_cast<std::size_t>(order_ * vectors_);
                    }

                    /** @brief Returns the leading dimension of the dense A. */
                    [[nodiscard]] int64_t lda() const noexcept { return leading(order_); }

                    /** @brief Returns the leading dimension of the dense B. */
                    [[nodiscard]] int64_t ldb() const noexcept
                    {
                        return leading(flags_.layout == Core::Layout::RowMajor ? cols() : rows());
                    }

                    /** @brief Returns whether unknown t has a factor in step s, itself included. */
                    [[nodiscard]] bool is_stored(int64_t s, int64_t t) const noexcept
                    {
                        const Position p = source(s, t);
                        return p.row == p.col || is_off_diagonal_stored(p);
                    }

                    /** @brief Returns the factor of unknown t in step s; 0 off the triangle. */
                    [[nodiscard]] T coefficient(int64_t s, int64_t t) const
                    {
                        const Position p = source(s, t);
                        if (p.row == p.col && flags_.diag == Core::Diag::Unit)
                            return element_of<T>(1.0);
                        if (p.row != p.col && !is_off_diagonal_stored(p))
                            return element_of<T>(0.0);
                        return Kernels::conjugate_if(Core::is_conjugated(flags_.trans),
                                                     full_[at(p.row, p.col)]);
                    }

                    /** @brief Returns the offset of unknown (s, v) in the dense B. */
                    [[nodiscard]] std::size_t unknown(int64_t s, int64_t v) const noexcept
                    {
                        return is_left() ? offset_of(flags_.layout, rows(), cols(), s, v)
                                         : offset_of(flags_.layout, rows(), cols(), v, s);
                    }

                    /** @brief Returns the dense A, poisoned wherever the call must not read. */
                    [[nodiscard]] std::vector<T> stored_matrix() const
                    {
                        std::vector<T> stored(full_.size(), poison<T>());
                        for (int64_t r = 0; r < order_; ++r)
                        {
                            for (int64_t c = 0; c < order_; ++c)
                            {
                                const bool read = (r == c) ? flags_.diag != Core::Diag::Unit
                                                           : is_off_diagonal_stored(Position{r, c});
                                if (read)
                                    stored[offset_of(flags_.layout, order_, order_, r, c)] =
                                        full_[at(r, c)];
                            }
                        }
                        return stored;
                    }

                    /**
                     * @brief Returns op(A) * X on the left, X * op(A) on the right, laid out as B.
                     * @param x The elements of X, laid out as B.
                     */
                    [[nodiscard]] std::vector<T> product(const std::vector<T>& x) const
                    {
                        std::vector<T> out(x.size(), element_of<T>(0.0));
                        for (int64_t v = 0; v < vectors_; ++v)
                        {
                            for (int64_t s = 0; s < order_; ++s)
                            {
                                T sum = element_of<T>(0.0);
                                for (int64_t t = 0; t < order_; ++t)
                                    if (is_stored(s, t))
                                        sum += coefficient(s, t) * x[unknown(t, v)];
                                out[unknown(s, v)] = sum;
                            }
                        }
                        return out;
                    }

                private:
                    /** @brief Holds the row and the column of one element of A. */
                    struct Position
                    {
                        int64_t row; /**< The row of the element. */
                        int64_t col; /**< The column of the element. */
                    };

                    /** @brief Returns whether op(A) stands on the left of X. */
                    [[nodiscard]] bool is_left() const noexcept
                    {
                        return flags_.side == Core::Side::Left;
                    }

                    /** @brief Returns the offset of element (r, c) of the full matrix. */
                    [[nodiscard]] std::size_t at(int64_t r, int64_t c) const noexcept
                    {
                        return static_cast<std::size_t>(r * order_ + c);
                    }

                    /** @brief Returns the element of A with the factor of unknown t in step s. */
                    [[nodiscard]] Position source(int64_t s, int64_t t) const noexcept
                    {
                        const Position of_op = is_left() ? Position{s, t} : Position{t, s};
                        return Core::is_transposed(flags_.trans)
                                   ? Position{of_op.col, of_op.row}
                                   : of_op;
                    }

                    /** @brief Returns whether an off-diagonal element lies in the triangle read. */
                    [[nodiscard]] bool is_off_diagonal_stored(const Position& p) const noexcept
                    {
                        return flags_.uplo == Core::Uplo::Upper ? p.row < p.col : p.row > p.col;
                    }

                    /// @brief The flags of the call.
                    TriangularFlags flags_;

                    /// @brief The order of A.
                    int64_t order_;

                    /// @brief The independent vectors of B.
                    int64_t vectors_;

                    /// @brief The full square matrix, row by row.
                    std::vector<T> full_;
            };

            /** @brief Returns the queue every kernel of this file is launched on. */
            [[nodiscard]] sycl::queue& test_queue()
            {
                return TestEnvironment::queue();
            }

            /**
             * @brief Returns a rank-1 tensor of the engine that holds values.
             * @return A tensor without storage when values is empty.
             */
            template<typename E>
            [[nodiscard]] Tensor device_buffer(Engine& engine, const std::vector<E>& values)
            {
                Tensor buffer = engine.create_tensor({static_cast<int64_t>(values.size())},
                                                     sushiblas_dtype<E>());
                if (!values.empty())
                    test_queue().memcpy(buffer.data(), values.data(), values.size() * sizeof(E))
                        .wait();
                return buffer;
            }

            /** @brief Returns the elements of a dense tensor of E. */
            template<typename E>
            [[nodiscard]] std::vector<E> read_buffer(const Tensor& buffer)
            {
                std::vector<E> host(static_cast<std::size_t>(buffer.num_elements));
                if (!host.empty())
                    test_queue().memcpy(host.data(), buffer.data(), host.size() * sizeof(E))
                        .wait();
                return host;
            }

            /** @brief Returns the first element of a tensor as E, or null without storage. */
            template<typename E>
            [[nodiscard]] E* pointer_of(Tensor& buffer)
            {
                return static_cast<E*>(buffer.data_or_null());
            }

            /** @brief Returns a dense view of a buffer's elements as dims in a layout. */
            [[nodiscard]] Tensor matrix_view(const Tensor& buffer,
                                             std::initializer_list<int64_t> dims,
                                             Core::Layout layout)
            {
                return Tensor(buffer.storage, dims, buffer.dtype, buffer.storage_offset, layout);
            }

            /** @brief Returns a rank-1 view of every increment-th element of a buffer. */
            [[nodiscard]] Tensor strided_view(const Tensor& buffer, int64_t length,
                                              int64_t increment)
            {
                const int64_t dims[] = {length};
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                strides[0] = increment;
                return Tensor(buffer.storage, SushiRuntime::span<const int64_t>(dims, 1), strides,
                              buffer.dtype, buffer.storage_offset, buffer.layout);
            }

            /** @brief Expects two results to hold the same bytes. */
            template<typename E>
            void expect_same_bits(const std::vector<E>& actual, const std::vector<E>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                if (actual.empty())
                    return;
                EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(),
                                         actual.size() * sizeof(E)));
            }

            /** @brief Expects two results to hold equal values; stops at the first that differs. */
            template<typename E>
            void expect_equal_values(const std::vector<E>& actual, const std::vector<E>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    ASSERT_EQ(actual[i], expected[i]) << "index " << i;
            }

            /**
             * @brief Runs the triangular kernels and the public calls on host-described systems.
             * @tparam T The element type.
             */
            template<typename T>
            class TriangularParallelFixture : public SushiBLASTest
            {
                protected:
                    /**
                     * @brief Submits the SYCL TRSM kernel on device operands at one block width.
                     * @param shape Gives the flags and the extents of one matrix of the batch.
                     * @param str_a Elements from one A to the next; 0 shares one A.
                     */
                    [[nodiscard]] sycl::event submit_solve(const HostSolve<T>& shape, Tensor& A,
                                                           int64_t str_a, Tensor& B,
                                                           int64_t batch_size, T alpha,
                                                           int64_t width,
                                                           const std::vector<sycl::event>& deps)
                    {
                        const TriangularFlags& flags = shape.flags();
                        return Kernels::SYCL::trsm<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.side, flags.uplo, flags.trans, flags.diag, shape.rows(),
                            shape.cols(), alpha, pointer_of<const T>(A), shape.lda(), str_a,
                            pointer_of<T>(B), shape.ldb(), shape.rows() * shape.cols(), batch_size,
                            nullptr, deps);
                    }

                    /**
                     * @brief Returns the device B after the SYCL kernel solved a batch.
                     * @param a One dense A, or batch_size of them when str_a is not 0.
                     * @param b The whole device B, which may hold more elements than the batch.
                     */
                    [[nodiscard]] std::vector<T> run_solve_kernel(const HostSolve<T>& shape,
                                                                  const std::vector<T>& a,
                                                                  int64_t str_a,
                                                                  const std::vector<T>& b,
                                                                  int64_t batch_size, T alpha,
                                                                  int64_t width)
                    {
                        Tensor A = device_buffer<T>(*engine, a);
                        Tensor B = device_buffer<T>(*engine, b);
                        submit_solve(shape, A, str_a, B, batch_size, alpha, width, {})
                            .wait_and_throw();
                        return read_buffer<T>(B);
                    }

                    /** @brief Returns X of one system, solved by the SYCL kernel at one width. */
                    [[nodiscard]] std::vector<T> solve_with_kernel(const HostSolve<T>& system,
                                                                   const std::vector<T>& b, T alpha,
                                                                   int64_t width)
                    {
                        return run_solve_kernel(system, system.stored_matrix(), 0, b, 1, alpha,
                                                width);
                    }

                    /** @brief Returns X of one system, solved by Level3::trsm in a mode. */
                    [[nodiscard]] std::vector<T> solve_through_view(const HostSolve<T>& system,
                                                                    const std::vector<T>& b,
                                                                    double alpha,
                                                                    ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        engine->set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor b_buffer = device_buffer<T>(*engine, b);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);
                        Tensor B = matrix_view(b_buffer, {system.rows(), system.cols()},
                                               flags.layout);

                        SushiBLAS::BLASOps(*engine).trsm(A, B, flags.side, flags.uplo, flags.trans,
                                                         flags.diag, alpha);
                        engine->execute().value();
                        return read_buffer<T>(b_buffer);
                    }

                    /** @brief Returns x of a one-vector system, solved by Level2::trsv in mode. */
                    [[nodiscard]] std::vector<T> solve_vector_through_view(
                        const HostSolve<T>& system, const std::vector<T>& b, ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        engine->set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);

                        SushiBLAS::BLASOps(*engine).trsv(A, x, flags.uplo, flags.trans, flags.diag);
                        engine->execute().value();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Returns the whole buffer of x after Level2::trmv ran in a mode.
                     * @param spaced_x The elements of x, increment apart.
                     */
                    [[nodiscard]] std::vector<T> multiply_through_view(
                        const HostSolve<T>& system, const std::vector<T>& spaced_x,
                        int64_t increment, ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        engine->set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x_buffer = device_buffer<T>(*engine, spaced_x);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);
                        Tensor x = strided_view(x_buffer, system.order(), increment);

                        SushiBLAS::BLASOps(*engine).trmv(A, x, flags.uplo, flags.trans, flags.diag);
                        engine->execute().value();
                        return read_buffer<T>(x_buffer);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TRMV kernel ran in one
                     *        row shape.
                     * @param group_width 0 for one work-item per row, else the work-group width.
                     */
                    [[nodiscard]] std::vector<T> multiply_with_kernel(
                        const HostSolve<T>& system, const std::vector<T>& spaced_x,
                        int64_t increment, std::size_t group_width)
                    {
                        const TriangularFlags& flags = system.flags();
                        const std::size_t order = static_cast<std::size_t>(system.order());

                        Tensor A = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x = device_buffer<T>(*engine, spaced_x);
                        Tensor work =
                            device_buffer<T>(*engine, std::vector<T>(order, element_of<T>(0.0)));

                        Kernels::SYCL::trmv<T>(test_queue(), group_width, flags.layout, flags.uplo,
                                               flags.trans, flags.diag, system.order(),
                                               pointer_of<const T>(A), system.lda(),
                                               pointer_of<T>(x), increment, pointer_of<T>(work),
                                               {})
                            .wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Submits the SYCL TPSV kernel on device operands at one block width.
                     * @param AP The packed storage of system's triangle.
                     */
                    [[nodiscard]] sycl::event submit_packed_solve(
                        const HostSolve<T>& system, Tensor& AP, Tensor& x, int64_t increment,
                        int64_t width, const std::vector<sycl::event>& deps)
                    {
                        const TriangularFlags& flags = system.flags();
                        return Kernels::SYCL::tpsv<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.uplo, flags.trans, flags.diag, system.order(),
                            pointer_of<const T>(AP), pointer_of<T>(x), increment, nullptr, deps);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TPSV kernel solved it.
                     * @param full The matrix system was built from.
                     * @param spaced_b The right-hand side, increment apart.
                     */
                    [[nodiscard]] std::vector<T> solve_packed_with_kernel(
                        const HostSolve<T>& system, const std::vector<T>& full,
                        const std::vector<T>& spaced_b, int64_t increment, int64_t width)
                    {
                        Tensor AP = device_buffer<T>(
                            *engine, packed_storage(system.flags(), system.order(), full));
                        Tensor x = device_buffer<T>(*engine, spaced_b);
                        submit_packed_solve(system, AP, x, increment, width, {}).wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TBSV kernel solved it.
                     * @param full The matrix system was built from, zero beyond k off-diagonals.
                     * @param spaced_b The right-hand side, increment apart.
                     */
                    [[nodiscard]] std::vector<T> solve_band_with_kernel(
                        const HostSolve<T>& system, int64_t k, const std::vector<T>& full,
                        const std::vector<T>& spaced_b, int64_t increment, int64_t width)
                    {
                        const TriangularFlags& flags = system.flags();
                        Tensor A = device_buffer<T>(
                            *engine, band_storage(flags, system.order(), k, full));
                        Tensor x = device_buffer<T>(*engine, spaced_b);

                        Kernels::SYCL::tbsv<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.uplo, flags.trans, flags.diag, system.order(), k,
                            pointer_of<const T>(A), k + 1, pointer_of<T>(x), increment, nullptr,
                            {})
                            .wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns x of a packed system, solved by Level2::tpsv in ForceSYCL. */
                    [[nodiscard]] std::vector<T> solve_packed_through_view(
                        const HostSolve<T>& system, const std::vector<T>& full,
                        const std::vector<T>& b)
                    {
                        const TriangularFlags& flags = system.flags();
                        engine->set_mode(ExecutionMode::ForceSYCL);

                        Tensor buffer = device_buffer<T>(
                            *engine, packed_storage(flags, system.order(), full));
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor AP = matrix_view(
                            buffer, {static_cast<int64_t>(buffer.num_elements)}, flags.layout);

                        SushiBLAS::BLASOps(*engine).tpsv(AP, x, flags.uplo, flags.trans, flags.diag);
                        engine->execute().value();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns x of a band system, solved by Level2::tbsv in ForceSYCL. */
                    [[nodiscard]] std::vector<T> solve_band_through_view(
                        const HostSolve<T>& system, int64_t k, const std::vector<T>& full,
                        const std::vector<T>& b)
                    {
                        const TriangularFlags& flags = system.flags();
                        const bool by_rows = (flags.layout == Core::Layout::RowMajor);
                        engine->set_mode(ExecutionMode::ForceSYCL);

                        Tensor buffer = device_buffer<T>(
                            *engine, band_storage(flags, system.order(), k, full));
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor A = by_rows
                            ? matrix_view(buffer, {system.order(), k + 1}, flags.layout)
                            : matrix_view(buffer, {k + 1, system.order()}, flags.layout);

                        SushiBLAS::BLASOps(*engine).tbsv(A, x, flags.uplo, k, flags.trans,
                                                         flags.diag);
                        engine->execute().value();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns the unit roundoff of the type the kernel of mode sums in. */
                    [[nodiscard]] static double summing_unit(ExecutionMode mode) noexcept
                    {
                        return mode == ExecutionMode::ForceReference
                                   ? unit_roundoff<Kernels::ReferenceAccumulator<T>>()
                                   : unit_roundoff<Kernels::Accumulator<T>>();
                    }

                    /**
                     * @brief Returns the roundings the kernel of mode makes on a sum of len terms.
                     * @param fixed The roundings counted whatever the length.
                     */
                    [[nodiscard]] static std::size_t roundings(ExecutionMode mode, int64_t len,
                                                               std::size_t fixed) noexcept
                    {
                        const bool compensated = mode == ExecutionMode::ForceReference &&
                                                 Kernels::ReferenceSum<T>::K_IS_COMPENSATED;
                        return (compensated ? K_COMPENSATED_ROUNDINGS
                                            : static_cast<std::size_t>(len)) + fixed;
                    }

                    /**
                     * @brief Expects every step of a solved real system to leave a residual within
                     *        the bound of DESIGN_BLAS_L3, section 7.6.
                     * @param x The solution a kernel of mode wrote over b.
                     */
                    void expect_residual_within_bound(const HostSolve<T>& system,
                                                      const std::vector<T>& b, double alpha,
                                                      const std::vector<T>& x, ExecutionMode mode)
                    {
                        ASSERT_EQ(x.size(), b.size());
                        const double unit = summing_unit(mode);

                        for (int64_t v = 0; v < system.vectors(); ++v)
                        {
                            for (int64_t s = 0; s < system.order(); ++s)
                            {
                                TruthSum sum;
                                sum.add_product(alpha,
                                                static_cast<double>(b[system.unknown(s, v)]));

                                int64_t len = 0;
                                for (int64_t t = 0; t < system.order(); ++t)
                                {
                                    if (!system.is_stored(s, t))
                                        continue;
                                    sum.add_product(-static_cast<double>(system.coefficient(s, t)),
                                                    static_cast<double>(x[system.unknown(t, v)]));
                                    if (t != s)
                                        ++len;
                                }

                                const TruthValue residual = sum.value();
                                const double bound =
                                    rounding_bound(roundings(mode, len, K_STEP_ROUNDINGS), unit,
                                                   residual.magnitude) +
                                    2.0 * unit_roundoff<T>() * residual.magnitude;
                                ASSERT_LE(std::fabs(residual.value), bound)
                                    << "step " << s << ", vector " << v;
                            }
                        }
                    }

                    /**
                     * @brief Expects every row of a real triangular product within its rounding
                     *        bound of the truth.
                     * @param actual The product a kernel of mode wrote over x.
                     */
                    void expect_product_within_bound(const HostSolve<T>& system,
                                                     const std::vector<T>& x,
                                                     const std::vector<T>& actual,
                                                     ExecutionMode mode)
                    {
                        ASSERT_EQ(actual.size(), x.size());
                        const double unit = summing_unit(mode);

                        for (int64_t s = 0; s < system.order(); ++s)
                        {
                            TruthSum sum;
                            int64_t len = 0;
                            for (int64_t t = 0; t < system.order(); ++t)
                            {
                                if (!system.is_stored(s, t))
                                    continue;
                                const T factor = system.coefficient(s, t);
                                sum.add_product(static_cast<double>(factor),
                                                static_cast<double>(x[system.unknown(t, 0)]));
                                ++len;
                            }

                            const TruthValue truth = sum.value();
                            const double bound =
                                rounding_bound(roundings(mode, len, K_FIXED_ROUNDINGS), unit,
                                               truth.magnitude) +
                                unit_roundoff<T>() * std::fabs(truth.value);
                            ASSERT_NEAR(static_cast<double>(actual[system.unknown(s, 0)]),
                                        truth.value, bound)
                                << "row " << s;
                        }
                    }
            };

            /** @brief Returns the flags of a row-major, left-side, non-unit call. */
            [[nodiscard]] TriangularFlags plain_flags(Core::Uplo uplo,
                                                      Core::Transpose trans) noexcept
            {
                TriangularFlags flags;
                flags.uplo = uplo;
                flags.trans = trans;
                return flags;
            }
        } // namespace

        /** @brief Runs the triangular kernels on every element type but HALF. */
        template<typename T>
        class Unit_TriangularParallel : public TriangularParallelFixture<T>
        {

        };

        /// @brief Element types the triangular kernels are tested on.
        using TriangularTypes =
            ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_TriangularParallel, TriangularTypes);

        /** @brief Runs the cases measured against a double-precision truth, on the real types. */
        template<typename T>
        class Unit_TriangularParallelReal : public TriangularParallelFixture<T>
        {

        };

        /// @brief Element types a double-precision truth exists for.
        using TriangularRealTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_TriangularParallelReal, TriangularRealTypes);

        /**
         * @brief Tests that a solve leaves the same bytes at every block width.
         */
        TYPED_TEST(Unit_TriangularParallel, EveryBlockWidthGivesTheSameBits)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T alpha = element_of<T>(K_ROUNDED_ALPHA);

            for (const TriangularFlags& flags :
                 flags_over<T>({Core::Side::Left, Core::Side::Right}))
            {
                SCOPED_TRACE(describe(flags));
                const HostSolve<T> system(flags, K_ORDER, K_VECTORS,
                                          dominant_matrix<T>(K_ORDER, draws));
                const std::vector<T> b = fraction_values<T>(system.elements(), draws);
                const std::vector<T> whole = this->solve_with_kernel(system, b, alpha, K_ORDER);

                for (const int64_t width : K_BLOCK_WIDTHS)
                {
                    SCOPED_TRACE(::testing::Message() << "block width " << width);
                    expect_same_bits(this->solve_with_kernel(system, b, alpha, width), whole);
                }
            }
        }

        /**
         * @brief Tests that a blocked solve of an integer system returns its solution with no
         *        error, as the reference kernel does.
         */
        TYPED_TEST(Unit_TriangularParallel, ABlockedSolveIsExactOnIntegerSystems)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T alpha = element_of<T>(K_EXACT_ALPHA);

            for (const TriangularFlags& flags :
                 flags_over<T>({Core::Side::Left, Core::Side::Right}))
            {
                SCOPED_TRACE(describe(flags));
                const HostSolve<T> system(flags, K_ORDER, K_VECTORS,
                                          exact_matrix<T>(K_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(system.elements(), -3, 3, draws);
                const std::vector<T> b = system.product(solution);
                const std::vector<T> expected = scaled(solution, alpha);

                for (const int64_t width : K_EXACT_WIDTHS)
                {
                    SCOPED_TRACE(::testing::Message() << "block width " << width);
                    expect_equal_values(this->solve_with_kernel(system, b, alpha, width),
                                        expected);
                }

                SCOPED_TRACE("ForceReference");
                expect_equal_values(this->solve_through_view(system, b, K_EXACT_ALPHA,
                                                             ExecutionMode::ForceReference),
                                    expected);
            }
        }

        /**
         * @brief Tests that a batch solved in blocks equals each of its matrices solved alone.
         */
        TYPED_TEST(Unit_TriangularParallel, ABatchIsSolvedMatrixByMatrix)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T alpha = element_of<T>(K_ROUNDED_ALPHA);
            const std::size_t matrix_elements = static_cast<std::size_t>(K_ORDER * K_ORDER);

            for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
            {
                for (const bool shares_a : {false, true})
                {
                    TriangularFlags flags;
                    flags.side = side;
                    SCOPED_TRACE(describe(flags));
                    SCOPED_TRACE(shares_a ? "A of rank 2" : "A batched");

                    std::vector<HostSolve<T>> systems;
                    std::vector<T> a;
                    std::vector<T> b;
                    std::vector<T> full;
                    for (int64_t k = 0; k < K_BATCH; ++k)
                    {
                        if (k == 0 || !shares_a)
                        {
                            full = dominant_matrix<T>(K_ORDER, draws);
                            const std::vector<T> stored =
                                HostSolve<T>(flags, K_ORDER, K_VECTORS, full).stored_matrix();
                            a.insert(a.end(), stored.begin(), stored.end());
                        }
                        systems.emplace_back(flags, K_ORDER, K_VECTORS, full);

                        const std::vector<T> rhs =
                            fraction_values<T>(systems.back().elements(), draws);
                        b.insert(b.end(), rhs.begin(), rhs.end());
                    }

                    const std::vector<T> solved = this->run_solve_kernel(
                        systems.front(), a, shares_a ? 0 : static_cast<int64_t>(matrix_elements),
                        b, K_BATCH, alpha, K_NARROW_WIDTH);
                    ASSERT_EQ(solved.size(), b.size());

                    const std::size_t per_matrix = systems.front().elements();
                    for (std::size_t k = 0; k < systems.size(); ++k)
                    {
                        SCOPED_TRACE(::testing::Message() << "matrix " << k);
                        const std::vector<T> alone = this->solve_with_kernel(
                            systems[k], slice_of(b, k * per_matrix, (k + 1) * per_matrix), alpha,
                            K_ORDER);
                        expect_same_bits(
                            slice_of(solved, k * per_matrix, (k + 1) * per_matrix), alone);
                    }
                }
            }
        }

        /**
         * @brief Tests that every launch of a blocked solve stays behind the events it is given.
         */
        TYPED_TEST(Unit_TriangularParallel, ABlockedSolveWaitsForItsDependencies)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T alpha = element_of<T>(K_EXACT_ALPHA);

            const HostSolve<T> system(TriangularFlags{}, K_ORDER, K_VECTORS,
                                      exact_matrix<T>(K_ORDER, draws));
            const std::vector<T> solution = integer_values<T>(system.elements(), -3, 3, draws);

            Tensor A = device_buffer<T>(*this->engine, system.stored_matrix());
            Tensor B = device_buffer<T>(*this->engine, system.product(solution));

            DependencyGate gate(test_queue());
            sycl::event done = this->submit_solve(system, A, 0, B, 1, alpha, K_NARROW_WIDTH,
                                                  std::vector<sycl::event>{gate.event()});

            EXPECT_FALSE(is_complete(done))
                << "The solve's last event completed while its dependency was still running.";

            gate.open();
            done.wait_and_throw();
            expect_equal_values(read_buffer<T>(B), scaled(solution, alpha));
        }

        /**
         * @brief Tests that a solve without steps, or without vectors, finishes and writes nothing.
         */
        TYPED_TEST(Unit_TriangularParallel, AnEmptySystemReturnsAnEvent)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T alpha = element_of<T>(K_EXACT_ALPHA);
            const std::vector<T> untouched(static_cast<std::size_t>(K_EMPTY_EXTENT),
                                           element_of<T>(K_SENTINEL));

            const HostSolve<T> no_steps(TriangularFlags{}, 0, K_EMPTY_EXTENT, {});
            expect_same_bits(this->run_solve_kernel(no_steps, {}, 0, untouched, 1, alpha, 1),
                             untouched);

            const HostSolve<T> no_vectors(TriangularFlags{}, K_EMPTY_EXTENT, 0,
                                          exact_matrix<T>(K_EMPTY_EXTENT, draws));
            expect_same_bits(this->run_solve_kernel(no_vectors, no_vectors.stored_matrix(), 0,
                                                    untouched, 1, alpha, 1),
                             untouched);
        }

        /**
         * @brief Tests that TRMV of small integers equals the host product in both modes and
         *        in both row shapes of the SYCL kernel.
         */
        TYPED_TEST(Unit_TriangularParallel, TrmvIsExactOnSmallIntegers)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T gap = element_of<T>(K_SENTINEL);

            for (const int64_t order : K_PRODUCT_ORDERS)
            {
                for (const TriangularFlags& flags : flags_over<T>({Core::Side::Left}))
                {
                    const std::size_t elements = static_cast<std::size_t>(order * order);
                    const HostSolve<T> system(flags, order, 1,
                                              integer_values<T>(elements, -3, 3, draws));
                    const std::vector<T> x =
                        integer_values<T>(static_cast<std::size_t>(order), -3, 3, draws);
                    const std::vector<T> product = system.product(x);

                    for (const int64_t increment : K_INCREMENTS)
                    {
                        SCOPED_TRACE(describe(flags));
                        SCOPED_TRACE(::testing::Message() << "order " << order << ", increment "
                                                          << increment);

                        const std::vector<T> input = spaced(x, increment, gap);
                        const std::vector<T> expected = spaced(product, increment, gap);

                        for (const ExecutionMode mode : K_MODES)
                            expect_equal_values(
                                this->multiply_through_view(system, input, increment, mode),
                                expected);

                        for (const std::size_t group_width : K_GROUP_WIDTHS)
                            expect_equal_values(
                                this->multiply_with_kernel(system, input, increment, group_width),
                                expected);
                    }
                }
            }
        }

        /**
         * @brief Tests that two runs of one TRMV on the SYCL kernel leave the same bytes.
         */
        TYPED_TEST(Unit_TriangularParallel, TrmvRepeatsToTheBit)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const std::size_t order = static_cast<std::size_t>(K_PRODUCT_BOUND_ORDER);

            const HostSolve<T> system(TriangularFlags{}, K_PRODUCT_BOUND_ORDER, 1,
                                      fraction_values<T>(order * order, draws));
            const std::vector<T> x = fraction_values<T>(order, draws);

            const std::vector<T> first =
                this->multiply_through_view(system, x, 1, ExecutionMode::ForceSYCL);
            const std::vector<T> second =
                this->multiply_through_view(system, x, 1, ExecutionMode::ForceSYCL);
            expect_same_bits(first, second);
        }

        /**
         * @brief Tests that the snapshot of a TRMV returns to the pool and is lent again.
         */
        TYPED_TEST(Unit_TriangularParallel, TrmvReturnsItsSnapshotToThePool)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const std::size_t order = static_cast<std::size_t>(K_POOLED_ORDER);

            const HostSolve<T> system(TriangularFlags{}, K_POOLED_ORDER, 1,
                                      fraction_values<T>(order * order, draws));
            Tensor a_buffer = device_buffer<T>(*this->engine, system.stored_matrix());
            Tensor x = device_buffer<T>(*this->engine, fraction_values<T>(order, draws));
            Tensor A = matrix_view(a_buffer, {K_POOLED_ORDER, K_POOLED_ORDER},
                                   Core::Layout::RowMajor);

            ScratchPool& pool = this->engine->scratch_pool();
            this->engine->set_mode(ExecutionMode::ForceSYCL);
            const std::size_t outstanding_before =
                TestEnvironment::live_allocation_count() - pool.idle_block_count();

            SushiBLAS::BLASOps(*this->engine).trmv(A, x, Core::Uplo::Lower);
            this->engine->execute().value();

            EXPECT_EQ(outstanding_before,
                      TestEnvironment::live_allocation_count() - pool.idle_block_count())
                << "Every allocation the product made must be idle in the pool or freed.";

            const std::size_t live_after_first = TestEnvironment::live_allocation_count();
            const std::size_t idle_after_first = pool.idle_block_count();

            SushiBLAS::BLASOps(*this->engine).trmv(A, x, Core::Uplo::Lower);
            this->engine->execute().value();

            EXPECT_EQ(live_after_first, TestEnvironment::live_allocation_count());
            EXPECT_EQ(idle_after_first, pool.idle_block_count());
        }

        /**
         * @brief Tests that both launches of TRMV stay behind the events the kernel is given.
         */
        TYPED_TEST(Unit_TriangularParallel, TrmvWaitsForItsDependencies)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const std::size_t order = static_cast<std::size_t>(K_ORDER);

            const HostSolve<T> system(TriangularFlags{}, K_ORDER, 1,
                                      integer_values<T>(order * order, -3, 3, draws));
            const std::vector<T> x = integer_values<T>(order, -3, 3, draws);

            Tensor A = device_buffer<T>(*this->engine, system.stored_matrix());
            Tensor device_x = device_buffer<T>(*this->engine, x);
            Tensor work =
                device_buffer<T>(*this->engine, std::vector<T>(order, element_of<T>(0.0)));

            const TriangularFlags& flags = system.flags();
            DependencyGate gate(test_queue());
            sycl::event done = Kernels::SYCL::trmv<T>(
                test_queue(), flags.layout, flags.uplo, flags.trans, flags.diag, system.order(),
                pointer_of<const T>(A), system.lda(), pointer_of<T>(device_x), 1,
                pointer_of<T>(work), std::vector<sycl::event>{gate.event()});

            EXPECT_FALSE(is_complete(done))
                << "The product's last event completed while its dependency was still running.";

            gate.open();
            done.wait_and_throw();
            expect_equal_values(read_buffer<T>(device_x), system.product(x));
        }

        /**
         * @brief Tests that a packed and a band solve leave the same bytes at every block width.
         */
        TYPED_TEST(Unit_TriangularParallel, AVectorSolveGivesTheSameBitsAtEveryBlockWidth)
        {
            using T = TypeParam;
            Draws draws(K_SEED);

            for (const TriangularFlags& flags : flags_over<T>({Core::Side::Left}))
            {
                SCOPED_TRACE(describe(flags));
                const std::vector<T> full = dominant_matrix<T>(K_ORDER, draws);
                const std::vector<T> b =
                    fraction_values<T>(static_cast<std::size_t>(K_ORDER), draws);

                const HostSolve<T> packed(flags, K_ORDER, 1, full);
                const std::vector<T> whole =
                    this->solve_packed_with_kernel(packed, full, b, 1, K_ORDER);
                for (const int64_t width : K_BLOCK_WIDTHS)
                {
                    SCOPED_TRACE(::testing::Message() << "packed, block width " << width);
                    expect_same_bits(this->solve_packed_with_kernel(packed, full, b, 1, width),
                                     whole);
                }

                for (const int64_t k : K_BAND_WIDTHS)
                {
                    const std::vector<T> narrow = banded(full, K_ORDER, k);
                    const HostSolve<T> band(flags, K_ORDER, 1, narrow);
                    const std::vector<T> whole_band =
                        this->solve_band_with_kernel(band, k, narrow, b, 1, K_ORDER);
                    for (const int64_t width : K_BLOCK_WIDTHS)
                    {
                        SCOPED_TRACE(::testing::Message() << "band of " << k << ", block width "
                                                          << width);
                        expect_same_bits(
                            this->solve_band_with_kernel(band, k, narrow, b, 1, width),
                            whole_band);
                    }
                }
            }
        }

        /**
         * @brief Tests that a blocked packed and a blocked band solve return the solution of an
         *        integer system with no error and write nothing between the elements of x.
         */
        TYPED_TEST(Unit_TriangularParallel, ABlockedVectorSolveIsExactOnIntegerSystems)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const T gap = element_of<T>(K_SENTINEL);

            for (const TriangularFlags& flags : flags_over<T>({Core::Side::Left}))
            {
                const std::vector<T> full = exact_matrix<T>(K_ORDER, draws);
                const std::vector<T> solution =
                    integer_values<T>(static_cast<std::size_t>(K_ORDER), -3, 3, draws);

                for (const int64_t increment : K_INCREMENTS)
                {
                    SCOPED_TRACE(describe(flags));
                    SCOPED_TRACE(::testing::Message() << "increment " << increment);
                    const std::vector<T> expected = spaced(solution, increment, gap);

                    const HostSolve<T> packed(flags, K_ORDER, 1, full);
                    const std::vector<T> packed_b =
                        spaced(packed.product(solution), increment, gap);
                    for (const int64_t width : K_EXACT_WIDTHS)
                    {
                        SCOPED_TRACE(::testing::Message() << "packed, block width " << width);
                        expect_equal_values(
                            this->solve_packed_with_kernel(packed, full, packed_b, increment,
                                                           width),
                            expected);
                    }

                    for (const int64_t k : K_BAND_WIDTHS)
                    {
                        const std::vector<T> narrow = banded(full, K_ORDER, k);
                        const HostSolve<T> band(flags, K_ORDER, 1, narrow);
                        const std::vector<T> band_b =
                            spaced(band.product(solution), increment, gap);
                        for (const int64_t width : K_EXACT_WIDTHS)
                        {
                            SCOPED_TRACE(::testing::Message() << "band of " << k
                                                              << ", block width " << width);
                            expect_equal_values(
                                this->solve_band_with_kernel(band, k, narrow, band_b, increment,
                                                             width),
                                expected);
                        }
                    }
                }
            }
        }

        /**
         * @brief Tests that every launch of a blocked packed solve stays behind the events it is
         *        given.
         */
        TYPED_TEST(Unit_TriangularParallel, ABlockedVectorSolveWaitsForItsDependencies)
        {
            using T = TypeParam;
            Draws draws(K_SEED);

            const std::vector<T> full = exact_matrix<T>(K_ORDER, draws);
            const HostSolve<T> system(TriangularFlags{}, K_ORDER, 1, full);
            const std::vector<T> solution =
                integer_values<T>(static_cast<std::size_t>(K_ORDER), -3, 3, draws);

            Tensor AP = device_buffer<T>(*this->engine,
                                         packed_storage(system.flags(), K_ORDER, full));
            Tensor x = device_buffer<T>(*this->engine, system.product(solution));

            DependencyGate gate(test_queue());
            sycl::event done = this->submit_packed_solve(
                system, AP, x, 1, K_NARROW_WIDTH, std::vector<sycl::event>{gate.event()});

            EXPECT_FALSE(is_complete(done))
                << "The solve's last event completed while its dependency was still running.";

            gate.open();
            done.wait_and_throw();
            expect_equal_values(read_buffer<T>(x), solution);
        }

        /**
         * @brief Tests that a packed and a band solve of no rows finish.
         */
        TYPED_TEST(Unit_TriangularParallel, AnEmptyVectorSolveReturnsAnEvent)
        {
            using T = TypeParam;
            const HostSolve<T> empty(TriangularFlags{}, 0, 1, {});

            EXPECT_TRUE(this->solve_packed_with_kernel(empty, {}, {}, 1, 1).empty());
            EXPECT_TRUE(this->solve_band_with_kernel(empty, 0, {}, {}, 1, 1).empty());
        }

        /**
         * @brief Tests that the public TPSV and TBSV solve a large integer system with no error
         *        on the path the device plans.
         */
        TYPED_TEST(Unit_TriangularParallelReal, ThePlannedVectorPathIsExactAtALargeOrder)
        {
            using T = TypeParam;
            Draws draws(K_SEED);

            for (const Core::Uplo uplo : {Core::Uplo::Lower, Core::Uplo::Upper})
            {
                const TriangularFlags flags = plain_flags(uplo, Core::Transpose::NoTrans);
                SCOPED_TRACE(describe(flags));

                const std::vector<T> full = exact_matrix<T>(K_LARGE_ORDER, draws);
                const std::vector<T> x =
                    integer_values<T>(static_cast<std::size_t>(K_LARGE_ORDER), -3, 3, draws);

                const HostSolve<T> packed(flags, K_LARGE_ORDER, 1, full);
                expect_equal_values(
                    this->solve_packed_through_view(packed, full, packed.product(x)), x);

                const std::vector<T> narrow = banded(full, K_LARGE_ORDER, K_LARGE_BAND);
                const HostSolve<T> band(flags, K_LARGE_ORDER, 1, narrow);
                expect_equal_values(
                    this->solve_band_through_view(band, K_LARGE_BAND, narrow, band.product(x)),
                    x);
            }
        }

        /**
         * @brief Tests that the public TRSV and TRSM solve a large integer system with no error
         *        on the path the device plans.
         */
        TYPED_TEST(Unit_TriangularParallelReal, ThePlannedPathIsExactAtALargeOrder)
        {
            using T = TypeParam;
            Draws draws(K_SEED);

            for (const Core::Uplo uplo : {Core::Uplo::Lower, Core::Uplo::Upper})
            {
                const TriangularFlags flags = plain_flags(uplo, Core::Transpose::NoTrans);
                SCOPED_TRACE(describe(flags));

                const HostSolve<T> single(flags, K_LARGE_ORDER, 1,
                                          exact_matrix<T>(K_LARGE_ORDER, draws));
                const std::vector<T> x = integer_values<T>(single.elements(), -3, 3, draws);
                expect_equal_values(
                    this->solve_vector_through_view(single, single.product(x),
                                                    ExecutionMode::ForceSYCL),
                    x);

                const HostSolve<T> several(flags, K_LARGE_ORDER, K_LARGE_VECTORS,
                                           exact_matrix<T>(K_LARGE_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(several.elements(), -3, 3, draws);
                expect_equal_values(
                    this->solve_through_view(several, several.product(solution), K_EXACT_ALPHA,
                                             ExecutionMode::ForceSYCL),
                    scaled(solution, element_of<T>(K_EXACT_ALPHA)));
            }
        }

        /**
         * @brief Tests that both kernels leave every step of a solve within its residual bound.
         */
        TYPED_TEST(Unit_TriangularParallelReal, TheResidualIsWithinItsBound)
        {
            using T = TypeParam;
            Draws draws(K_SEED);

            for (const TriangularFlags& flags :
                 flags_over<T>({Core::Side::Left, Core::Side::Right}))
            {
                const HostSolve<T> system(flags, K_RESIDUAL_ORDER, K_RESIDUAL_VECTORS,
                                          dominant_matrix<T>(K_RESIDUAL_ORDER, draws));
                const std::vector<T> b = fraction_values<T>(system.elements(), draws);

                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(describe(flags));
                    SCOPED_TRACE(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                   : "ForceReference");
                    this->expect_residual_within_bound(
                        system, b, K_ROUNDED_ALPHA,
                        this->solve_through_view(system, b, K_ROUNDED_ALPHA, mode), mode);
                }
            }
        }

        /**
         * @brief Tests that TRMV stays within its rounding bound of the truth in both modes and
         *        in the work-group shape of the SYCL kernel.
         */
        TYPED_TEST(Unit_TriangularParallelReal, TrmvIsWithinItsBoundOfTheTruth)
        {
            using T = TypeParam;
            Draws draws(K_SEED);
            const std::size_t order = static_cast<std::size_t>(K_PRODUCT_BOUND_ORDER);

            for (const Core::Uplo uplo : {Core::Uplo::Lower, Core::Uplo::Upper})
            {
                for (const Core::Transpose trans :
                     {Core::Transpose::NoTrans, Core::Transpose::Trans})
                {
                    const TriangularFlags flags = plain_flags(uplo, trans);
                    SCOPED_TRACE(describe(flags));

                    const HostSolve<T> system(flags, K_PRODUCT_BOUND_ORDER, 1,
                                              fraction_values<T>(order * order, draws));
                    const std::vector<T> x = fraction_values<T>(order, draws);

                    for (const ExecutionMode mode : K_MODES)
                        this->expect_product_within_bound(
                            system, x, this->multiply_through_view(system, x, 1, mode), mode);

                    this->expect_product_within_bound(
                        system, x, this->multiply_with_kernel(system, x, 1, K_GROUP_WIDTH),
                        ExecutionMode::ForceSYCL);
                }
            }
        }

        /**
         * @brief Runs the triangular kernels on HALF operands, on a device that has fp16.
         */
        class Unit_TriangularParallelHalf : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without fp16. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!test_queue().get_device().has(sycl::aspect::fp16))
                        GTEST_SKIP() << "the device does not report the fp16 aspect";
                }

                /** @brief Returns the row-major unit-lower A of order K_HALF_ORDER, zero above. */
                [[nodiscard]] static std::vector<sycl::half> unit_lower_values()
                {
                    const std::size_t order = static_cast<std::size_t>(K_HALF_ORDER);
                    std::vector<sycl::half> values(order * order, sycl::half(0.0f));
                    for (std::size_t i = 0; i < order; ++i)
                    {
                        for (std::size_t k = 0; k < i; ++k)
                        {
                            const int64_t step = static_cast<int64_t>((i * 7u + k * 13u) % 17u) - 8;
                            values[i * order + k] = sycl::half(static_cast<float>(step) / 64.0f);
                        }
                        values[i * order + i] = sycl::half(1.0f);
                    }
                    return values;
                }

                /** @brief Returns count right-hand-side elements, each a multiple of 1/8. */
                [[nodiscard]] static std::vector<sycl::half> right_hand_side_values(int64_t count)
                {
                    std::vector<sycl::half> values(static_cast<std::size_t>(count));
                    for (std::size_t i = 0; i < values.size(); ++i)
                    {
                        const int64_t step = static_cast<int64_t>((i * 5u) % 23u) - 11;
                        values[i] = sycl::half(static_cast<float>(step) / 8.0f);
                    }
                    return values;
                }

                /**
                 * @brief Returns the host solution of the unit-lower system, summed and carried in
                 *        float, subtracting in the order of DESIGN_BLAS_L3, section 7.2.
                 * @return Each solved element rounded once to HALF.
                 */
                [[nodiscard]] static std::vector<sycl::half> substitute_in_float(
                    Core::Side side, int64_t rows, int64_t cols, const std::vector<sycl::half>& a,
                    const std::vector<sycl::half>& b)
                {
                    std::vector<float> carried(b.size());
                    std::vector<sycl::half> solution(b.size());

                    const auto factor = [&a](int64_t r, int64_t c)
                    {
                        const int64_t index = r * K_HALF_ORDER + c;
                        return static_cast<float>(a[static_cast<std::size_t>(index)]);
                    };
                    const auto at = [cols](int64_t r, int64_t c)
                    {
                        return static_cast<std::size_t>(r * cols + c);
                    };

                    if (side == Core::Side::Left)
                    {
                        for (int64_t j = 0; j < cols; ++j)
                        {
                            for (int64_t i = 0; i < rows; ++i)
                            {
                                float sum = static_cast<float>(b[at(i, j)]);
                                for (int64_t k = 0; k < i; ++k)
                                    sum -= factor(i, k) * carried[at(k, j)];
                                carried[at(i, j)] = sum;
                                solution[at(i, j)] = sycl::half(sum);
                            }
                        }
                    }
                    else
                    {
                        for (int64_t i = 0; i < rows; ++i)
                        {
                            for (int64_t p = cols - 1; p >= 0; --p)
                            {
                                float sum = static_cast<float>(b[at(i, p)]);
                                for (int64_t q = cols - 1; q > p; --q)
                                    sum -= factor(q, p) * carried[at(i, q)];
                                carried[at(i, p)] = sum;
                                solution[at(i, p)] = sycl::half(sum);
                            }
                        }
                    }
                    return solution;
                }

                /** @brief Returns X of the unit-lower system, solved by the kernel at one width. */
                [[nodiscard]] std::vector<sycl::half> solve_with_kernel(
                    Core::Side side, int64_t rows, int64_t cols, const std::vector<sycl::half>& a,
                    const std::vector<sycl::half>& b, int64_t width)
                {
                    Tensor A = device_buffer<sycl::half>(*engine, a);
                    Tensor B = device_buffer<sycl::half>(*engine, b);
                    Tensor work = device_buffer<float>(*engine, std::vector<float>(b.size(), 0.0f));

                    Kernels::SYCL::trsm<sycl::half>(
                        test_queue(), Kernels::SYCL::TriangularSolvePlan{width},
                        Core::Layout::RowMajor, side, Core::Uplo::Lower, Core::Transpose::NoTrans,
                        Core::Diag::Unit, rows, cols, sycl::half(1.0f),
                        pointer_of<const sycl::half>(A), K_HALF_ORDER, 0,
                        pointer_of<sycl::half>(B), cols, rows * cols, 1, pointer_of<float>(work),
                        {})
                        .wait_and_throw();
                    return read_buffer<sycl::half>(B);
                }
        };

        /**
         * @brief Tests that a blocked HALF solve carries its solution in FLOAT32 at every width.
         */
        TEST_F(Unit_TriangularParallelHalf, ABlockedHalfSolveCarriesTheSolutionInFloat)
        {
            const std::vector<sycl::half> a = unit_lower_values();

            for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
            {
                SCOPED_TRACE(Core::to_string(side));
                const bool left = (side == Core::Side::Left);
                const int64_t rows = left ? K_HALF_ORDER : K_HALF_VECTORS;
                const int64_t cols = left ? K_HALF_VECTORS : K_HALF_ORDER;
                const std::vector<sycl::half> b = right_hand_side_values(rows * cols);
                const std::vector<sycl::half> expected =
                    substitute_in_float(side, rows, cols, a, b);

                for (const int64_t width : K_HALF_WIDTHS)
                {
                    SCOPED_TRACE(::testing::Message() << "block width " << width);
                    expect_same_bits(solve_with_kernel(side, rows, cols, a, b, width), expected);
                }
            }
        }

        /**
         * @brief Tests that a blocked HALF TPSV carries its solution in FLOAT32 at every width.
         */
        TEST_F(Unit_TriangularParallelHalf, ABlockedHalfVectorSolveCarriesTheSolutionInFloat)
        {
            TriangularFlags flags;
            flags.diag = Core::Diag::Unit;

            const std::vector<sycl::half> a = unit_lower_values();
            const std::vector<sycl::half> b = right_hand_side_values(K_HALF_ORDER);
            const std::vector<sycl::half> expected =
                substitute_in_float(Core::Side::Left, K_HALF_ORDER, 1, a, b);

            for (const int64_t width : K_HALF_WIDTHS)
            {
                SCOPED_TRACE(::testing::Message() << "block width " << width);
                Tensor AP = device_buffer<sycl::half>(*engine,
                                                      packed_storage(flags, K_HALF_ORDER, a));
                Tensor x = device_buffer<sycl::half>(*engine, b);
                Tensor work = device_buffer<float>(*engine, std::vector<float>(b.size(), 0.0f));

                Kernels::SYCL::tpsv<sycl::half>(
                    test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                    flags.uplo, flags.trans, flags.diag, K_HALF_ORDER,
                    pointer_of<const sycl::half>(AP), pointer_of<sycl::half>(x), 1,
                    pointer_of<float>(work), {})
                    .wait_and_throw();
                expect_same_bits(read_buffer<sycl::half>(x), expected);
            }
        }

        /**
         * @brief Tests that a HALF TRMV sums a row in FLOAT32, where a HALF sum loses the 1.
         */
        TEST_F(Unit_TriangularParallelHalf, HalfTrmvSumsInFloat)
        {
            const sycl::half nan(std::numeric_limits<float>::quiet_NaN());
            const sycl::half one(1.0f);
            const std::vector<sycl::half> a = {one, nan, nan,
                                               one, one, nan,
                                               sycl::half(2048.0f), one, sycl::half(-2048.0f)};

            for (const ExecutionMode mode : K_MODES)
            {
                SCOPED_TRACE(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                               : "ForceReference");
                engine->set_mode(mode);

                Tensor a_buffer = device_buffer<sycl::half>(*engine, a);
                Tensor x = device_buffer<sycl::half>(*engine,
                                                     std::vector<sycl::half>(3, one));
                Tensor A = matrix_view(a_buffer, {3, 3}, Core::Layout::RowMajor);

                SushiBLAS::BLASOps(*engine).trmv(A, x, Core::Uplo::Lower);
                engine->execute().value();

                const std::vector<sycl::half> product = read_buffer<sycl::half>(x);
                ASSERT_EQ(std::size_t{3}, product.size());
                EXPECT_EQ(1.0f, static_cast<float>(product[2]));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
