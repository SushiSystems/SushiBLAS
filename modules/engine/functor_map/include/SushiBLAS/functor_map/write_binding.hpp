/**************************************************************************/
/* write_binding.hpp                                                      */
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
 * @file write_binding.hpp
 * @brief Declares the binding that declares the tensors an element functor writes and hands them to it as typed pointers.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <tuple>
#include <string>
#include <cstddef>
#include <optional>
#include <utility>
#include <type_traits>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Names a tensor a functor writes as an array of the operation's own element type.
         * @see modules/operations/MATH_README.md
         */
        struct ElementWrite
        {
            /** @brief The tensor written; it must outlive the bind_writes call. */
            Tensor& tensor;
        };

        /**
         * @brief Names a tensor a functor writes as an array of U, whatever the operation's element type.
         * @tparam U One of the seven element types.
         * @see modules/operations/MATH_README.md
         */
        template<typename U>
        struct TypedWrite
        {
            /** @brief The tensor written; it must outlive the bind_writes call. */
            Tensor& tensor;
        };

        namespace Detail
        {
            /** @brief Names the pointer a declared write hands a functor whose elements are T. */
            template<typename Write, typename T>
            struct WritePointer;

            /** @brief Hands an element write as `T*`. */
            template<typename T>
            struct WritePointer<ElementWrite, T>
            {
                /** @brief The pointer type. */
                using type = T*;
            };

            /** @brief Hands a typed write as `U*`. */
            template<typename U, typename T>
            struct WritePointer<TypedWrite<U>, T>
            {
                /** @brief The pointer type. */
                using type = U*;
            };

            /** @brief Returns the dtype an element write's tensor must hold: the operation's own. */
            [[nodiscard]] constexpr Core::DataType write_dtype(const ElementWrite*,
                                                               Core::DataType element) noexcept
            {
                return element;
            }

            /** @brief Returns the dtype a typed write's tensor must hold: the one U names. */
            template<typename U>
            [[nodiscard]] constexpr Core::DataType write_dtype(const TypedWrite<U>*,
                                                               Core::DataType) noexcept
            {
                return dtype_named_by<U>();
            }

            /** @brief Reports false: a typed write does not name the operation's element type. */
            template<typename Write>
            struct IsElementWrite : std::false_type
            {

            };

            /** @brief Reports true: an element write names the operation's element type. */
            template<>
            struct IsElementWrite<ElementWrite> : std::true_type
            {

            };

            /** @brief Reports whether two regions name the same bytes of the same allocation. */
            [[nodiscard]] inline bool same_region(const SushiRuntime::BufferRegion& a,
                                                  const SushiRuntime::BufferRegion& b) noexcept
            {
                return a.base == b.base && a.offset == b.offset && a.length == b.length;
            }
        } // namespace Detail

        /**
         * @brief Declares that a functor writes tensor as an array of the operation's element type.
         * @see modules/operations/MATH_README.md
         */
        [[nodiscard]] inline ElementWrite write_as_element(Tensor& tensor) noexcept
        {
            return ElementWrite{tensor};
        }

        /** @brief Refuses a temporary: a write names a tensor that outlives the bind_writes call. */
        ElementWrite write_as_element(Tensor&&) = delete;

        /**
         * @brief Declares that a functor writes tensor as an array of U.
         * @see modules/operations/MATH_README.md
         */
        template<typename U>
        [[nodiscard]] inline TypedWrite<U> write_as(Tensor& tensor) noexcept
        {
            return TypedWrite<U>{tensor};
        }

        /** @brief Refuses a temporary: a write names a tensor that outlives the bind_writes call. */
        template<typename U>
        TypedWrite<U> write_as(Tensor&&) = delete;

        /**
         * @brief Holds an element functor and the typed addresses of its declared writes; this is
         *        what a kernel captures.
         * @tparam Fn Callable `(arguments..., write pointers...)`, already narrowed for T.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename T, typename... Writes>
        class WritingOp final
        {
            public:
                /** @brief Stores the functor and the base address of each declared write. */
                WritingOp(Fn fn, const std::array<void*, sizeof...(Writes)>& addresses)
                    : fn_(std::move(fn)), addresses_(addresses)
                {

                }

                /** @brief Calls the functor with the arguments, then one pointer per write in bound order. */
                template<typename... Arguments>
                decltype(auto) operator()(Arguments&&... arguments) const
                {
                    return call(std::index_sequence_for<Writes...>{},
                                std::forward<Arguments>(arguments)...);
                }

            private:
                /** @brief Names the pointer type of write K. */
                template<std::size_t K>
                using Pointer = typename Detail::WritePointer<
                    std::tuple_element_t<K, std::tuple<Writes...>>, T>::type;

                /** @brief Expands the write pointers after the arguments. */
                template<std::size_t... K, typename... Arguments>
                decltype(auto) call(std::index_sequence<K...>, Arguments&&... arguments) const
                {
                    return fn_(std::forward<Arguments>(arguments)...,
                               static_cast<Pointer<K>>(addresses_[K])...);
                }

                /// @brief The element functor.
                Fn fn_;

                /// @brief The base address of each declared write, in bound order.
                std::array<void*, sizeof...(Writes)> addresses_;
        };

        /**
         * @brief Holds an element functor and the addresses of its declared writes before the
         *        element type is known; this is what a task closure holds, or what an
         *        AddressedOp holds in place of the functor when the bundle also reads.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename... Writes>
        class WriteAddressedOp final
        {
            public:
                /** @brief Stores the functor and the base address of each declared write. */
                WriteAddressedOp(Fn fn, const std::array<void*, sizeof...(Writes)>& addresses)
                    : fn_(std::move(fn)), addresses_(addresses)
                {

                }

                /**
                 * @brief Returns the kernel functor for element type T, its scalars narrowed first.
                 * @tparam T The element type of the with_element_type arm.
                 */
                template<typename T>
                [[nodiscard]] WritingOp<std::decay_t<decltype(Detail::narrow_for(
                                            ElementTag<T>{}, std::declval<const Fn&>()))>,
                                        T, Writes...>
                narrow() const
                {
                    using Narrowed = std::decay_t<decltype(
                        Detail::narrow_for(ElementTag<T>{}, fn_))>;
                    return WritingOp<Narrowed, T, Writes...>(
                        Detail::narrow_for(ElementTag<T>{}, fn_), addresses_);
                }

            private:
                /// @brief The element functor, or a bind_scalars bundle of one.
                Fn fn_;

                /// @brief The base address of each declared write, in bound order.
                std::array<void*, sizeof...(Writes)> addresses_;
        };

        /**
         * @brief Holds an element functor and the tensors it reads and writes in place;
         *        update_binding.hpp defines it.
         * @see modules/operations/MATH_README.md
         */
        template<typename Inner, typename... Updates>
        class UpdateBound;

        namespace Detail
        {
            /** @brief Returns an update bundle's element dtype; defined in update_binding.hpp. */
            template<typename Inner, typename... Updates>
            [[nodiscard]] inline std::optional<Core::DataType> element_dtype_of(
                const UpdateBound<Inner, Updates...>& bound) noexcept;

            /** @brief Checks the reads an update bundle wraps; defined in update_binding.hpp. */
            template<typename Inner, typename... Updates>
            inline void require_declared_reads(const UpdateBound<Inner, Updates...>& bound,
                                               Core::DataType element, const char* name);

            /** @brief Adds the reads an update bundle wraps; defined in update_binding.hpp. */
            template<typename Inner, typename... Updates>
            inline void declare_reads(const UpdateBound<Inner, Updates...>& bound,
                                      OperandSet& operands, const char* name);

            /**
             * @brief Returns the bundle's update and read addresses around make(fn); defined in
             *        update_binding.hpp.
             */
            template<typename Inner, typename... Updates, typename Make>
            [[nodiscard]] inline auto address_around(const UpdateBound<Inner, Updates...>& bound,
                                                     const Make& make);

            /**
             * @brief Returns make(op) for a functor that declares no reads.
             * @param make Callable that wraps a functor into a WriteAddressedOp.
             */
            template<typename Op, typename Make>
            [[nodiscard]] inline auto address_around(const Op& op, const Make& make)
            {
                return make(op);
            }

            /**
             * @brief Returns the bundle's read addresses around make(fn), so the reads come before
             *        the writes in the call.
             */
            template<typename Fn, typename... Reads, typename Make>
            [[nodiscard]] inline auto address_around(const ReadBound<Fn, Reads...>& bound,
                                                     const Make& make)
            {
                return bound.addressed_around(make(bound.functor()));
            }
        } // namespace Detail

        /**
         * @brief Holds an element functor, with or without declared reads, and the tensors it
         *        writes at execution time; it never leaves the record call.
         * @tparam Inner A bare functor, a bind_scalars bundle or a bind_reads bundle.
         * @see modules/operations/MATH_README.md
         */
        template<typename Inner, typename... Writes>
        class WriteBound final
        {
            public:
                /** @brief Stores the functor and one pointer to each declared tensor. */
                explicit WriteBound(Inner inner, const Writes&... writes)
                    : inner_(std::move(inner)), tensors_{&writes.tensor...}
                {

                }

                /** @brief Returns the functor as bound, with the reads it declares. */
                [[nodiscard]] const Inner& inner() const noexcept
                {
                    return inner_;
                }

                /**
                 * @brief Checks every declared write as an operand of one operation.
                 * @param element The dtype of the operation's elements.
                 * @throws std::invalid_argument for a write that is empty, has no storage, holds
                 *         another dtype than it was declared with, or is not contiguous.
                 * @throws UnsupportedExecution for a write whose device cannot compute in its dtype.
                 */
                void require(Core::DataType element, const char* name) const
                {
                    const std::array<Core::DataType, sizeof...(Writes)> expected{
                        Detail::write_dtype(static_cast<const Writes*>(nullptr), element)...};
                    Detail::require_bound_tensors("write", tensor_at(), expected, name);
                }

                /**
                 * @brief Declares every write in operands, after the task's reads and updates were
                 *        declared. A tensor already read keeps its read and gains one write; a
                 *        tensor named twice is declared once.
                 * @throws std::invalid_argument for a write that overlaps a region the task reads
                 *         without being that region, or any region the task updates.
                 */
                void declare_into(OperandSet& operands, const char* name) const
                {
                    Detail::require_bound_disjoint<sizeof...(Writes)>(
                        "write", tensor_at(), operands.write_regions(),
                        [](const SushiRuntime::BufferRegion&, const SushiRuntime::BufferRegion&)
                        {
                            return false;
                        },
                        "overlaps a tensor the operation updates", name);
                    Detail::require_bound_disjoint<sizeof...(Writes)>(
                        "write", tensor_at(), operands.read_regions(),
                        [](const SushiRuntime::BufferRegion& own,
                           const SushiRuntime::BufferRegion& read)
                        {
                            return Detail::same_region(own, read);
                        },
                        "overlaps a tensor the operation reads through another view", name);

                    for (std::size_t k = 0; k < sizeof...(Writes); ++k)
                    {
                        if (!is_declared_write(operands, tensors_[k]->region()))
                            operands.writes(*tensors_[k]);
                    }
                }

                /**
                 * @brief Returns the dtype of the first element binding, the wrapped functor's
                 *        reads before this bundle's writes, or nothing when none is declared.
                 */
                [[nodiscard]] std::optional<Core::DataType> element_dtype() const noexcept
                {
                    if (const std::optional<Core::DataType> inner = Detail::element_dtype_of(inner_))
                        return inner;

                    constexpr std::array<bool, sizeof...(Writes)> is_element{
                        Detail::IsElementWrite<Writes>::value...};

                    for (std::size_t k = 0; k < sizeof...(Writes); ++k)
                    {
                        if (is_element[k])
                            return tensors_[k]->dtype;
                    }

                    return std::nullopt;
                }

                /** @brief Returns the functor with the base addresses of its writes and its reads. */
                [[nodiscard]] auto addressed() const
                {
                    std::array<void*, sizeof...(Writes)> addresses{};
                    for (std::size_t k = 0; k < sizeof...(Writes); ++k)
                        addresses[k] = tensors_[k]->data_or_null();

                    return Detail::address_around(
                        inner_,
                        [&addresses](const auto& fn)
                        {
                            return WriteAddressedOp<std::decay_t<decltype(fn)>, Writes...>(
                                fn, addresses);
                        });
                }

            private:
                /** @brief Returns a callable that hands the declared tensor at a position. */
                [[nodiscard]] auto tensor_at() const noexcept
                {
                    return [this](std::size_t k) -> const Tensor& { return *tensors_[k]; };
                }

                /** @brief Reports whether operands already holds a write of exactly region. */
                [[nodiscard]] static bool is_declared_write(const OperandSet& operands,
                                                            const SushiRuntime::BufferRegion& region)
                {
                    for (const SushiRuntime::BufferRegion& written : operands.write_regions())
                    {
                        if (Detail::same_region(region, written))
                            return true;
                    }
                    return false;
                }

                /// @brief The element functor, or a bundle of one.
                Inner inner_;

                /// @brief The declared tensors, in bound order; the caller owns them.
                std::array<Tensor*, sizeof...(Writes)> tensors_;
        };

        /**
         * @brief Binds the tensors an element functor writes at execution time, for a seam to declare.
         * @param fn Callable `(elements..., read pointers..., write pointers...)`, or a bind_reads
         *        or bind_scalars bundle of one; its scalars come last. Reads bind inside:
         *        bind_writes(bind_reads(fn, reads...), writes...).
         * @param writes One write_as_element or write_as<U> per tensor, in the order fn takes them.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename... Writes>
        [[nodiscard]] inline WriteBound<Fn, Writes...> bind_writes(Fn fn, const Writes&... writes)
        {
            static_assert(sizeof...(Writes) >= 1, "bind_writes needs at least one declared write.");

            return WriteBound<Fn, Writes...>(std::move(fn), writes...);
        }

        namespace Detail
        {
            /** @brief Checks nothing, because a bare functor declares no writes. */
            template<typename Op>
            inline void require_declared_writes(const Op&, Core::DataType, const char*) noexcept
            {

            }

            /** @brief Checks the writes a bundle declares; see WriteBound::require. */
            template<typename Inner, typename... Writes>
            inline void require_declared_writes(const WriteBound<Inner, Writes...>& bound,
                                                Core::DataType element, const char* name)
            {
                bound.require(element, name);
            }

            /** @brief Adds nothing, because a bare functor declares no writes. */
            template<typename Op>
            inline void declare_writes(const Op&, OperandSet&, const char*) noexcept
            {

            }

            /**
             * @brief Adds a bundle's writes to a task's operand set; see WriteBound::declare_into.
             * @pre The bundle's reads were declared first, with declare_reads.
             */
            template<typename Inner, typename... Writes>
            inline void declare_writes(const WriteBound<Inner, Writes...>& bound,
                                       OperandSet& operands, const char* name)
            {
                bound.declare_into(operands, name);
            }

            /** @brief Returns the dtype of the first element binding a write bundle declares, if any. */
            template<typename Inner, typename... Writes>
            [[nodiscard]] inline std::optional<Core::DataType> element_dtype_of(
                const WriteBound<Inner, Writes...>& bound) noexcept
            {
                return bound.element_dtype();
            }

            /** @brief Checks the reads of the functor a write bundle wraps; see ReadBound::require. */
            template<typename Inner, typename... Writes>
            inline void require_declared_reads(const WriteBound<Inner, Writes...>& bound,
                                               Core::DataType element, const char* name)
            {
                require_declared_reads(bound.inner(), element, name);
            }

            /** @brief Adds the reads of the functor a write bundle wraps; see ReadBound::declare_into. */
            template<typename Inner, typename... Writes>
            inline void declare_reads(const WriteBound<Inner, Writes...>& bound,
                                      OperandSet& operands, const char* name)
            {
                declare_reads(bound.inner(), operands, name);
            }

            /** @brief Returns a write bundle's functor with its read and write addresses, for the task closure. */
            template<typename Inner, typename... Writes>
            [[nodiscard]] inline auto task_functor(const WriteBound<Inner, Writes...>& bound)
            {
                return bound.addressed();
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
