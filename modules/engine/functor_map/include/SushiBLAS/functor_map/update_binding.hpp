/**************************************************************************/
/* update_binding.hpp                                                     */
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
 * @file update_binding.hpp
 * @brief Declares the binding that declares the tensors an element functor reads and writes in place.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
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
#include <SushiBLAS/functor_map/write_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Names a tensor a functor reads and writes as an array of the operation's own
         *        element type.
         * @see modules/operations/MATH_README.md
         */
        struct ElementUpdate
        {
            /** @brief The tensor updated; it must outlive the bind_updates call. */
            Tensor& tensor;
        };

        /**
         * @brief Names a tensor a functor reads and writes as an array of U, whatever the
         *        operation's element type.
         * @tparam U One of the eight element types.
         * @see modules/operations/MATH_README.md
         */
        template<typename U>
        struct TypedUpdate
        {
            /** @brief The tensor updated; it must outlive the bind_updates call. */
            Tensor& tensor;
        };

        namespace Detail
        {
            /** @brief Hands an element update as `T*`. */
            template<typename T>
            struct WritePointer<ElementUpdate, T>
            {
                /** @brief The pointer type. */
                using type = T*;
            };

            /** @brief Hands a typed update as `U*`. */
            template<typename U, typename T>
            struct WritePointer<TypedUpdate<U>, T>
            {
                /** @brief The pointer type. */
                using type = U*;
            };

            /** @brief Returns the dtype an element update's tensor must hold: the operation's own. */
            [[nodiscard]] constexpr Core::DataType update_dtype(const ElementUpdate*,
                                                                Core::DataType element) noexcept
            {
                return element;
            }

            /** @brief Returns the dtype a typed update's tensor must hold: the one U names. */
            template<typename U>
            [[nodiscard]] constexpr Core::DataType update_dtype(const TypedUpdate<U>*,
                                                                Core::DataType) noexcept
            {
                return dtype_named_by<U>();
            }

            /** @brief Reports false: a typed update does not name the operation's element type. */
            template<typename Update>
            struct IsElementUpdate : std::false_type
            {

            };

            /** @brief Reports true: an element update names the operation's element type. */
            template<>
            struct IsElementUpdate<ElementUpdate> : std::true_type
            {

            };

            /** @brief Returns a functor that declares no reads, as the innermost callable. */
            template<typename Op>
            [[nodiscard]] inline const Op& innermost_functor(const Op& op) noexcept
            {
                return op;
            }

            /** @brief Returns the functor a bind_reads bundle holds, as the innermost callable. */
            template<typename Fn, typename... Reads>
            [[nodiscard]] inline const Fn& innermost_functor(
                const ReadBound<Fn, Reads...>& bound) noexcept
            {
                return bound.functor();
            }

            /** @brief Returns wrapped unchanged, because a functor that declares no reads adds none. */
            template<typename Op, typename Wrapped>
            [[nodiscard]] inline Wrapped wrap_reads(const Op&, Wrapped wrapped)
            {
                return wrapped;
            }

            /** @brief Returns wrapped with the read addresses of a bind_reads bundle around it. */
            template<typename Fn, typename... Reads, typename Wrapped>
            [[nodiscard]] inline auto wrap_reads(const ReadBound<Fn, Reads...>& bound,
                                                 Wrapped wrapped)
            {
                return bound.addressed_around(std::move(wrapped));
            }
        } // namespace Detail

        /**
         * @brief Declares that a functor reads and writes tensor as an array of the operation's
         *        element type.
         * @see modules/operations/MATH_README.md
         */
        [[nodiscard]] inline ElementUpdate update_as_element(Tensor& tensor) noexcept
        {
            return ElementUpdate{tensor};
        }

        /** @brief Refuses a temporary: an update names a tensor that outlives the bind_updates call. */
        ElementUpdate update_as_element(Tensor&&) = delete;

        /**
         * @brief Declares that a functor reads and writes tensor as an array of U.
         * @see modules/operations/MATH_README.md
         */
        template<typename U>
        [[nodiscard]] inline TypedUpdate<U> update_as(Tensor& tensor) noexcept
        {
            return TypedUpdate<U>{tensor};
        }

        /** @brief Refuses a temporary: an update names a tensor that outlives the bind_updates call. */
        template<typename U>
        TypedUpdate<U> update_as(Tensor&&) = delete;

        /**
         * @brief Holds an element functor, with or without declared reads, and the tensors it
         *        reads and writes in place at execution time; it never leaves the record call.
         * @tparam Inner A bare functor, a bind_scalars bundle or a bind_reads bundle.
         * @see modules/operations/MATH_README.md
         */
        template<typename Inner, typename... Updates>
        class UpdateBound final
        {
            public:
                /** @brief Stores the functor and one pointer to each declared tensor. */
                explicit UpdateBound(Inner inner, const Updates&... updates)
                    : inner_(std::move(inner)), tensors_{&updates.tensor...}
                {

                }

                /** @brief Returns the functor as bound, with the reads it declares. */
                [[nodiscard]] const Inner& inner() const noexcept
                {
                    return inner_;
                }

                /**
                 * @brief Checks every declared update as an operand of one operation.
                 * @param element The dtype of the operation's elements.
                 * @throws std::invalid_argument for an update that is empty, has no storage, holds
                 *         another dtype than it was declared with, or is not contiguous.
                 * @throws UnsupportedExecution for an update whose device cannot compute in its dtype.
                 */
                void require(Core::DataType element, const char* name) const
                {
                    const std::array<Core::DataType, sizeof...(Updates)> expected{
                        Detail::update_dtype(static_cast<const Updates*>(nullptr), element)...};
                    Detail::require_bound_tensors("update", tensor_at(), expected, name);
                }

                /**
                 * @brief Declares every update in operands as read and written, after the task's
                 *        reads and before its writes.
                 * @throws std::invalid_argument for an update that overlaps any region the task
                 *         reads or writes, or another update of this bundle.
                 */
                void declare_into(OperandSet& operands, const char* name) const
                {
                    const auto refuse_any = [](const SushiRuntime::BufferRegion&,
                                               const SushiRuntime::BufferRegion&)
                    {
                        return false;
                    };
                    Detail::require_bound_disjoint<sizeof...(Updates)>(
                        "update", tensor_at(), operands.read_regions(), refuse_any,
                        "overlaps a tensor the operation reads", name);
                    Detail::require_bound_disjoint<sizeof...(Updates)>(
                        "update", tensor_at(), operands.write_regions(), refuse_any,
                        "overlaps a tensor the operation writes", name);
                    require_mutually_disjoint(name);

                    for (const Tensor* tensor : tensors_)
                        operands.updates(*tensor);
                }

                /**
                 * @brief Returns the dtype of the first element binding, the wrapped functor's
                 *        reads before this bundle's updates, or nothing when none is declared.
                 */
                [[nodiscard]] std::optional<Core::DataType> element_dtype() const noexcept
                {
                    if (const std::optional<Core::DataType> inner = Detail::element_dtype_of(inner_))
                        return inner;

                    constexpr std::array<bool, sizeof...(Updates)> is_element{
                        Detail::IsElementUpdate<Updates>::value...};

                    for (std::size_t k = 0; k < sizeof...(Updates); ++k)
                    {
                        if (is_element[k])
                            return tensors_[k]->dtype;
                    }

                    return std::nullopt;
                }

                /** @brief Returns the functor with the base addresses of its updates and its reads. */
                [[nodiscard]] auto addressed() const
                {
                    return addressed_around(Detail::innermost_functor(inner_));
                }

                /**
                 * @brief Returns wrapped with the updates' base addresses, inside the reads'.
                 * @tparam Wrapped A functor that calls the innermost functor, as bind_writes builds.
                 */
                template<typename Wrapped>
                [[nodiscard]] auto addressed_around(Wrapped wrapped) const
                {
                    std::array<void*, sizeof...(Updates)> addresses{};
                    for (std::size_t k = 0; k < sizeof...(Updates); ++k)
                        addresses[k] = tensors_[k]->data_or_null();

                    return Detail::wrap_reads(
                        inner_, WriteAddressedOp<Wrapped, Updates...>(std::move(wrapped), addresses));
                }

            private:
                /** @brief Returns a callable that hands the declared tensor at a position. */
                [[nodiscard]] auto tensor_at() const noexcept
                {
                    return [this](std::size_t k) -> const Tensor& { return *tensors_[k]; };
                }

                /**
                 * @brief Refuses two updates of this bundle whose regions overlap.
                 * @throws std::invalid_argument naming the later of the two.
                 */
                void require_mutually_disjoint(const char* name) const
                {
                    for (std::size_t k = 1; k < sizeof...(Updates); ++k)
                    {
                        const std::string operand = "update " + std::to_string(k);
                        const SushiRuntime::BufferRegion region = tensors_[k]->region();
                        for (std::size_t j = 0; j < k; ++j)
                            require_operand(!region.overlaps(tensors_[j]->region()), name,
                                            operand.c_str(), "{}",
                                            "overlaps another tensor the operation updates");
                    }
                }

                /// @brief The element functor, or a bind_scalars or bind_reads bundle of one.
                Inner inner_;

                /// @brief The declared tensors, in bound order; the caller owns them.
                std::array<Tensor*, sizeof...(Updates)> tensors_;
        };

        /**
         * @brief Binds the tensors an element functor reads and writes in place at execution time.
         * @param fn Callable `(elements..., read pointers..., update pointers...)`, or a bind_reads
         *        or bind_scalars bundle of one. Writes bind outside:
         *        bind_writes(bind_updates(bind_reads(fn, reads...), updates...), writes...).
         * @param updates One update_as_element or update_as<U> per tensor, in the order fn takes them.
         * @see modules/operations/MATH_README.md
         */
        template<typename Fn, typename... Updates>
        [[nodiscard]] inline UpdateBound<Fn, Updates...> bind_updates(Fn fn,
                                                                      const Updates&... updates)
        {
            static_assert(sizeof...(Updates) >= 1, "bind_updates needs at least one declared update.");
            static_assert(!Detail::IsWriteBound<Fn>::value,
                          "bind_updates wraps a functor or a bind_reads bundle; bind_writes wraps "
                          "bind_updates, not the other way round.");

            return UpdateBound<Fn, Updates...>(std::move(fn), updates...);
        }

        namespace Detail
        {
            /** @brief Reports true: an update bundle declares writes, as a bind_writes bundle does. */
            template<typename Inner, typename... Updates>
            struct IsWriteBound<UpdateBound<Inner, Updates...>> : std::true_type
            {

            };

            /** @brief Checks nothing, because a bare functor declares no updates. */
            template<typename Op>
            inline void require_declared_updates(const Op&, Core::DataType, const char*) noexcept
            {

            }

            /** @brief Checks the updates a bundle declares; see UpdateBound::require. */
            template<typename Inner, typename... Updates>
            inline void require_declared_updates(const UpdateBound<Inner, Updates...>& bound,
                                                 Core::DataType element, const char* name)
            {
                bound.require(element, name);
            }

            /** @brief Checks the updates of the functor a write bundle wraps. */
            template<typename Inner, typename... Writes>
            inline void require_declared_updates(const WriteBound<Inner, Writes...>& bound,
                                                 Core::DataType element, const char* name)
            {
                require_declared_updates(bound.inner(), element, name);
            }

            /** @brief Adds nothing, because a bare functor declares no updates. */
            template<typename Op>
            inline void declare_updates(const Op&, OperandSet&, const char*) noexcept
            {

            }

            /**
             * @brief Adds a bundle's updates to a task's operand set; see UpdateBound::declare_into.
             * @pre The bundle's reads were declared first, with declare_reads.
             */
            template<typename Inner, typename... Updates>
            inline void declare_updates(const UpdateBound<Inner, Updates...>& bound,
                                        OperandSet& operands, const char* name)
            {
                bound.declare_into(operands, name);
            }

            /** @brief Adds the updates of the functor a write bundle wraps. */
            template<typename Inner, typename... Writes>
            inline void declare_updates(const WriteBound<Inner, Writes...>& bound,
                                        OperandSet& operands, const char* name)
            {
                declare_updates(bound.inner(), operands, name);
            }

            /** @brief Returns the dtype of the first element binding an update bundle declares, if any. */
            template<typename Inner, typename... Updates>
            [[nodiscard]] inline std::optional<Core::DataType> element_dtype_of(
                const UpdateBound<Inner, Updates...>& bound) noexcept
            {
                return bound.element_dtype();
            }

            /** @brief Checks the reads of the functor an update bundle wraps; see ReadBound::require. */
            template<typename Inner, typename... Updates>
            inline void require_declared_reads(const UpdateBound<Inner, Updates...>& bound,
                                               Core::DataType element, const char* name)
            {
                require_declared_reads(bound.inner(), element, name);
            }

            /** @brief Adds the reads of the functor an update bundle wraps; see ReadBound::declare_into. */
            template<typename Inner, typename... Updates>
            inline void declare_reads(const UpdateBound<Inner, Updates...>& bound,
                                      OperandSet& operands, const char* name)
            {
                declare_reads(bound.inner(), operands, name);
            }

            /** @brief Returns make(fn) inside the bundle's update and read addresses. */
            template<typename Inner, typename... Updates, typename Make>
            [[nodiscard]] inline auto address_around(const UpdateBound<Inner, Updates...>& bound,
                                                     const Make& make)
            {
                return bound.addressed_around(make(innermost_functor(bound.inner())));
            }

            /** @brief Returns an update bundle's functor with its read and update addresses. */
            template<typename Inner, typename... Updates>
            [[nodiscard]] inline auto task_functor(const UpdateBound<Inner, Updates...>& bound)
            {
                return bound.addressed();
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
