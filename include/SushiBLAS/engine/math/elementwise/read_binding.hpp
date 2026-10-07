/**************************************************************************/
/* read_binding.hpp                                                       */
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
 * @file read_binding.hpp
 * @brief Declares the binding that declares the tensors an element functor reads and hands them to it as typed pointers.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <tuple>
#include <string>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/math/elementwise/scalar_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Names a tensor a functor reads as an array of the operation's own element type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        struct ElementRead
        {
            /** @brief The tensor read; it must outlive the bind_reads call. */
            const Tensor& tensor;
        };

        /**
         * @brief Names a tensor a functor reads as an array of U, whatever the operation's element type.
         * @tparam U One of the seven element types.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename U>
        struct TypedRead
        {
            /** @brief The tensor read; it must outlive the bind_reads call. */
            const Tensor& tensor;
        };

        namespace Detail
        {
            /** @brief Returns the dtype whose element type is U. */
            template<typename U>
            [[nodiscard]] constexpr Core::DataType dtype_named_by() noexcept
            {
                if constexpr (std::is_same_v<U, sycl::half>)
                    return Core::DataType::HALF;
                else if constexpr (std::is_same_v<U, float>)
                    return Core::DataType::FLOAT32;
                else if constexpr (std::is_same_v<U, double>)
                    return Core::DataType::FLOAT64;
                else if constexpr (std::is_same_v<U, std::complex<float>>)
                    return Core::DataType::COMPLEX32;
                else if constexpr (std::is_same_v<U, std::complex<double>>)
                    return Core::DataType::COMPLEX64;
                else if constexpr (std::is_same_v<U, std::int32_t>)
                    return Core::DataType::INT32;
                else
                {
                    static_assert(std::is_same_v<U, std::int64_t>,
                                  "A typed read names one of the seven element types.");
                    return Core::DataType::INT64;
                }
            }

            /** @brief Names the pointer a declared read hands a functor whose elements are T. */
            template<typename Read, typename T>
            struct ReadPointer;

            /** @brief Hands an element read as `const T*`. */
            template<typename T>
            struct ReadPointer<ElementRead, T>
            {
                /** @brief The pointer type. */
                using type = const T*;
            };

            /** @brief Hands a typed read as `const U*`. */
            template<typename U, typename T>
            struct ReadPointer<TypedRead<U>, T>
            {
                /** @brief The pointer type. */
                using type = const U*;
            };

            /** @brief Returns the dtype an element read's tensor must hold: the operation's own. */
            [[nodiscard]] constexpr Core::DataType read_dtype(const ElementRead*,
                                                              Core::DataType element) noexcept
            {
                return element;
            }

            /** @brief Returns the dtype a typed read's tensor must hold: the one U names. */
            template<typename U>
            [[nodiscard]] constexpr Core::DataType read_dtype(const TypedRead<U>*,
                                                              Core::DataType) noexcept
            {
                return dtype_named_by<U>();
            }
        } // namespace Detail

        /**
         * @brief Declares that a functor reads tensor as an array of the operation's element type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        [[nodiscard]] inline ElementRead read_as_element(const Tensor& tensor) noexcept
        {
            return ElementRead{tensor};
        }

        /**
         * @brief Declares that a functor reads tensor as an array of U.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename U>
        [[nodiscard]] inline TypedRead<U> read_as(const Tensor& tensor) noexcept
        {
            return TypedRead<U>{tensor};
        }

        /**
         * @brief Holds an element functor and the typed addresses of its declared reads; this is
         *        what a kernel captures.
         * @tparam Fn Callable `(elements..., read pointers...)`, already narrowed for T.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Fn, typename T, typename... Reads>
        class ReadingOp final
        {
            public:
                /** @brief Stores the functor and the base address of each declared read. */
                ReadingOp(Fn fn, const std::array<const void*, sizeof...(Reads)>& addresses)
                    : fn_(std::move(fn)), addresses_(addresses)
                {

                }

                /** @brief Calls the functor with the elements, then one pointer per read in bound order. */
                template<typename... Elements>
                decltype(auto) operator()(Elements&&... elements) const
                {
                    return call(std::index_sequence_for<Reads...>{},
                                std::forward<Elements>(elements)...);
                }

            private:
                /** @brief Names the pointer type of read K. */
                template<std::size_t K>
                using Pointer = typename Detail::ReadPointer<
                    std::tuple_element_t<K, std::tuple<Reads...>>, T>::type;

                /** @brief Expands the read pointers after the elements. */
                template<std::size_t... K, typename... Elements>
                decltype(auto) call(std::index_sequence<K...>, Elements&&... elements) const
                {
                    return fn_(std::forward<Elements>(elements)...,
                               static_cast<Pointer<K>>(addresses_[K])...);
                }

                /// @brief The element functor.
                Fn fn_;

                /// @brief The base address of each declared read, in bound order.
                std::array<const void*, sizeof...(Reads)> addresses_;
        };

        /**
         * @brief Holds an element functor and the addresses of its declared reads before the
         *        element type is known; this is what a task closure holds.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Fn, typename... Reads>
        class AddressedOp final
        {
            public:
                /** @brief Stores the functor and the base address of each declared read. */
                AddressedOp(Fn fn, const std::array<const void*, sizeof...(Reads)>& addresses)
                    : fn_(std::move(fn)), addresses_(addresses)
                {

                }

                /**
                 * @brief Returns the kernel functor for element type T, its scalars narrowed first.
                 * @tparam T The element type of the with_element_type arm.
                 */
                template<typename T>
                [[nodiscard]] auto typed(ElementTag<T> tag) const
                {
                    using Narrowed = std::decay_t<decltype(Detail::narrow_for(tag, fn_))>;
                    return ReadingOp<Narrowed, T, Reads...>(Detail::narrow_for(tag, fn_),
                                                            addresses_);
                }

            private:
                /// @brief The element functor, or a bind_scalars bundle of one.
                Fn fn_;

                /// @brief The base address of each declared read, in bound order.
                std::array<const void*, sizeof...(Reads)> addresses_;
        };

        /**
         * @brief Holds an element functor with the tensors it reads at execution time; it never
         *        leaves the record call.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Fn, typename... Reads>
        class ReadBound final
        {
            public:
                /** @brief Stores the functor and one view of each declared tensor. */
                explicit ReadBound(Fn fn, const Reads&... reads)
                    : fn_(std::move(fn)), tensors_{reads.tensor...}
                {

                }

                /**
                 * @brief Checks every declared read as an operand of one operation.
                 * @param element The dtype of the operation's elements.
                 * @throws std::invalid_argument for a read that is empty, has no storage, holds
                 *         another dtype than it was declared with, or is not contiguous.
                 */
                void require(Core::DataType element, const char* name) const
                {
                    const std::array<Core::DataType, sizeof...(Reads)> expected{
                        Detail::read_dtype(static_cast<const Reads*>(nullptr), element)...};

                    for (std::size_t k = 0; k < sizeof...(Reads); ++k)
                    {
                        const std::string operand = "read " + std::to_string(k);
                        require_min_elements(tensors_[k], 1, name, operand.c_str());
                        require_storage(tensors_[k], name, operand.c_str());
                        require_dtype(tensors_[k], expected[k], name, operand.c_str());
                        require_device_dtype(tensors_[k], name, operand.c_str());
                        require_contiguous(tensors_[k], name, operand.c_str());
                    }
                }

                /**
                 * @brief Declares every read in operands, after the operation declared its own tensors.
                 * @throws std::invalid_argument for a read that overlaps a region the task writes.
                 */
                void declare_into(OperandSet& operands, const char* name) const
                {
                    for (std::size_t k = 0; k < sizeof...(Reads); ++k)
                    {
                        const std::string operand = "read " + std::to_string(k);
                        const SushiRuntime::Core::ResourceRegion region = tensors_[k].region();
                        for (const SushiRuntime::Core::ResourceRegion& written :
                             operands.write_regions())
                            require_operand(!region.overlaps(written), name, operand.c_str(),
                                            "overlaps a tensor the operation writes");
                    }

                    for (const Tensor& tensor : tensors_)
                        operands.reads(tensor);
                }

                /** @brief Returns the functor with the reads' base addresses, without the tensors. */
                [[nodiscard]] AddressedOp<Fn, Reads...> addressed() const
                {
                    std::array<const void*, sizeof...(Reads)> addresses{};
                    for (std::size_t k = 0; k < sizeof...(Reads); ++k)
                        addresses[k] = tensors_[k].data_or_null();
                    return AddressedOp<Fn, Reads...>(fn_, addresses);
                }

            private:
                /// @brief The element functor, or a bind_scalars bundle of one.
                Fn fn_;

                /// @brief One view of each declared tensor, in bound order.
                std::array<Tensor, sizeof...(Reads)> tensors_;
        };

        /**
         * @brief Binds the tensors an element functor reads at execution time, for a seam to declare.
         * @param fn Callable `(elements..., read pointers...)`; it takes one pointer per read after
         *        the elements, and may be a bind_scalars bundle, whose scalars then come last.
         * @param reads One read_as_element or read_as<U> per tensor, in the order fn takes them.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Fn, typename... Reads>
        [[nodiscard]] inline ReadBound<Fn, Reads...> bind_reads(Fn fn, const Reads&... reads)
        {
            static_assert(sizeof...(Reads) >= 1, "bind_reads needs at least one declared read.");

            return ReadBound<Fn, Reads...>(std::move(fn), reads...);
        }

        namespace Detail
        {
            /** @brief Checks nothing, because a bare functor declares no reads. */
            template<typename Op>
            inline void require_declared_reads(const Op&, Core::DataType, const char*) noexcept
            {

            }

            /** @brief Checks the reads a bundle declares; see ReadBound::require. */
            template<typename Fn, typename... Reads>
            inline void require_declared_reads(const ReadBound<Fn, Reads...>& bound,
                                               Core::DataType element, const char* name)
            {
                bound.require(element, name);
            }

            /** @brief Adds nothing, because a bare functor declares no reads. */
            template<typename Op>
            inline void declare_reads(const Op&, OperandSet&, const char*) noexcept
            {

            }

            /** @brief Adds a bundle's reads to a task's operand set; see ReadBound::declare_into. */
            template<typename Fn, typename... Reads>
            inline void declare_reads(const ReadBound<Fn, Reads...>& bound, OperandSet& operands,
                                      const char* name)
            {
                bound.declare_into(operands, name);
            }

            /** @brief Returns a functor without declared reads unchanged, for the task closure. */
            template<typename Op>
            [[nodiscard]] inline const Op& task_functor(const Op& op) noexcept
            {
                return op;
            }

            /** @brief Returns a bundle's functor with its read addresses, for the task closure. */
            template<typename Fn, typename... Reads>
            [[nodiscard]] inline AddressedOp<Fn, Reads...> task_functor(
                const ReadBound<Fn, Reads...>& bound)
            {
                return bound.addressed();
            }

            /** @brief Returns the kernel functor of an op without declared reads; see narrow_for. */
            template<typename T, typename Op>
            [[nodiscard]] inline decltype(auto) kernel_op(ElementTag<T> tag, const Op& op)
            {
                return narrow_for(tag, op);
            }

            /** @brief Returns the kernel functor of an op with declared reads, typed for T. */
            template<typename T, typename Fn, typename... Reads>
            [[nodiscard]] inline auto kernel_op(ElementTag<T> tag,
                                                const AddressedOp<Fn, Reads...>& op)
            {
                return op.typed(tag);
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
