/****************************************************************/
/* gemm_epilogue.hpp                                            */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file gemm_epilogue.hpp
 * @brief The GEMM epilogue contract, shared by both GEMM kernels and the seam.
 *
 * It lives in its own header, under neither `reference/` nor `sycl/`, because
 * it belongs to neither: `Kernels::Reference::gemm` and `Kernels::SYCL::gemm`
 * both take it as a template parameter and both default to the identity below,
 * and the public seam that exposes it - `Dispatch::execute_gemm`
 * (<SushiBLAS/engine/blas/dispatch.hpp>) - names it in its own signature. One
 * definition, three users, no direction of dependency between them.
 */

#include <cstdint>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
        * @struct GemmIdentityEpilogue
        * @brief The default GEMM epilogue: stores the computed element unchanged.
        *
        * Exists so that the fusion hook costs nothing when unused - the call
        * inlines to the value itself and the generated kernel is byte for byte
        * the unfused one.
        *
        * @see Kernels::SYCL::gemm and SushiBLAS::Dispatch::execute_gemm for the
        *      fusion contract, and for the training-correctness trap that comes
        *      with any *activation* epilogue.
        */
        struct GemmIdentityEpilogue
        {
            /**
            * @brief Returns @p value unchanged.
            *
            * @tparam T The element type of C.
            * @param value The fully formed C element, alpha*(op(A)*op(B)) + beta*C.
            * @param row The element's row in C. Unused by the identity.
            * @param col The element's column in C. Unused by the identity.
            * @return @p value.
            */
            template<typename T>
            [[nodiscard]] T operator()(T value,
                                       [[maybe_unused]] int64_t row,
                                       [[maybe_unused]] int64_t col) const noexcept
            {
                return value;
            }
        };

    } // namespace Kernels
} // namespace SushiBLAS
