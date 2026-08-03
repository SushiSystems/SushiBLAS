/**************************************************************************/
/* elementwise.hpp                                                        */
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

#pragma once

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class ElementwiseOps
     * @brief Element-wise arithmetic and mathematical operations for tensors.
     * 
     * Provides high-performance independent operations where each element 
     * in the result depends only on its corresponding element in the input(s).
     */
    class ElementwiseOps 
    {
        public:
            /**
            * @brief Construct ElementwiseOps with a reference to the engine.
            * @param e The SushiBLAS engine.
            */
            explicit ElementwiseOps(Engine& e) : engine_(e) {}

            /** 
             * @brief Element-wise addition.
             * Computes C = A + B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor to store the result.
             * @return sycl::event representing task completion.
             */
            sycl::event add(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise subtraction.
             * Computes C = A - B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event sub(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise multiplication (Hadamard product).
             * Computes C = A * B element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event mul(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise division.
             * Computes C = A / B element-wise.
             * @param A First input tensor (dividend).
             * @param B Second input tensor (divisor).
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event div(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise square root.
             * Computes t = sqrt(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event sqrt(Tensor& t);

            /** 
             * @brief Element-wise square.
             * Computes t = t * t for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event square(Tensor& t);

            /** 
             * @brief Element-wise reciprocal.
             * Computes t = 1 / t for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event reciprocal(Tensor& t);

            /** 
             * @brief Element-wise negation.
             * Computes t = -t for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event neg(Tensor& t);

            /** 
             * @brief Element-wise exponential.
             * Computes t = exp(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event exp(Tensor& t);

            /** 
             * @brief Element-wise natural logarithm.
             * Computes t = log(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event log(Tensor& t);

            /** 
             * @brief Element-wise absolute value.
             * Computes t = abs(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event abs(Tensor& t);

            /** 
             * @brief Element-wise power function.
             * Computes t = t^exponent for each element in-place.
             * @param t Input/Output tensor.
             * @param exponent The power to raise elements to.
             * @return sycl::event.
             */
            sycl::event pow(Tensor& t, float exponent);

            /** 
             * @brief Element-wise sine.
             * Computes t = sin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event sin(Tensor& t);

            /** 
             * @brief Element-wise cosine.
             * Computes t = cos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event cos(Tensor& t);

            /** 
             * @brief Element-wise tangent.
             * Computes t = tan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event tan(Tensor& t);

            /** 
             * @brief Element-wise arcsine.
             * Computes t = asin(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event asin(Tensor& t);

            /** 
             * @brief Element-wise arccosine.
             * Computes t = acos(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event acos(Tensor& t);

            /** 
             * @brief Element-wise arctangent.
             * Computes t = atan(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event atan(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic sine.
             * Computes t = sinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event sinh(Tensor& t);

            /** 
             * @brief Element-wise hyperbolic cosine.
             * Computes t = cosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event cosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic sine.
             * Computes t = asinh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event asinh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic cosine.
             * Computes t = acosh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event acosh(Tensor& t);

            /** 
             * @brief Element-wise inverse hyperbolic tangent.
             * Computes t = atanh(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event atanh(Tensor& t);

            /** 
             * @brief Element-wise ceiling.
             * Computes t = ceil(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event ceil(Tensor& t);

            /** 
             * @brief Element-wise floor.
             * Computes t = floor(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event floor(Tensor& t);

            /** 
             * @brief Element-wise rounding.
             * Computes t = round(t) for each element in-place.
             * @param t Input/Output tensor.
             * @return sycl::event.
             */
            sycl::event round(Tensor& t);

            /** 
             * @brief Element-wise minimum.
             * Computes C = min(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event min(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise maximum.
             * Computes C = max(A, B) element-wise.
             * @param A First input tensor.
             * @param B Second input tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event max(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise floating-point remainder.
             * Computes C = fmod(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event fmod(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise remainder.
             * Computes C = remainder(A, B) element-wise.
             * @param A Dividend tensor.
             * @param B Divisor tensor.
             * @param C Output tensor.
             * @return sycl::event.
             */
            sycl::event remainder(const Tensor& A, const Tensor& B, Tensor& C);

            /** 
             * @brief Element-wise clamping.
             * Computes t = max(min_val, min(max_val, t)) for each element in-place.
             * @param t Input/Output tensor.
             * @param min_val Minimum value.
             * @param max_val Maximum value.
             * @return sycl::event.
             */
            sycl::event clamp(Tensor& t, float min_val, float max_val);

            /**
             * @brief Copies @p src into @p dst, converting the element type.
             *
             * The one operation in this library whose two operands are *allowed* to
             * disagree on dtype, and it exists because before it there was no way to
             * produce an fp16 view of an fp32 tensor at all: `Level1::copy` throws on a
             * dtype mismatch, and every other op in the library dispatches on exactly one
             * `Core::DataType` shared by all its operands. A mixed-precision training step
             * needs exactly the missing thing - fp32 master weights cast down to the fp16
             * the matrix multiply runs in, and fp16 gradients cast back up before they
             * reach an optimiser that must not accumulate in fp16.
             *
             * **Why it lives here and not beside `copy`.** `Level1` is the BLAS level-1
             * surface, and reference BLAS has no converting copy at any level: every
             * routine there is fixed to a single precision, which is why `?copy` carries
             * one type letter and not two. Putting a routine that is not in the standard
             * under `engine.blas()` would cost that accessor the only thing it means. What
             * is left is the family this op already belongs to by shape - `dst[i]` is a
             * function of `src[i]` and of nothing else, which is the sentence
             * @ref ElementwiseOps opens with. The only thing separating it from every
             * other elementwise op is that its input and output element types differ, and
             * that is a property of the dispatch it needs rather than of what it computes.
             *
             * **The dtype cross product is bounded to REAL x REAL: nine arms.** This is
             * the real design constraint. A converting op dispatches on *two* runtime
             * dtypes, so the naive spelling - `with_element_type` nested inside
             * `with_element_type` - is 7 x 7 = 49 kernel instantiations for one operation,
             * more than this library's entire elementwise family costs put together. It is
             * also a shape this tree's clang is known to miscompile: the nested generic
             * lambda is exactly what crashes it during instantiation, which is why
             * `Internal::execute_blas_task` spells its own switch out by hand
             * (<SushiBLAS/engine/blas/utils.hpp>). So the pair is restricted to
             * `Dispatch::Domain::REAL` on **both** sides - HALF, FLOAT32, FLOAT64, nine
             * arms - and the switch is written as two plain, un-nested `switch`
             * statements over ordinary function templates rather than as lambdas.
             *
             * REAL x REAL is not an arbitrary trim. It is what mixed precision consists
             * of, and every pair outside it is either meaningless or a bug wearing a cast:
             * a complex operand has no single real value to narrow to, and a float-to-int
             * conversion is a *rounding policy* decision (truncate, round-half-even,
             * saturate?) that this layer has no basis to make on the caller's behalf. Both
             * therefore throw at record time, **naming both dtypes**, rather than silently
             * doing nothing - the defect this project has called out by name more than
             * once. An integer conversion, if one is ever wanted, belongs in its own op
             * whose name states the rounding it performs.
             *
             * **Converting from a dtype to itself is legal and copies.** The alternative -
             * making `convert(f32, f32)` an error - would push a branch back into every
             * caller, and the caller least able to take it is the one this op was written
             * for: a mixed-precision policy holds its compute dtype as runtime data, and
             * when that dtype happens to equal the master dtype the op degenerates
             * naturally into the copy the policy meant. It is not a silent no-op either -
             * the task is still recorded and the elements are still written, so the
             * dependency edges a caller gets are the same whatever the two dtypes are,
             * which is the property that actually has to hold for a graph to be correct.
             *
             * **What it declares to the dependency tracker.** `src.region()` as a read and
             * `dst.region()` as a write, taken from @ref Tensor::region and never from a
             * bare pointer. A `ResourceRegion` converts implicitly from a pointer and a
             * bare pointer widens to the *whole* allocation, so two views of one arena
             * become two unrelated resources, `overlaps` answers false and the tracker
             * emits no edge at all - not a wrong edge, none. That mistake has produced a
             * silently wrong training loss in this stack twice
             * (<SushiBLAS/engine/math/reductions/dispatch.hpp> records one of them), and
             * it is precisely the arena-packed fp32/fp16 pair this op exists to serve.
             *
             * @param src The tensor to read. Must be contiguous and have storage.
             * @param dst The tensor to overwrite, in its own dtype. Must be contiguous,
             *        have storage, have the same shape as @p src, and not overlap it -
             *        each work-item reads one element and writes another of a different
             *        width, so a partial overlap is a race no launch shape can fix.
             * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
             * @throws std::runtime_error naming both dtypes if either falls outside
             *         `Domain::REAL`; naming both shapes if they disagree; naming the
             *         offending operand if it is storage-less, non-contiguous, or overlaps
             *         the other.
             */
            sycl::event convert(const Tensor& src, Tensor& dst);

        private:
            Engine& engine_;
    };
} // namespace SushiBLAS
