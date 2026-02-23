/**************************************************************************/
/* blas.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <SushiBLAS/ops/blas/level1.hpp>
#include <SushiBLAS/ops/blas/level2.hpp>
#include <SushiBLAS/ops/blas/level3.hpp>
#include <SushiBLAS/ops/blas/sparse.hpp>

namespace SushiBLAS 
{
    class Engine;

    /**
     * @class BLASOps
     * @brief A unified interface for all BLAS operations (Standard & Sparse).
     * 
     * This class inherits from Level1, Level2, Level3, and SparseBLAS to provide 
     * a single point of access for all BLAS functionality within SushiBLAS.
     */
    class BLASOps : public Level1, public Level2, public Level3, public SparseBLAS
    {
        public:
            /**
             * @brief Construct BLASOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit BLASOps(Engine& e) : Level1(e), Level2(e), Level3(e), SparseBLAS(e) {}
    };

} // namespace SushiBLAS
