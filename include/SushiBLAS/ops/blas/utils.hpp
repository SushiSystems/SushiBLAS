/**************************************************************************/
/* utils.hpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/**************************************************************************/

#pragma once
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine.hpp>

namespace SushiBLAS 
{
    namespace Internal 
    {
        /**
         * @brief Helper to extract 1D vector parameters (size and increment) from a Tensor.
         * 
         * @param t The input tensor.
         * @param n Output: Number of elements.
         * @param inc Output: Increment (stride).
         */
        inline void get_vec_params(const Tensor& t, int64_t& n, int64_t& inc) 
        {
            n = t.num_elements;
            if (t.is_contiguous()) 
            {
                inc = 1;
            } 
            else 
            {
                // TODO: Support rank > 1 tensors by allowing 2D matrix row/column slices (extracting correct strides).
                SB_THROW_IF(t.rank > 1, "Level 1 BLAS expects contiguous memory for rank > 1 tensors.");
                inc = t.strides[0];
            }
        }
    }
}
