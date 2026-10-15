/**************************************************************************/
/* serial_reductions.hpp                                                  */
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
 * @file serial_reductions.hpp
 * @brief Defines serial reference reductions over a host vector.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>

namespace ReferenceKernels 
{
    template<typename T>
    T sum(const std::vector<T>& data) 
    {
        return std::accumulate(data.begin(), data.end(), T(0));
    }

    template<typename T>
    T mean(const std::vector<T>& data) 
    {
        if (data.empty())
            return T(0);
        return sum(data) / static_cast<T>(data.size());
    }

    template<typename T>
    T max(const std::vector<T>& data) 
    {
        if (data.empty())
            return std::numeric_limits<T>::lowest();
        T max_val = data[0];
        for (const auto& val : data) 
            if (val > max_val)
                max_val = val;
        return max_val;
    }

    template<typename T>
    T min(const std::vector<T>& data) 
    {
        if (data.empty())
            return std::numeric_limits<T>::max();
        T min_val = data[0];
        for (const auto& val : data) 
            if (val < min_val)
                min_val = val;
        return min_val;
    }

    template<typename T>
    T argmax(const std::vector<T>& data) 
    {
        if (data.empty())
            return T(0);
        T max_val = data[0];
        int64_t max_idx = 0;
        for (size_t i = 1; i < data.size(); ++i) 
        {
            if (data[i] > max_val) 
            {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<T>(max_idx);
    }

    template<typename T>
    T argmin(const std::vector<T>& data) 
    {
        if (data.empty())
            return T(0);
        T min_val = data[0];
        int64_t min_idx = 0;
        for (size_t i = 1; i < data.size(); ++i) 
        {
            if (data[i] < min_val) 
            {
                min_val = data[i];
                min_idx = i;
            }
        }
        return static_cast<T>(min_idx);
    }

    template<typename T>
    T var(const std::vector<T>& data) 
    {
        if (data.empty())
            return T(0);
        T m = mean(data);
        T accum = 0;
        for (const auto& val : data) 
        {
            accum += (val - m) * (val - m);
        }
        return accum / static_cast<T>(data.size());
    }

    template<typename T>
    T std_dev(const std::vector<T>& data) 
    {
        return std::sqrt(var(data));
    }

    template<typename T>
    T prod(const std::vector<T>& data) 
    {
        T p = 1;
        for (const auto& val : data)
            p *= val;
        return p;
    }

    template<typename T>
    T logsumexp(const std::vector<T>& data) 
    {
        if (data.empty())
            return T(0);
        T max_val = max(data);
        T sum_exp = 0;
        for (const auto& val : data) 
        {
            sum_exp += std::exp(val - max_val);
        }
        return max_val + std::log(sum_exp);
    }

    template<typename T>
    T asum(const std::vector<T>& data) 
    {
        T s = 0;
        for (const auto& val : data)
            s += std::abs(val);
        return s;
    }

    template<typename T>
    T nrm2(const std::vector<T>& data) 
    {
        T s = 0;
        for (const auto& val : data)
            s += val * val;
        return std::sqrt(s);
    }
}
