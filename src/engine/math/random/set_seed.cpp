/****************************************************************/
/* set_seed.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>

namespace SushiBLAS 
{
    void RandomOps::set_seed(uint64_t seed) 
    {
        SB_LOG_INFO("RandomOps: set_seed ({})", seed);
        engine_.set_seed(seed);
    }
} // namespace SushiBLAS
