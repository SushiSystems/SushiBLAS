/****************************************************************/
/* export.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

  #ifdef SUSHIBLAS_EXPORTS
    #define SUSHIBLAS_API __declspec(dllexport)
  #else
    #define SUSHIBLAS_API __declspec(dllimport)
  #endif
#else
  #define SUSHIBLAS_API __attribute__((visibility("default")))
#endif
