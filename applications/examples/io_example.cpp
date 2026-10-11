/**************************************************************************/
/* io_example.cpp                                                         */
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
 * @file io_example.cpp
 * @brief Runs a tensor through to_string, save_npy and load_npy, then prints a complex tensor.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

int main() 
{
    try 
    {
        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);

        std::cout << "--- SushiBLAS IO: load_npy & to_string Example ---\n\n";

        // 1. Create a tensor and fill it
        auto A = engine.create_tensor({2, 3});
        std::vector<float> values(6);

        for (int i = 0; i < 6; ++i)
            values[i] = static_cast<float>(i) * 1.5f;

        SushiBLAS::IO(engine).from_vector(A, values);

        // 2. Demonstrate to_string()
        std::cout << "Step 2: Testing to_string()...\n";
        std::string s = SushiBLAS::IO(engine).to_string(A, 2);
        std::cout << "Tensor A string representation:\n" << s << std::endl;

        // 3. Demonstrate save_npy/load_npy
        std::cout << "Step 3: Testing save_npy() and load_npy()...\n";
        std::string npy_path = "io_test.npy";
        const SushiBLAS::IOStatus saved = SushiBLAS::IO(engine).save_npy(A, npy_path);
        if (saved != SushiBLAS::IOStatus::Ok)
        {
            std::cerr << "save_npy failed: " << SushiBLAS::to_string(saved) << std::endl;
            return 1;
        }

        auto B = engine.create_tensor({2, 3});
        const SushiBLAS::IOStatus loaded = SushiBLAS::IO(engine).load_npy(B, npy_path);
        if (loaded != SushiBLAS::IOStatus::Ok)
        {
            std::cerr << "load_npy failed: " << SushiBLAS::to_string(loaded) << std::endl;
            return 1;
        }

        std::cout << "Loaded Tensor B from " << npy_path << ":\n";
        SushiBLAS::IO(engine).print(B, 2);

        // 4. Test COMPLEX support again with new structure
        std::cout << "\nStep 4: Testing Complex Numbers with to_string()...\n";
        auto C = engine.create_tensor({2}, SushiBLAS::Core::DataType::Complex32);
        SushiBLAS::IO(engine).from_vector(
            C, std::vector<std::complex<float>>{{1.0f, 2.0f}, {-0.5f, 0.0f}});

        std::cout << "Tensor C:\n" << SushiBLAS::IO(engine).to_string(C) << std::endl;

        std::cout << "Example completed successfully.\n";
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
