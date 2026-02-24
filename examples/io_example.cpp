/**************************************************************************/
/* io_example.cpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
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

#include <complex>
#include <iostream>
#include <SushiBLAS/SushiBLAS.h>

int main() 
{
    try 
    {
        SushiRuntime::Execution::RuntimeContext ctx;
        SushiBLAS::Engine engine(ctx);

        std::cout << "--- SushiBLAS IO: load_npy & to_string Example ---\n\n";

        // 1. Create a tensor and fill it
        auto A = engine.create_tensor({2, 3});
        float* data = A.data_as<float>();

        for (int i = 0; i < 6; ++i)
            data[i] = static_cast<float>(i) * 1.5f;

        // 2. Demonstrate to_string()
        std::cout << "Step 2: Testing to_string()...\n";
        std::string s = engine.io().to_string(A, 2);
        std::cout << "Tensor A string representation:\n" << s << std::endl;

        // 3. Demonstrate save_npy/load_npy
        std::cout << "Step 3: Testing save_npy() and load_npy()...\n";
        std::string npy_path = "io_test.npy";
        engine.io().save_npy(A, npy_path);
        
        auto B = engine.create_tensor({2, 3});
        engine.io().load_npy(B, npy_path);

        std::cout << "Loaded Tensor B from " << npy_path << ":\n";
        engine.io().print(B, 2);

        // 4. Test COMPLEX support again with new structure
        std::cout << "\nStep 4: Testing Complex Numbers with to_string()...\n";
        auto C = engine.create_tensor({2}, SushiBLAS::Core::DataType::COMPLEX32);
        float* c_ptr = C.data_as<float>();
        c_ptr[0] = 1.0f; c_ptr[1] = 2.0f; // (1+2j)
        c_ptr[2] = -0.5f; c_ptr[3] = 0.0f; // -0.5

        std::cout << "Tensor C:\n" << engine.io().to_string(C) << std::endl;

        std::cout << "Example completed successfully.\n";
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
