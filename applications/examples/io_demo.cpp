/**************************************************************************/
/* io_demo.cpp                                                            */
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
 * @file io_demo.cpp
 * @brief Saves a tensor to a .npy file, loads it back, prints it and tests every IOStatus.
 * @author Mustafa Garip
 */

#include <complex>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <SushiBLAS/SushiBLAS.h>

namespace
{
    /** @brief Names a file and removes it when the object is destroyed. */
    class TemporaryFile final
    {
        public:
            /** @brief Takes the path of a file that need not exist yet. */
            explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path))
            {

            }

            TemporaryFile(const TemporaryFile&) = delete;
            TemporaryFile& operator=(const TemporaryFile&) = delete;
            TemporaryFile(TemporaryFile&&) = delete;
            TemporaryFile& operator=(TemporaryFile&&) = delete;

            /** @brief Removes the file; a file that is not there is not an error. */
            ~TemporaryFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            /** @brief Returns the path as the string the IO calls take. */
            [[nodiscard]] std::string name() const
            {
                return path_.string();
            }

        private:
            /// @brief The file this object removes.
            std::filesystem::path path_;
    };
} // namespace

int main()
{
    SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
    SushiBLAS::Engine engine(runtime);
    SushiBLAS::IO io(engine);

    const std::vector<float> values{0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.5f};
    SushiBLAS::Tensor written = engine.create_tensor({2, 3});
    io.from_vector(written, values);
    std::cout << "written:\n" << io.to_string(written, 2) << "\n";

    const TemporaryFile file(std::filesystem::temp_directory_path() / "sushiblas_io_demo.npy");
    const SushiBLAS::IOStatus saved = io.save_npy(written, file.name());
    if (saved != SushiBLAS::IOStatus::Ok)
    {
        std::cerr << "save_npy: " << SushiBLAS::to_string(saved) << "\n";
        return 1;
    }

    SushiBLAS::Tensor loaded = engine.create_tensor({2, 3});
    const SushiBLAS::IOStatus load_status = io.load_npy(loaded, file.name());
    if (load_status != SushiBLAS::IOStatus::Ok)
    {
        std::cerr << "load_npy: " << SushiBLAS::to_string(load_status) << "\n";
        return 1;
    }
    std::cout << "loaded:\n";
    io.print(loaded, 2);
    std::cout << "\n";

    // The file holds a 2 x 3 tensor, so a 3 x 2 tensor refuses it.
    SushiBLAS::Tensor other_shape = engine.create_tensor({3, 2});
    const SushiBLAS::IOStatus mismatched = io.load_npy(other_shape, file.name());
    std::cout << "load_npy into a 3 x 2 tensor: " << SushiBLAS::to_string(mismatched) << "\n";

    const std::vector<std::complex<float>> complex_values{{1.0f, 2.0f}, {-0.5f, 0.0f}};
    SushiBLAS::Tensor complex_tensor =
        engine.create_tensor({2}, SushiBLAS::Core::DataType::Complex32);
    io.from_vector(complex_tensor, complex_values);
    std::cout << "complex:\n" << io.to_string(complex_tensor) << "\n";

    const bool correct = io.to_vector<float>(loaded) == values &&
                         mismatched == SushiBLAS::IOStatus::ShapeMismatch &&
                         io.to_vector<std::complex<float>>(complex_tensor) == complex_values;
    std::cout << "io_demo: " << (correct ? "ok" : "MISMATCH") << "\n";
    return correct ? 0 : 1;
}
