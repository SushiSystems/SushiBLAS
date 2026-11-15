/**************************************************************************/
/* test_file_backed_product.cpp                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_file_backed_product.cpp
 * @brief Tests a blocked product graph over file-backed tensors larger than any device's own memory.
 * @author Mustafa Garip
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/engine_fixture/recording_fixture.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/coherence/transfer_observer.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rows and columns of every weight block, the rows of every panel.
            constexpr std::uint64_t K_BLOCK_DIM = 4096;

            /// @brief Columns of every panel a chain carries.
            constexpr std::uint64_t K_PANEL_COLUMNS = 64;

            /// @brief Weight blocks in one chain: the products one panel goes through.
            constexpr std::size_t K_CHAIN_LENGTH = 8;

            /// @brief Runs of the graph; the first loads the file.
            constexpr std::size_t K_RUNS = 10;

            /// @brief Size of the file's blocks against the memory of the largest device that has its own, in percent.
            constexpr std::uint64_t K_OVERSIZE_PERCENT = 125;

            /// @brief Bytes of free disk the file leaves untouched.
            constexpr std::uint64_t K_DISK_MARGIN_BYTES = std::uint64_t{1} << 30;

            /// @brief Share of a device's memory the stated-device run may fill, in percent.
            constexpr std::uint64_t K_STATED_FILL_PERCENT = 90;

            /// @brief The ratio of the T33 row: the planned step is at most this many times the floor's.
            constexpr double K_ALLOWED_RATIO = 1.05;

            /// @brief Multiplier set of the permutations: the odd numbers below twice this value.
            constexpr std::size_t K_MULTIPLIER_SET = 61;

            /// @brief The name the file path of the runtime gives its copies.
            constexpr const char K_FILE_PATH_NAME[] = "file";

            /// @brief Bytes of one weight block.
            constexpr std::size_t K_BLOCK_BYTES =
                K_BLOCK_DIM * K_BLOCK_DIM * sizeof(float);

            /// @brief Elements of one panel.
            constexpr std::size_t K_PANEL_ELEMENTS =
                K_BLOCK_DIM * K_PANEL_COLUMNS;

            /// @brief Bytes of the tensors a chain holds in all: its weights, and its input, stages, sum and output.
            constexpr std::size_t K_CHAIN_BYTES =
                K_CHAIN_LENGTH * K_BLOCK_BYTES + (K_CHAIN_LENGTH + 3) * K_PANEL_ELEMENTS * sizeof(float);

            /** @brief Returns whether a backend is the CUDA backend. */
            bool is_cuda(sycl::backend backend) noexcept
            {
#if defined(SYCL_IMPLEMENTATION_ACPP)
                return backend == sycl::backend::cuda;
#else
                return backend == sycl::backend::ext_oneapi_cuda;
#endif
            }

            /** @brief Returns whether a backend is the HIP backend. */
            bool is_hip(sycl::backend backend) noexcept
            {
#if defined(SYCL_IMPLEMENTATION_ACPP)
                return backend == sycl::backend::hip;
#else
                return backend == sycl::backend::ext_oneapi_hip;
#endif
            }

            /** @brief Returns whether the runtime drives a GPU whose backend @p wanted accepts. */
            template<class Wanted>
            bool has_gpu(const std::vector<SushiRuntime::DeviceDescription>& devices, Wanted wanted)
            {
                return std::any_of(devices.begin(), devices.end(),
                                   [&](const SushiRuntime::DeviceDescription& device)
                                   { return device.kind == SushiRuntime::DeviceKind::GPU && wanted(device.backend); });
            }

            /** @brief Returns whether the runtime drives a device of kind CPU. */
            bool has_cpu(const std::vector<SushiRuntime::DeviceDescription>& devices)
            {
                return std::any_of(devices.begin(), devices.end(),
                                   [](const SushiRuntime::DeviceDescription& device)
                                   { return device.kind == SushiRuntime::DeviceKind::CPU; });
            }

            /** @brief Returns the memory of the largest GPU that has memory of its own; zero when none has. */
            std::uint64_t largest_own_memory(const std::vector<SushiRuntime::DeviceDescription>& devices)
            {
                std::uint64_t largest = 0;
                for (const SushiRuntime::DeviceDescription& device : devices)
                    if (device.kind == SushiRuntime::DeviceKind::GPU && !device.integrated)
                        largest = std::max(largest, device.memory_bytes);
                return largest;
            }

            /** @brief Returns the description of the device a key names, or null. */
            const SushiRuntime::DeviceDescription*
            describe(const std::vector<SushiRuntime::DeviceDescription>& devices, const SushiRuntime::DeviceKey& key)
            {
                for (const SushiRuntime::DeviceDescription& device : devices)
                    if (device.key == key)
                        return &device;
                return nullptr;
            }

            /** @brief Returns the column of the one 1 in row @p row of the weight block @p step of chain @p chain. */
            std::uint64_t permuted_row(std::size_t chain, std::size_t step, std::uint64_t row) noexcept
            {
                const std::uint64_t multiplier = 2 * ((chain * K_CHAIN_LENGTH + step) % K_MULTIPLIER_SET) + 1;
                const std::uint64_t offset = 7 * chain + 13 * step + 1;
                return (multiplier * row + offset) % K_BLOCK_DIM;
            }

            /** @brief Returns element (row, column) of the input panel of a chain: a positive integer below 252. */
            float panel_value(std::size_t chain, std::uint64_t row, std::uint64_t column) noexcept
            {
                return static_cast<float>((chain * 131 + row * 7 + column * 3) % 251 + 1);
            }

            /** @brief Returns the input panel of a chain, row major. */
            std::vector<float> input_panel(std::size_t chain)
            {
                std::vector<float> panel(K_PANEL_ELEMENTS);
                for (std::uint64_t row = 0; row < K_BLOCK_DIM; ++row)
                    for (std::uint64_t column = 0; column < K_PANEL_COLUMNS; ++column)
                        panel[row * K_PANEL_COLUMNS + column] = panel_value(chain, row, column);
                return panel;
            }

            /**
             * @brief Returns the outputs of every chain, one after the other, computed on the host.
             *
             * A weight block holds one 1 per row, so one product takes row p(i) of the panel to row i; the
             * chain's panel at its end is row p0(p1(...p7(i))) of the input, and the output adds the input.
             * Every value is a sum of small integers, exact in float in any order.
             */
            std::vector<float> expected_outputs(std::size_t chain_count)
            {
                std::vector<float> expected;
                expected.reserve(chain_count * K_PANEL_ELEMENTS);
                for (std::size_t chain = 0; chain < chain_count; ++chain)
                {
                    for (std::uint64_t row = 0; row < K_BLOCK_DIM; ++row)
                    {
                        std::uint64_t source = row;
                        for (std::size_t step = K_CHAIN_LENGTH; step > 0; --step)
                            source = permuted_row(chain, step - 1, source);
                        for (std::uint64_t column = 0; column < K_PANEL_COLUMNS; ++column)
                            expected.push_back(panel_value(chain, source, column) + panel_value(chain, row, column));
                    }
                }
                return expected;
            }

            /** @brief Returns how many elements of two equal-length arrays differ in their bytes. */
            std::size_t count_differences(const std::vector<float>& actual, const std::vector<float>& expected)
            {
                if (actual.size() != expected.size())
                    return std::max(actual.size(), expected.size());
                std::size_t differing = 0;
                for (std::size_t index = 0; index < actual.size(); ++index)
                    if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0)
                        ++differing;
                return differing;
            }

            /** @brief Writes the weight blocks of every chain to @p path; returns the failure, or empty. */
            std::string write_weight_file(const std::filesystem::path& path, std::size_t chain_count)
            {
                std::ofstream file(path, std::ios::binary | std::ios::trunc);
                if (!file)
                    return "the weight file could not be opened for writing: " + path.string();
                std::vector<float> block(K_BLOCK_DIM * K_BLOCK_DIM, 0.0f);
                for (std::size_t chain = 0; chain < chain_count; ++chain)
                {
                    for (std::size_t step = 0; step < K_CHAIN_LENGTH; ++step)
                    {
                        for (std::uint64_t row = 0; row < K_BLOCK_DIM; ++row)
                            block[row * K_BLOCK_DIM + permuted_row(chain, step, row)] = 1.0f;
                        file.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(K_BLOCK_BYTES));
                        for (std::uint64_t row = 0; row < K_BLOCK_DIM; ++row)
                            block[row * K_BLOCK_DIM + permuted_row(chain, step, row)] = 0.0f;
                    }
                }
                file.flush();
                if (!file)
                    return "the weight file could not be written: " + path.string();
                return {};
            }

            /** @brief Reads weight block number @p block of the file at @p path. */
            std::vector<float> read_weight_block(const std::filesystem::path& path, std::size_t block)
            {
                std::vector<float> values(K_BLOCK_BYTES / sizeof(float));
                std::ifstream file(path, std::ios::binary);
                file.seekg(static_cast<std::streamoff>(block * K_BLOCK_BYTES));
                file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(K_BLOCK_BYTES));
                return values;
            }

            /**
             * @brief Returns how many of @p count chains each weighted device takes, none above its capacity.
             * @param weights Relative share of each device; all zero means equal shares.
             * @param capacities Chains each device can hold.
             * @return One count per device, or an empty vector when the capacities hold fewer than @p count.
             */
            std::vector<std::size_t> apportion(std::size_t count, std::vector<std::size_t> weights,
                                               const std::vector<std::size_t>& capacities)
            {
                const std::size_t total_capacity = [&]
                {
                    std::size_t sum = 0;
                    for (const std::size_t capacity : capacities)
                        sum += capacity;
                    return sum;
                }();
                if (weights.empty() || total_capacity < count)
                    return {};
                std::size_t weight_sum = 0;
                for (const std::size_t weight : weights)
                    weight_sum += weight;
                if (weight_sum == 0)
                {
                    std::fill(weights.begin(), weights.end(), std::size_t{1});
                    weight_sum = weights.size();
                }
                std::vector<std::size_t> counts(weights.size(), 0);
                std::size_t given = 0;
                for (std::size_t device = 0; device < weights.size(); ++device)
                {
                    counts[device] = count * weights[device] / weight_sum;
                    given += counts[device];
                }
                for (std::size_t device = 0; given < count; device = (device + 1) % weights.size())
                {
                    if (weights[device] != 0)
                    {
                        ++counts[device];
                        ++given;
                    }
                }
                std::size_t excess = 0;
                for (std::size_t device = 0; device < counts.size(); ++device)
                {
                    if (counts[device] > capacities[device])
                    {
                        excess += counts[device] - capacities[device];
                        counts[device] = capacities[device];
                    }
                }
                for (std::size_t device = 0; excess > 0; device = (device + 1) % counts.size())
                {
                    if (counts[device] < capacities[device])
                    {
                        ++counts[device];
                        --excess;
                    }
                }
                return counts;
            }

            /** @brief Counts the copies the runtime's transfer paths report that read a file space. */
            class FileCopyCounter final : public SushiRuntime::Internal::Coherence::TransferObserver
            {
                public:
                    /** @brief Counts the copy when the file path took it. */
                    void copied(const SushiRuntime::Internal::Coherence::TransferRecord& record) override
                    {
                        if (record.path == K_FILE_PATH_NAME)
                        {
                            file_copies_.fetch_add(1, std::memory_order_relaxed);
                            file_bytes_.fetch_add(record.bytes, std::memory_order_relaxed);
                        }
                    }

                    /** @brief Returns the copies whose source was a file space, so far. */
                    [[nodiscard]] std::uint64_t file_copies() const noexcept
                    {
                        return file_copies_.load(std::memory_order_relaxed);
                    }

                    /** @brief Returns the bytes those copies moved, so far. */
                    [[nodiscard]] std::uint64_t file_bytes() const noexcept
                    {
                        return file_bytes_.load(std::memory_order_relaxed);
                    }

                private:
                    /// @brief Copies read from a file space.
                    std::atomic<std::uint64_t> file_copies_{0};

                    /// @brief Bytes read from a file space.
                    std::atomic<std::uint64_t> file_bytes_{0};
            };

            /** @brief Holds the tensors of one chain: weights, the panel, its stages, the sum and the output. */
            struct Chain
            {
                Tensor input;                /**< The panel the chain starts from. */
                std::vector<Tensor> weights; /**< The weight blocks, in the order they multiply. */
                std::vector<Tensor> stages;  /**< The panel after each block; the last one is the product. */
                Tensor sum;                  /**< The product plus the input. */
                Tensor output;               /**< The copy of the sum the host reads. */
            };

            /**
             * @brief Owns the tensors of a blocked product graph and records it on a recorder.
             *
             * Each chain multiplies one panel by its weight blocks in turn, adds the panel to the product
             * and copies the sum out. A home of none leaves the chain's tensors to the runtime, the weights
             * read from the file; a home of a device index states that device for them all, the weights
             * read from the file by the host once.
             */
            class ProductGraph
            {
                public:
                    /** @brief Makes the tensors of every chain, placed by @p homes, one entry per chain. */
                    ProductGraph(SushiRuntime::Runtime& runtime, const std::filesystem::path& weight_path,
                                 const std::vector<std::optional<std::size_t>>& homes)
                        : factory_(runtime), weight_path_(weight_path)
                    {
                        chains_.reserve(homes.size());
                        for (std::size_t chain = 0; chain < homes.size(); ++chain)
                            chains_.push_back(make_chain(chain, homes[chain]));
                    }

                    /** @brief Records every chain's tasks on @p recorder. */
                    void record(TaskRecorder& recorder)
                    {
                        const std::vector<int64_t> flat{static_cast<int64_t>(K_PANEL_ELEMENTS)};
                        for (Chain& chain : chains_)
                        {
                            for (std::size_t step = 0; step < K_CHAIN_LENGTH; ++step)
                                BLASOps(recorder).gemm(chain.weights[step], chain.stages[step], chain.stages[step + 1]);
                            ElementwiseOps(recorder).add(chain.stages[K_CHAIN_LENGTH], chain.input, chain.sum);
                            const Tensor flat_sum = chain.sum.reshape(flat);
                            Tensor flat_output = chain.output.reshape(flat);
                            BLASOps(recorder).copy(flat_sum, flat_output);
                        }
                    }

                    /** @brief Returns the output of every chain read back to the host, one after the other. */
                    [[nodiscard]] std::vector<float> read_outputs() const
                    {
                        std::vector<float> values(chains_.size() * K_PANEL_ELEMENTS);
                        for (std::size_t chain = 0; chain < chains_.size(); ++chain)
                            chains_[chain].output.storage->copy_to_host(
                                values.data() + chain * K_PANEL_ELEMENTS, 0, K_PANEL_ELEMENTS * sizeof(float));
                        return values;
                    }

                private:
                    /** @brief Returns an empty panel on the home device, or one the runtime places. */
                    [[nodiscard]] Tensor make_panel(const std::optional<std::size_t>& home) const
                    {
                        const std::vector<int64_t> dims{static_cast<int64_t>(K_BLOCK_DIM), static_cast<int64_t>(K_PANEL_COLUMNS)};
                        const SushiRuntime::span<const int64_t> shape(dims.data(), dims.size());
                        if (home)
                            return factory_.create(shape, Core::DataType::Float32, SushiRuntime::Residency::Device, *home);
                        return factory_.planned(shape, Core::DataType::Float32);
                    }

                    /** @brief Returns weight block number @p block: read from the file, or filled from it on the home device. */
                    [[nodiscard]] Tensor make_weight(std::size_t block, const std::optional<std::size_t>& home) const
                    {
                        const std::vector<int64_t> dims{static_cast<int64_t>(K_BLOCK_DIM), static_cast<int64_t>(K_BLOCK_DIM)};
                        const SushiRuntime::span<const int64_t> shape(dims.data(), dims.size());
                        if (!home)
                            return factory_.from_file(weight_path_.string(), block * K_BLOCK_BYTES, shape,
                                                      Core::DataType::Float32);
                        Tensor weight =
                            factory_.create(shape, Core::DataType::Float32, SushiRuntime::Residency::Device, *home);
                        const std::vector<float> values = read_weight_block(weight_path_, block);
                        weight.storage->copy_from_host(values.data(), 0, K_BLOCK_BYTES);
                        return weight;
                    }

                    /** @brief Returns the tensors of chain number @p chain on its home. */
                    [[nodiscard]] Chain make_chain(std::size_t chain, const std::optional<std::size_t>& home) const
                    {
                        Chain made{make_panel(home), {}, {}, make_panel(home), make_panel(home)};
                        const std::vector<float> values = input_panel(chain);
                        made.input.storage->copy_from_host(values.data(), 0, K_PANEL_ELEMENTS * sizeof(float));
                        made.stages.push_back(made.input);
                        for (std::size_t step = 0; step < K_CHAIN_LENGTH; ++step)
                        {
                            made.weights.push_back(make_weight(chain * K_CHAIN_LENGTH + step, home));
                            made.stages.push_back(make_panel(home));
                        }
                        return made;
                    }

                    /// @brief Makes the tensors; it lives as long as they do.
                    TensorFactory factory_;

                    /// @brief The file the weight blocks are read from.
                    std::filesystem::path weight_path_;

                    /// @brief The chains, in order.
                    std::vector<Chain> chains_;
            };

            /** @brief What the runs of one placement of the graph gave. */
            struct Measurement
            {
                std::vector<double> step_seconds;          /**< Seconds each run took, in run order. */
                std::vector<std::size_t> differing;        /**< Elements of each run's output that differ in bytes from the host's. */
                std::vector<std::uint64_t> file_copies;    /**< Copies read from a file space during each run. */
                std::vector<std::uint64_t> file_bytes;     /**< Bytes those copies moved during each run. */
                SushiRuntime::PartitionReport report;      /**< The partition report after the runs. */
                std::string failure;                       /**< What stopped the runs; empty when nothing did. */

                /** @brief Returns the shortest run after the first, which loads; infinity when there is none. */
                [[nodiscard]] double best_step_seconds() const
                {
                    double best = std::numeric_limits<double>::infinity();
                    for (std::size_t run = 1; run < step_seconds.size(); ++run)
                        best = std::min(best, step_seconds[run]);
                    return best;
                }
            };

            /** @brief Prints one number in the form the report takes: `[t33] name value unit`. */
            void print_number(const std::string& name, double value, const char* unit)
            {
                std::cout << "[t33] " << name << ' ' << std::setprecision(9) << value << ' ' << unit << std::endl;
            }

            /** @brief Returns a name with every space turned into an underscore, so a line stays three fields. */
            std::string one_word(std::string text)
            {
                std::replace(text.begin(), text.end(), ' ', '_');
                return text;
            }

            /**
             * @brief Runs the graph K_RUNS times, timing each run and reading its bytes back.
             * @param counter Hears the copies of the runtime during the runs; null for none.
             * @param expected The bytes every run must leave.
             */
            Measurement run_product(SushiRuntime::Runtime& runtime, ProductGraph& graph, const FileCopyCounter* counter,
                                    const std::vector<float>& expected)
            {
                Measurement measurement;
                Engine engine(runtime, Core::Layout::RowMajor);
                RecordingFixture recording(runtime, engine);
                for (std::size_t run = 0; run < K_RUNS; ++run)
                {
                    graph.record(recording.recorder());
                    const std::uint64_t copies_before = counter != nullptr ? counter->file_copies() : 0;
                    const std::uint64_t bytes_before = counter != nullptr ? counter->file_bytes() : 0;
                    const auto begin = std::chrono::steady_clock::now();
                    const SushiRuntime::Result<void> ran = recording.run();
                    const auto end = std::chrono::steady_clock::now();
                    if (!ran)
                    {
                        measurement.failure = "run " + std::to_string(run) + ": " + ran.error().message;
                        return measurement;
                    }
                    measurement.step_seconds.push_back(std::chrono::duration<double>(end - begin).count());
                    measurement.file_copies.push_back(counter != nullptr ? counter->file_copies() - copies_before : 0);
                    measurement.file_bytes.push_back(counter != nullptr ? counter->file_bytes() - bytes_before : 0);
                    measurement.differing.push_back(count_differences(graph.read_outputs(), expected));
                }
                measurement.report = runtime.partition_report();
                return measurement;
            }

            /** @brief Expects every device the report gives a node to have measured time, and at least two to hold one. */
            void expect_every_holder_ran(const SushiRuntime::PartitionReport& report)
            {
                std::size_t holders = 0;
                for (const SushiRuntime::PartitionDeviceRow& row : report.rows)
                {
                    if (row.nodes == 0)
                        continue;
                    ++holders;
                    EXPECT_GT(row.measured_seconds, 0.0)
                        << "the plan gives " << SushiRuntime::to_text(row.device) << " " << row.nodes
                        << " nodes and the device measured no time\n" << report.to_text();
                }
                EXPECT_GE(holders, 2u) << "the plan does not span devices\n" << report.to_text();
            }

            /** @brief Expects no copy to have read a file space in any run after the first. */
            void expect_no_file_read_after_the_first_run(const std::vector<std::uint64_t>& file_copies)
            {
                for (std::size_t run = 1; run < file_copies.size(); ++run)
                    EXPECT_EQ(file_copies[run], 0u) << "run " << run << " read a file space";
            }

            /** @brief Expects every run to have left the host's bytes, and so the bytes of every other run. */
            void expect_every_run_equals_the_host(const std::vector<std::size_t>& differing)
            {
                for (std::size_t run = 0; run < differing.size(); ++run)
                    EXPECT_EQ(differing[run], 0u) << "run " << run << " left bytes that differ from the host";
            }
        } // namespace

        /**
         * @brief Writes a weight file larger than any device's own memory, runs the graph on the runtime's own
         *        placement and again on the floor devices alone, and skips naming what the machine lacks.
         *
         * The runtime is made here, with a profile file, because the shared test environment makes it
         * with none; the partitioner needs the file to price its copies.
         */
        class Large_FileBackedProduct : public ::testing::Test
        {
            protected:
                /** @brief Runs both placements and prints their numbers; notes what is missing or what failed. */
                static void SetUpTestSuite()
                {
                    directory_ = std::filesystem::current_path() / "t33_file_backed_product";
                    std::filesystem::remove_all(directory_);
                    std::filesystem::create_directories(directory_);
                    SushiRuntime::RuntimeConfig config;
                    config.device_profile_file = (directory_ / "device_profile.txt").string();
                    SushiRuntime::Result<SushiRuntime::Runtime> created = SushiRuntime::Runtime::create(config);
                    if (!created)
                    {
                        failure_ = "the runtime could not be created: " + created.error().message;
                        return;
                    }
                    runtime_.emplace(std::move(created).value());
                    note_missing_hardware();
                    if (!missing_.empty())
                        return;
                    try
                    {
                        measure_both();
                    }
                    catch (const std::exception& error)
                    {
                        failure_ = std::string("the measurement stopped: ") + error.what();
                    }
                }

                /** @brief Removes the folder with the weight file and the profile, and releases the runtime. */
                static void TearDownTestSuite()
                {
                    runtime_.reset();
                    std::error_code ignored;
                    std::filesystem::remove_all(directory_, ignored);
                }

                /** @brief Skips the test naming what the machine lacks, or fails it with what stopped the suite. */
                void SetUp() override
                {
                    if (!missing_.empty())
                        GTEST_SKIP() << missing_;
                    ASSERT_TRUE(failure_.empty()) << failure_;
                }

                /** @brief The runs on the runtime's own placement. */
                static const Measurement& planned() noexcept
                {
                    return planned_;
                }

                /** @brief The runs on the floor devices alone. */
                static const Measurement& floor_alone() noexcept
                {
                    return floor_;
                }

                /** @brief Why the floor run was not made; empty when it was. */
                static const std::string& floor_missing() noexcept
                {
                    return floor_missing_;
                }

            private:
                /** @brief Fills missing_ with every device or resource the machine lacks. */
                static void note_missing_hardware()
                {
                    const std::vector<SushiRuntime::DeviceDescription>& devices = runtime_->advanced().devices();
                    if (!has_gpu(devices, is_cuda))
                        missing_ += "the runtime drives no CUDA GPU. ";
                    if (!has_gpu(devices, is_hip))
                        missing_ += "the runtime drives no HIP GPU. ";
                    if (!has_cpu(devices))
                        missing_ += "the runtime drives no CPU device. ";
                    if (largest_own_memory(devices) == 0)
                        missing_ += "the runtime drives no GPU with memory of its own. ";
                    if (!missing_.empty())
                        return;
                    plan_sizes(largest_own_memory(devices));
                    const std::uint64_t available = std::filesystem::space(directory_).available;
                    if (available < file_bytes_ + K_DISK_MARGIN_BYTES)
                        missing_ += "not enough disk: the weight file needs " + std::to_string(file_bytes_) +
                                    " bytes and a margin, the build tree's disk has " + std::to_string(available) + ". ";
                }

                /** @brief Sets the chain count and the file size from the memory of the largest own-memory GPU. */
                static void plan_sizes(std::uint64_t largest_memory)
                {
                    const std::uint64_t wanted = largest_memory / 100 * K_OVERSIZE_PERCENT;
                    const std::uint64_t blocks = (wanted + K_BLOCK_BYTES - 1) / K_BLOCK_BYTES;
                    chain_count_ = static_cast<std::size_t>((blocks + K_CHAIN_LENGTH - 1) / K_CHAIN_LENGTH);
                    file_bytes_ = std::uint64_t{chain_count_} * K_CHAIN_LENGTH * K_BLOCK_BYTES;
                    largest_memory_ = largest_memory;
                }

                /** @brief Writes the file, runs the runtime's placement, then the floor devices alone, and prints. */
                static void measure_both()
                {
                    failure_ = write_weight_file(directory_ / "weights.bin", chain_count_);
                    if (!failure_.empty())
                        return;
                    const std::vector<float> expected = expected_outputs(chain_count_);
                    measure_planned(expected);
                    print_planned();
                    if (!planned_.failure.empty())
                        return;
                    measure_floor(expected);
                    print_floor();
                }

                /** @brief Runs the graph with every tensor left to the runtime and the weights read from the file. */
                static void measure_planned(const std::vector<float>& expected)
                {
                    const auto counter = std::make_shared<FileCopyCounter>();
                    runtime_->advanced().install_transfer_observer(counter);
                    {
                        ProductGraph graph(*runtime_, directory_ / "weights.bin",
                                           std::vector<std::optional<std::size_t>>(chain_count_));
                        planned_ = run_product(*runtime_, graph, counter.get(), expected);
                    }
                    runtime_->advanced().install_transfer_observer(nullptr);
                }

                /** @brief Runs the graph with each chain stated on one of the floor devices, shared as the plan shared them. */
                static void measure_floor(const std::vector<float>& expected)
                {
                    const std::vector<std::size_t> homes = floor_homes();
                    if (!floor_missing_.empty())
                        return;
                    std::vector<std::optional<std::size_t>> stated;
                    for (const std::size_t home : homes)
                        stated.emplace_back(home);
                    try
                    {
                        ProductGraph graph(*runtime_, directory_ / "weights.bin", stated);
                        floor_ = run_product(*runtime_, graph, nullptr, expected);
                    }
                    catch (const std::exception& error)
                    {
                        floor_.failure = std::string("the floor devices could not run the graph: ") + error.what();
                    }
                }

                /**
                 * @brief Returns the device of each chain on the floor devices, or empty with floor_missing_ set.
                 *
                 * Each floor device takes chains in proportion to the nodes the plan gave it, at most as many
                 * as fit in a share of its memory.
                 */
                static std::vector<std::size_t> floor_homes()
                {
                    const SushiRuntime::PartitionReport& report = planned_.report;
                    const std::vector<SushiRuntime::DeviceDescription>& devices = runtime_->advanced().devices();
                    if (report.floor_devices.empty())
                    {
                        floor_missing_ = "the partition report names no floor devices.";
                        return {};
                    }
                    std::vector<std::size_t> indices;
                    std::vector<std::size_t> weights;
                    std::vector<std::size_t> capacities;
                    for (const SushiRuntime::DeviceKey& key : report.floor_devices)
                    {
                        const SushiRuntime::DeviceDescription* device = describe(devices, key);
                        if (device == nullptr)
                        {
                            floor_missing_ = "a floor device of the report is not a device of the runtime.";
                            return {};
                        }
                        indices.push_back(device->index);
                        weights.push_back(nodes_of(report, key));
                        capacities.push_back(static_cast<std::size_t>(device->memory_bytes / 100 * K_STATED_FILL_PERCENT /
                                                                      K_CHAIN_BYTES));
                    }
                    const std::vector<std::size_t> counts = apportion(chain_count_, weights, capacities);
                    if (counts.empty())
                    {
                        floor_missing_ = "the floor devices do not hold the chains in " +
                                         std::to_string(K_STATED_FILL_PERCENT) + " percent of their memory.";
                        return {};
                    }
                    std::vector<std::size_t> homes;
                    for (std::size_t device = 0; device < counts.size(); ++device)
                        homes.insert(homes.end(), counts[device], indices[device]);
                    floor_chains_ = counts;
                    floor_indices_ = indices;
                    return homes;
                }

                /** @brief Returns the nodes the report gives the device a key names. */
                static std::size_t nodes_of(const SushiRuntime::PartitionReport& report, const SushiRuntime::DeviceKey& key)
                {
                    for (const SushiRuntime::PartitionDeviceRow& row : report.rows)
                        if (row.device == key)
                            return row.nodes;
                    return 0;
                }

                /** @brief Prints the sizes, the runs of the runtime's placement and its report. */
                static void print_planned()
                {
                    for (const SushiRuntime::DeviceDescription& device : runtime_->advanced().devices())
                        print_number("device." + std::to_string(device.index) + "." + one_word(device.name),
                                     static_cast<double>(device.memory_bytes), "bytes");
                    print_number("largest_own_memory", static_cast<double>(largest_memory_), "bytes");
                    print_number("file", static_cast<double>(file_bytes_), "bytes");
                    print_number("chains", static_cast<double>(chain_count_), "chains");
                    for (std::size_t run = 0; run < planned_.step_seconds.size(); ++run)
                    {
                        const std::string prefix = "planned.run" + std::to_string(run);
                        print_number(prefix + ".step", planned_.step_seconds[run], "s");
                        print_number(prefix + ".file_copies", static_cast<double>(planned_.file_copies[run]), "copies");
                        print_number(prefix + ".file_bytes", static_cast<double>(planned_.file_bytes[run]), "bytes");
                        print_number(prefix + ".differing", static_cast<double>(planned_.differing[run]), "elements");
                    }
                    print_number("planned.best_step", planned_.best_step_seconds(), "s");
                    print_number("planned.modelled_step", planned_.report.step_seconds, "s");
                    print_number("planned.floor_devices", static_cast<double>(planned_.report.floor_devices.size()), "devices");
                    for (const SushiRuntime::PartitionDeviceRow& row : planned_.report.rows)
                    {
                        const std::string prefix = "planned.device" + device_name(row.device);
                        print_number(prefix + ".nodes", static_cast<double>(row.nodes), "nodes");
                        print_number(prefix + ".bytes", static_cast<double>(row.bytes), "bytes");
                        print_number(prefix + ".measured", row.measured_seconds, "s");
                    }
                    if (!planned_.failure.empty())
                        std::cout << "[t33] planned.failure " << one_word(planned_.failure) << " text" << std::endl;
                }

                /** @brief Prints the runs of the floor devices and the ratio of the T33 row. */
                static void print_floor()
                {
                    if (!floor_missing_.empty())
                    {
                        std::cout << "[t33] floor.missing " << one_word(floor_missing_) << " text" << std::endl;
                        return;
                    }
                    for (std::size_t device = 0; device < floor_indices_.size(); ++device)
                        print_number("floor.device" + std::to_string(floor_indices_[device]) + ".chains",
                                     static_cast<double>(floor_chains_[device]), "chains");
                    for (std::size_t run = 0; run < floor_.step_seconds.size(); ++run)
                    {
                        print_number("floor.run" + std::to_string(run) + ".step", floor_.step_seconds[run], "s");
                        print_number("floor.run" + std::to_string(run) + ".differing",
                                     static_cast<double>(floor_.differing[run]), "elements");
                    }
                    print_number("floor.best_step", floor_.best_step_seconds(), "s");
                    print_number("planned_over_floor", planned_.best_step_seconds() / floor_.best_step_seconds(), "ratio");
                    print_number("allowed_ratio", K_ALLOWED_RATIO, "ratio");
                    if (!floor_.failure.empty())
                        std::cout << "[t33] floor.failure " << one_word(floor_.failure) << " text" << std::endl;
                }

                /** @brief Returns the index of the device a key names as text, or "?" for an unknown device. */
                static std::string device_name(const SushiRuntime::DeviceKey& key)
                {
                    const SushiRuntime::DeviceDescription* device = describe(runtime_->advanced().devices(), key);
                    return device != nullptr ? std::to_string(device->index) : std::string("?");
                }

                /// @brief The suite's folder: the weight file and the profile file; under the build tree's tests folder.
                inline static std::filesystem::path directory_;

                /// @brief The runtime every test of the suite reads; made with a profile file.
                inline static std::optional<SushiRuntime::Runtime> runtime_;

                /// @brief What the machine lacks; empty when nothing.
                inline static std::string missing_;

                /// @brief What stopped the suite's measurement; empty when nothing did.
                inline static std::string failure_;

                /// @brief Memory of the largest GPU that has memory of its own, in bytes.
                inline static std::uint64_t largest_memory_ = 0;

                /// @brief Bytes of the weight file.
                inline static std::uint64_t file_bytes_ = 0;

                /// @brief Chains of the graph.
                inline static std::size_t chain_count_ = 0;

                /// @brief The runs on the runtime's own placement.
                inline static Measurement planned_;

                /// @brief The runs on the floor devices alone.
                inline static Measurement floor_;

                /// @brief Why the floor run was not made; empty when it was.
                inline static std::string floor_missing_;

                /// @brief Device index of each floor device the floor run used.
                inline static std::vector<std::size_t> floor_indices_;

                /// @brief Chains each of those devices held.
                inline static std::vector<std::size_t> floor_chains_;
        };

        /** @brief Skips the floor comparison, naming why the floor run was not made. */
        class Large_FileBackedProductFloor : public Large_FileBackedProduct
        {
            protected:
                /** @brief Skips the test when the floor run was not made. */
                void SetUp() override
                {
                    Large_FileBackedProduct::SetUp();
                    if (IsSkipped() || HasFatalFailure())
                        return;
                    if (!floor_missing().empty())
                        GTEST_SKIP() << floor_missing();
                }
        };

        /**
         * @brief Tests that a graph larger than any one device's memory runs on every device the plan gives a
         *        node: each has measured time in the partition report, and at least two hold nodes.
         */
        TEST_F(Large_FileBackedProduct, RunsOnEveryDeviceThePlanGivesANode)
        {
            ASSERT_TRUE(planned().failure.empty()) << planned().failure;
            ASSERT_FALSE(planned().report.refused) << planned().report.refusal;

            expect_every_holder_ran(planned().report);
        }

        /**
         * @brief Tests that the file is read by the first run, as the observer sees it, and by no later run.
         */
        TEST_F(Large_FileBackedProduct, ReadsNoFileAfterTheFirstRun)
        {
            ASSERT_TRUE(planned().failure.empty()) << planned().failure;
            ASSERT_EQ(planned().file_copies.size(), K_RUNS);
            ASSERT_GT(planned().file_copies.front(), 0u) << "the observer saw no read of the file in the first run";

            expect_no_file_read_after_the_first_run(planned().file_copies);
        }

        /**
         * @brief Tests that each of ten runs leaves the bytes the host computes, so the runs equal each other.
         */
        TEST_F(Large_FileBackedProduct, LeavesEqualBytesOverTenRuns)
        {
            ASSERT_TRUE(planned().failure.empty()) << planned().failure;
            ASSERT_EQ(planned().differing.size(), K_RUNS);

            expect_every_run_equals_the_host(planned().differing);
        }

        /**
         * @brief Tests that the planned step is at most 1.05 times the step with the floor plan's devices alone,
         *        each chain stated on one of them, the best of the runs after the first on both sides.
         */
        TEST_F(Large_FileBackedProductFloor, StepsNoLongerThanTheFloorDevicesAlone)
        {
            ASSERT_TRUE(planned().failure.empty()) << planned().failure;
            ASSERT_TRUE(floor_alone().failure.empty()) << floor_alone().failure;
            ASSERT_EQ(floor_alone().differing.size(), K_RUNS);
            expect_every_run_equals_the_host(floor_alone().differing);

            EXPECT_LE(planned().best_step_seconds(), K_ALLOWED_RATIO * floor_alone().best_step_seconds())
                << planned().report.to_text();
        }
    } // namespace Test
} // namespace SushiBLAS
