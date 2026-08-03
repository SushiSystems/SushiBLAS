/**************************************************************************/
/* logger.hpp                                                             */
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

#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <sstream>
#include <utility>
#include <stdexcept>
#include <string_view>
#include <condition_variable>
#include <SushiBLAS/core/format.hpp>

#ifndef SB_ENABLE_LOGGING
    #define SB_ENABLE_LOGGING 1
#endif

/**
 * @def SB_FUNCTION_SIGNATURE
 * @brief The compiler's name for the function a log call sits in.
 *
 * The C++17 stand-in for `std::source_location::current().function_name()`, which is
 * where this used to come from. The pretty form rather than plain `__func__` on purpose:
 * at the dispatch seam every frame is a lambda inside a function template, and the bare
 * name is `operator()` for all of them - the signature is what tells one apart from
 * another, and it is what the level-choice rationale on @ref SB_ACTIVE_LOG_LEVEL is
 * costing out.
 */
#if defined(__clang__) || defined(__GNUC__)
    #define SB_FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
    #define SB_FUNCTION_SIGNATURE __FUNCSIG__
#else
    #define SB_FUNCTION_SIGNATURE __func__
#endif

/**
 * @def SB_ACTIVE_LOG_LEVEL
 * @brief The compiled-in log level, named as a `SushiBLAS::Core::LogType` enumerator.
 *
 * Set it from the build rather than by editing this header: the `SB_LOG_LEVEL` CMake
 * cache variable (`cmake/SushiBLASLogging.cmake`) forwards its value here, so
 * `sb build -D SB_LOG_LEVEL=INFO` raises the level for a debugging session and
 * `sb build -D SB_LOG_LEVEL=auto` puts it back. Valid values are the enumerator names:
 * `OFF`, `ERR`, `WARN`, `INFO`, `DEBUG` - a typo is a compile error naming the bad
 * enumerator rather than a silently wrong level.
 *
 * The default is WARN for a release build and DEBUG for a debug build. Release used to
 * default to INFO, which was expensive rather than merely noisy: every `SB_LOG_INFO`
 * that survives the `if constexpr` formats a @ref SushiBLAS::Core::Detail::SourceLocation
 * - file, line, and the fully templated function signature, several hundred characters at
 * the dispatch seam - and pushes the string through a mutex-guarded queue. The kernel
 * launch path logs at INFO once per launch, so a training loop paid that on every
 * operation of every step.
 */
#ifndef SB_ACTIVE_LOG_LEVEL
    #ifdef NDEBUG
        #define SB_ACTIVE_LOG_LEVEL WARN
    #else
        #define SB_ACTIVE_LOG_LEVEL DEBUG
    #endif
#endif

namespace SushiBLAS 
{
    namespace Core
    {
        /**
         * @brief Logging levels to filter messages.
         */
        enum class LogType : int 
        {
            OFF   = 0, /**< Logging is disabled */
            ERR   = 1, /**< Only error messages */
            WARN  = 2, /**< Warnings and errors */
            INFO  = 3, /**< Info, warnings and errors */
            DEBUG = 4  /**< All messages including debug */
        };

        /**
         * @brief The level messages are compiled in at, from @ref SB_ACTIVE_LOG_LEVEL.
         *
         * Every `SB_LOG_*` macro tests against this in an `if constexpr`, so anything above
         * the level costs nothing at all - not the formatting, not the source_location, not
         * the call.
         */
        #if SB_ENABLE_LOGGING
            inline constexpr LogType ACTIVE_LOG_LEVEL = LogType::SB_ACTIVE_LOG_LEVEL;
        #else
            inline constexpr LogType ACTIVE_LOG_LEVEL = LogType::OFF;
        #endif

        /**
         * @brief Internal tools for the logging system.
         */
        namespace Detail 
        {
            class AsyncLogger 
            {
                private:
                    std::queue<std::pair<LogType, std::string>> queue_;
                    std::mutex mutex_;
                    std::condition_variable cv_;
                    std::thread worker_;
                    std::atomic<bool> running_;

                    void process_logs() 
                    {
                        while (running_.load(std::memory_order_relaxed)) 
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            cv_.wait(lock, [this]() { return !queue_.empty() || !running_.load(std::memory_order_relaxed); });

                            while (!queue_.empty()) 
                            {
                                auto log_entry = std::move(queue_.front());
                                queue_.pop();
                                lock.unlock(); // Unlock while writing to file/terminal

                                FILE* stream = (log_entry.first <= LogType::ERR) ? stderr : stdout;

                                #ifdef _WIN32
                                    _lock_file(stream);
                                    fputs(log_entry.second.c_str(), stream);
                                    fflush(stream);
                                    _unlock_file(stream);
                                #else
                                    flockfile(stream);
                                    fputs(log_entry.second.c_str(), stream);
                                    fflush(stream);
                                    funlockfile(stream);
                                #endif

                                lock.lock(); // Lock again before checking queue
                            }
                        }
                    }

                public:
                    AsyncLogger() : running_(true) 
                    {
                        worker_ = std::thread(&AsyncLogger::process_logs, this);
                    }

                    ~AsyncLogger() 
                    {
                        running_.store(false, std::memory_order_relaxed);
                        cv_.notify_one();

                        if (worker_.joinable()) 
                        {
                            worker_.join();
                        }
                    }

                    void enqueue(LogType level, std::string&& msg) 
                    {
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            queue_.emplace(level, std::move(msg));
                        }
                        cv_.notify_one();
                    }
            };

            class SyncLogger 
            {
                private:
                    std::mutex sync_mutex_;
                public:
                    void enqueue(LogType level, std::string&& msg) 
                    {
                        std::lock_guard<std::mutex> lock(sync_mutex_);
                        FILE* stream = (level <= LogType::ERR) ? stderr : stdout;
                        #ifdef _WIN32
                            _lock_file(stream);
                            fputs(msg.c_str(), stream);
                            fflush(stream);
                            _unlock_file(stream);
                        #else
                            flockfile(stream);
                            fputs(msg.c_str(), stream);
                            fflush(stream);
                            funlockfile(stream);
                        #endif
                    }
            };

            inline std::atomic<bool>& is_sync_logging_enabled()
            {
                static std::atomic<bool> mode{false};
                return mode;
            }

            inline void dispatch_log(LogType level, std::string&& msg) 
            {
                if (is_sync_logging_enabled().load(std::memory_order_relaxed)) 
                {
                    static SyncLogger sync_logger;
                    sync_logger.enqueue(level, std::move(msg));
                } 
                else 
                {
                    static AsyncLogger async_logger;
                    async_logger.enqueue(level, std::move(msg));
                }
            }

            /**
             * @brief Gets local time in a platform-independent way.
             * @param result The tm structure to fill.
             * @param time The timestamp to convert.
             */
            inline void portable_localtime(std::tm* result, const std::time_t* time)
            {
                #ifdef _WIN32
                    localtime_s(result, time);
                #else
                    localtime_r(time, result);
                #endif
            }

            /**
             * @brief Creates a timestamp with millisecond precision.
             * @return Formatted time string (HH:MM:SS.mmm).
             */
            inline std::string get_timestamp()
            {
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                std::tm tm;
                portable_localtime(&tm, &time);

                // add milliseconds for better tracking in async tasks
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

                // snprintf rather than the library's own formatter: this is the one format
                // string in the codebase that runs on every single log line, and its shape
                // never varies, so there is nothing to gain from re-parsing it each time.
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                              tm.tm_hour, tm.tm_min, tm.tm_sec,
                              static_cast<int>(ms.count()));
                return std::string(buf);
            }

            /**
             * @brief Where a log call was written.
             *
             * The C++17 stand-in for `std::source_location`, which is C++20. Deliberately
             * shaped like it - same three accessors, same names - so that the day this
             * stack can assume C++20 again the type is deleted and `std::source_location`
             * dropped in with no other edit. It is filled by @ref SB_CURRENT_LOCATION,
             * which is what `std::source_location::current()` was doing underneath anyway:
             * `__FILE__`, the function signature, and `__LINE__`.
             */
            class SourceLocation
            {
                public:
                    /** @brief Constructs an empty location. */
                    constexpr SourceLocation() noexcept = default;

                    /**
                     * @brief Constructs a location from a call site's own macros.
                     * @param file The source file, from `__FILE__`.
                     * @param function The enclosing function, from @ref SB_FUNCTION_SIGNATURE.
                     * @param line The line number, from `__LINE__`.
                     */
                    constexpr SourceLocation(const char* file, const char* function, int line) noexcept
                        : file_(file), function_(function), line_(line) {}

                    /**
                     * @brief The source file the call was written in.
                     * @return A null-terminated file name; never null.
                     */
                    [[nodiscard]] constexpr const char* file_name() const noexcept { return file_; }

                    /**
                     * @brief The function the call was written in.
                     * @return A null-terminated function signature; never null.
                     */
                    [[nodiscard]] constexpr const char* function_name() const noexcept { return function_; }

                    /**
                     * @brief The line the call was written on.
                     * @return The line number, or 0 for a default-constructed location.
                     */
                    [[nodiscard]] constexpr int line() const noexcept { return line_; }

                private:
                    const char* file_     = "";
                    const char* function_ = "";
                    int         line_     = 0;
            };

            template<typename... Args>
            void log_base(LogType level, std::string_view prefix, const SourceLocation& loc, std::string_view fmt, const Args&... args)
            {
                #if SB_ENABLE_LOGGING
                    // The message goes through the format helper; the line around it does
                    // not. Its shape is fixed, so streaming it directly saves parsing a
                    // second format string on every line that survives the level test.
                    std::ostringstream os;
                    os << '[' << get_timestamp() << "] ["
                       << prefix << "] ["
                       << loc.file_name() << ':' << loc.line() << "] ["
                       << loc.function_name() << "] "
                       << format(fmt, args...) << '\n';

                    dispatch_log(level, os.str());
                #endif
            }

            /**
             * @brief The level test every `SB_LOG_*` macro goes through.
             *
             * A function template rather than an `if constexpr` spelled out in the macro,
             * for two reasons.
             *
             * First, arguments stay used. The macros used to expand to
             * `if constexpr (level ok) log_base(..., args...)`; inside a template, a
             * discarded statement is never instantiated, so lowering the compiled-in level
             * turned every variable a log line was the sole reader of into an unused one -
             * `-Wunused-lambda-capture` under `-Werror` - and the fix would have been to
             * scatter `(void)x;` through the call sites. Passing the arguments to a function
             * keeps them odr-used whatever the level, so the level is a pure logging decision
             * and never a compile error somewhere else.
             *
             * Second, it is an expression, not a statement, so `if (c) SB_LOG_INFO(...);
             * else ...` binds the `else` where the reader expects it instead of to the
             * macro's own hidden `if`.
             *
             * Everything above the active level still costs nothing at run time: the
             * discarded branch leaves an empty function body that inlines away.
             *
             * @tparam Level The level this message is emitted at.
             * @param prefix The bracketed tag written into the line ("INFO", "WARN", ...).
             * @param loc The call site.
             * @param fmt The format string; see <SushiBLAS/core/format.hpp> for the grammar.
             * @param args The format arguments.
             */
            template<LogType Level, typename... Args>
            inline void log_if(std::string_view prefix, const SourceLocation& loc, std::string_view fmt, Args&&... args)
            {
                if constexpr (ACTIVE_LOG_LEVEL >= Level)
                {
                    log_base(Level, prefix, loc, fmt, std::forward<Args>(args)...);
                }
                else
                {
                    (void)prefix;
                    (void)loc;
                    (void)fmt;
                    ((void)args, ...);
                }
            }
        } // namespace Detail
    } // namespace Core

    /**
     * @def SB_CURRENT_LOCATION
     * @brief The call site, as a @ref SushiBLAS::Core::Detail::SourceLocation.
     *
     * The C++17 replacement for `std::source_location::current()`. Costs nothing that
     * `current()` did not: all three components are compile-time constants, and the level
     * test in `log_if` discards them when the message is compiled out.
     */
    #define SB_CURRENT_LOCATION \
        SushiBLAS::Core::Detail::SourceLocation(__FILE__, SB_FUNCTION_SIGNATURE, __LINE__)

    /**
     * @brief Logs a message at error level.
     */
    // The format string is folded into __VA_ARGS__ - the macros take (...) rather than
    // (fmt, ...) - for two reasons, both C++17 rules that C++20 relaxed. It avoids
    // __VA_OPT__, which is C++20; and it keeps the variadic list non-empty for a call
    // with no arguments at all (`SB_LOG_INFO("done")`), which under `(fmt, ...)` would
    // pass zero arguments for `...` and trip -Wpedantic. `log_if`'s `std::string_view fmt`
    // parameter binds the first argument, so every existing call site is unchanged.
    #if SB_ENABLE_LOGGING
        #define SB_LOG_ERROR(...) \
            SushiBLAS::Core::Detail::log_if<SushiBLAS::Core::LogType::ERR>("ERROR", SB_CURRENT_LOCATION, __VA_ARGS__)

        /** @brief Logs a message at warning level. */
        #define SB_LOG_WARN(...) \
            SushiBLAS::Core::Detail::log_if<SushiBLAS::Core::LogType::WARN>("WARN", SB_CURRENT_LOCATION, __VA_ARGS__)

        /** @brief Logs a message at info level. */
        #define SB_LOG_INFO(...) \
            SushiBLAS::Core::Detail::log_if<SushiBLAS::Core::LogType::INFO>("INFO", SB_CURRENT_LOCATION, __VA_ARGS__)

        /** @brief Logs a message at debug level. */
        #define SB_LOG_DEBUG(...) \
            SushiBLAS::Core::Detail::log_if<SushiBLAS::Core::LogType::DEBUG>("DEBUG", SB_CURRENT_LOCATION, __VA_ARGS__)
    #else
        #define SB_LOG_ERROR(...) ((void)0)
        #define SB_LOG_WARN(...)  ((void)0)
        #define SB_LOG_INFO(...)  ((void)0)
        #define SB_LOG_DEBUG(...) ((void)0)
    #endif

    /** @brief Switches the logger to synchronous block-printing mode (useful for tests). */
    #define SB_LOGGER_SET_SYNC_MODE(mode) \
        SushiBLAS::Core::Detail::is_sync_logging_enabled().store(mode, std::memory_order_release)

    /**
     * @brief Custom assertion macro.
     */
    #ifdef NDEBUG
        #define SB_ASSERT(condition, msg) ((void)0)
    #else
        #define SB_ASSERT(condition, msg) \
            do { \
                if (!(condition)) { \
                    SB_LOG_ERROR("Assertion failed: {} ({}:{})", msg, __FILE__, __LINE__); \
                    std::abort(); \
                } \
            } while (0)
    #endif

    /**
     * @brief Throws an exception and logs it if a condition is met.
     * @param condition The condition to check.
     * @param ... The error message format string followed by its arguments.
     */
    #define SB_THROW_IF(condition, ...) \
        do { \
            if (condition) { \
                std::string msg_ = SushiBLAS::Core::Detail::format(__VA_ARGS__); \
                SB_LOG_ERROR("Runtime exception: {}", msg_); \
                throw std::runtime_error(msg_); \
            } \
        } while (0)

} // namespace SushiBLAS
