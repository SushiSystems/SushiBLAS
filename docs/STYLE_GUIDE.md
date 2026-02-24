# SushiBLAS Style Guide

This project follows a consistent coding style to ensure readability and maintainability.

## License Header

Every file must start with the following license signature. Replace `[filename]` with the actual name of the file.

```cpp
/**************************************************************************/
/* [filename]                                                             */
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
```

## C++ Coding Standards

*   **Standard**: C++20.
*   **Indentation**: 4 spaces.
*   **Naming Conventions**:
    *   **Classes/Structs**: `PascalCase` (e.g., `Engine`, `Tensor`).
    *   **Functions/Methods**: `snake_case` (e.g., `create_tensor`, `execute`).
    *   **Variables**: `snake_case` (e.g., `num_elements`).
    *   **Member Variables**: `snake_case_` for private members.
    *   **Namespaces**: `PascalCase` (e.g., `SushiBLAS`, `Core`).

### Braces Style

Curly braces `{}` are mandatory only for multi-statement blocks and must follow the Allman style (alone on a new line). For single-statement conditions or loops, braces must be omitted:
1. If the statement is short, it can stay on the same line.
2. If the statement is long or for better readability, it should be on a new line with indentation.

**Correct Examples:**
```cpp
// Correct: Short single-line (short)
if (condition) return;

// Correct: Single statement on a new line (longer)
if (very_long_condition_that_requires_careful_checking)
    execute_specific_logic_for_this_unique_case();

// Correct: Multi-statement block uses Allman style braces
if (condition)
{
    do_first();
    do_second();
}
```

**Wrong Example:**
```cpp
// Wrong: Braces are NOT allowed for single statements
if (condition)
{
    do_something();
}
```

### Include and Declaration Ordering

1.  `#pragma once` must be at the very top.
2.  Follow with exactly one empty line.
3.  **Group 1: `<>` Includes** (System headers).
    *   Sort by line length: **Shortest to Longest**.
    *   If length is the same, use **Alphabetical** order.
4.  **Group 2: `""` Includes** (Local headers).
    *   Sort by line length: **Longest to Shortest**.
    *   If length is the same, use **Alphabetical** order.

**Example Order:**
```cpp
#pragma once

#include <cassert>
#include <cstddef>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/SushiRuntime.h>
```

## Documentation

*   Use Doxygen-style comments for all public APIs.
*   Write comments in **Simple International English**.
*   Use `@brief` for short descriptions.
*   Use `@param` for parameters and `@return` for return values.

## Comments and TODOs

*   **TODO Comments**: Use `// TODO:` for tasks that need to be completed or improved later.
*   **Placement**: Always place `TODO` comments **above** the code they refer to. Do not put them on the same line as code.

**Correct Example:**
```cpp
// TODO: Implement support for half-precision floats
void process_data() { ... }
```

## Error Handling

*   Use the `SB_THROW_IF(condition, message, ...)` macro.
*   Use exceptions for errors that can be fixed. Use assertions for internal bugs.
