# RW-3 Remove `core/export.hpp` and the `<numbers>` include

The row of this entry is in `docs/backlog/GENERAL.md`. This file holds what did not fit the row.

- State: decision
- Done when: neither is in the tree (C13)
- Where: `modules/operations/nonlinear/source/gelu.cpp`

`gelu.cpp` includes a C++20 header in a C++17 library. A strict C++17 compiler refuses the file.
Cause: it includes `<numbers>`.
