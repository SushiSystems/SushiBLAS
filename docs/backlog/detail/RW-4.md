# RW-4 Split `IO` into one reader and writer per format

The row of this entry is in `docs/backlog/GENERAL.md`. This file holds what did not fit the row.

- State: decision
- Done when: the owner has approved the boundary, and each format has a reader and a writer over a shared host-copy helper, with a load that returns a status (C4)
- Where: `modules/operations/io/include/SushiBLAS/io/io.hpp`, `modules/operations/io/source/io.cpp`

`io.cpp` holds three file formats in about 1000 lines; the split by format is what remains.
