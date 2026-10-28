# KI-1 A print, a save or a reference run can fault on Windows with CUDA

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: sushiruntime
- Upstream: sushiruntime KI-1
- Where: `modules/engine/engine/include/SushiBLAS/engine/kernel_pair_dispatcher.hpp`, `modules/operations/io/source/io.cpp`
- Recognise: the process stops with `0xC0000006` while another kernel is in flight; `ForceReference`, a print and a save touch `Residency::Shared` storage from the host. Untested here
- Workaround: finish the queue before a print, a save or a `ForceReference` run
