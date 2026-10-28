# KI-2 The reference path of the elementwise, nonlinear and logic seams checks launches, not formulas

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: own
- Where: `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp`, `modules/engine/functor_map/include/SushiBLAS/functor_map/map_task.hpp`
- Cause: it runs the same functor as the SYCL kernel in a serial loop
- Recognise: a wrong formula passes parity; a wrong launch or index fails it
- Workaround: test a formula against an independent value
- Fix: none, the shared functor is what keeps the two paths one text
