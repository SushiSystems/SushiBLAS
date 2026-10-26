# Benchmarks

Scenario benchmarks: programs that measure one subsystem, print numbers and decide nothing. They are not tests and never gate a change. The micro benchmarks that sit beside the unit they time are under `tests/`, and `sb bench` runs them.

| Program | Measures |
| --- | --- |
| [`device_ceilings`](device_ceilings/README.md) | The compute rate per dtype, the memory bandwidth and the launch floor of one device, and writes them as a ceilings file |

Build with `sb build --benchmarks`, then run one with `sb bench <name> -- <options>`.
