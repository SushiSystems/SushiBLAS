# 8. The SushiRuntime dependency (D3)

SushiBLAS does not vendor SushiRuntime. `cmake/Runtime.cmake` resolves it in three steps, the
pattern the whole stack uses:

1. **A target guard.** If a superproject already defines `SushiRuntime::SushiRuntime`, that
   target is used.
2. **`find_package(SushiRuntime QUIET)`.** An installed package.
3. **A sibling `add_subdirectory`.** `../sushiruntime` beside this checkout, or
   `$SUSHISTACK_HOME/sushiruntime` inside a workspace.

Whichever step succeeds, every consumer links the same namespaced target,
`SushiRuntime::SushiRuntime`. SushiBLAS inherits SushiRuntime's
`sushiruntime_deterministic_fp` contract through that link.
