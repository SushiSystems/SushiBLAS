# SushiBLASSanitizers.cmake — the project's sanitizer policy as a target.
#
# Single responsibility: define an INTERFACE target `sushiblas_sanitizers`.
# Currently empty (no sanitizer is auto-enabled by build type here); it exists
# so BuildPolicy.cmake and every target can link it unconditionally, and a
# sanitizer lane can be added later without touching every target's
# CMakeLists.txt — mirrors sushiruntime/cmake/Sanitizers.cmake's shape.

add_library(sushiblas_sanitizers INTERFACE)
