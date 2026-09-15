# Tacker runtime library

`libtacker_runtime` is the reusable scheduling core extracted from the original
Tacker benchmark executable.  It schedules two independent `TaskGraph`
submissions, pairs ready GPTB nodes through a `MixedKernelRegistration`, and
advances both graphs from the mixed kernel's single completion event.

Mixed pairs are not selected from submission roles or leaf-level QoS
heuristics.  A pair may have multiple physical registrations, and the default
fail-closed scheduler dispatches only the registration whose `ProfileRecord`
is enabled, correctness-valid, explicitly selected for deployment, bound to a
non-empty manifest hash, and backed by a finite positive whole-sequence
`median_throughput_fps`.  `primary_slowdown_percent`, `mixed_p50_us`, and
estimated leaf savings remain diagnostics; the deprecated
`SchedulerOptions::max_primary_slowdown_percent` and `minimum_savings_us`
members are retained only for source compatibility and never reject an
offline-selected candidate.  Opaque CUDA-library operations such as CUB scan
or radix sort remain graph nodes but are never inlined into a mixed kernel.

When `require_profile=false`, the runtime may execute an unprofiled
registration for qualification/testing, but that mode is not a deployment
admission path.  Production callers should keep the default `true` value and
load exactly one selected profile for each logical pair.  Swapping that
selection/profile is therefore sufficient to roll back without recompiling or
performing an online search.

## Build and test without CUDA

```bash
cmake -S Tacker/src -B build/tacker-runtime \
  -DTACKER_BUILD_LEGACY=OFF \
  -DTACKER_BUILD_TESTS=ON
cmake --build build/tacker-runtime
ctest --test-dir build/tacker-runtime --output-on-failure
```

CUDA is detected rather than required.  When it is available,
`CudaExecutionBackend` is included and supports both CUDA runtime function
pointers and driver `CUfunction` handles.  Device architecture, SM count,
thread limits, and shared-memory limits are queried at runtime; no GPU count is
hard-coded.

## Install and consume

```bash
cmake --install build/tacker-runtime --prefix /path/to/prefix
```

```cmake
find_package(TackerRuntime CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Tacker::runtime)
```

Include the aggregate header with `#include <runtime/Runtime.h>`.  A PyTorch
extension must either compile these sources itself or use a runtime library
built with the same libstdc++ ABI as PyTorch.
