# NIXL Backport Tracking Document

This document tracks the backporting of NIXL commits from upstream
(~/work/nixl) to AsyncRT/lib/NIXL.

**Initial HackToOwn**: May 12, 2025 (commit c8a071df22)
**Tracking Period**: May 1 - July 10, 2025

## Week 1 (May 1-5, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| 3491e36 | 2025-05-01 | python: minor test improvement | Skip | - | Python API changes, not relevant to C++ AsyncRT |
| c4c1c0b | 2025-05-01 | POSIX: merge posix plugin to main | Skip | - | POSIX backend not included in AsyncRT |
| 9978771 | 2025-05-02 | meson: don't require liburing | Skip | - | Build system change (Meson), AsyncRT uses Bazel |
| 197a286 | 2025-05-03 | CI: disable POSIX CI test | Skip | - | CI configuration, not relevant |
| 5759122 | 2025-05-02 | README: Clarify hostname for etcd server | Skip | - | Documentation change |
| 242e4ea | 2025-05-04 | refactor: Prefer "using" over "typedef" | **Port** | ✅ Completed | Code style improvement affecting core files |
| dc04cc8 | 2025-05-05 | NIXLBench: Add ctrl-c handling | Skip | - | Benchmarking tool, not in AsyncRT |

### Commits to Port

#### 1. 242e4ea - refactor: Prefer "using" over "typedef"

**Files to update in AsyncRT/NIXL**:

- `nixl_descriptors.h` - ✅ Already uses `using` syntax
- `nixl_types.h` - ✅ Already uses `using` syntax
- `agent_data.h` - ✅ Already uses `using` syntax
- `nixl_time.h` - ❌ Still has typedef on lines 20-23, needs update
- `ucx_utils.h` - ✅ No typedefs found

**Action**: Only `nixl_time.h` needs to be updated to convert typedef to using:

- Line 20: `typedef uint64_t ns_t;` → `using ns_t = uint64_t;`
- Line 21: `typedef uint64_t us_t;` → `using us_t = uint64_t;`
- Line 22: `typedef uint64_t ms_t;` → `using ms_t = uint64_t;`
- Line 23: `typedef uint64_t sec_t;` → `using sec_t = uint64_t;`

### Summary

- **Total commits reviewed**: 7
- **Commits to port**: 1
- **Commits ported**: 1 ✅
- **Commits skipped**: 6

### Changes Made

1. **242e4ea**: Updated `AsyncRT/include/AsyncRT/NIXL/utils/common/nixl_time.h`
   to use modern C++ `using` syntax instead of `typedef` for type aliases
   (ns_t, us_t, ms_t, sec_t).

### Testing

#### Week 2

- Added ReceiverFailure regression test to NixlTestUcx.cpp

### Reasons for skipping commits

- Python API changes (not relevant to C++ AsyncRT)
- POSIX backend (not included in initial port)
- Build system differences (Meson vs Bazel)
- CI/documentation changes
- NIXLBench (benchmarking tool not included)

## Week 2 (May 6-12, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| 3298d77 | 2025-05-06 | python: don't require sleep in partial md test | Skip | - | Python API changes |
| 436273e | 2025-05-07 | Extend nixl_test with multi-threading | Skip | - | Test framework changes, AsyncRT has own tests |
| a0748ea | 2025-05-07 | refactor: updates for kv block manager | Skip | - | KV block manager not in AsyncRT scope |
| 620bc45 | 2025-05-07 | POSIX: Add more logs | Skip | - | POSIX backend not included |
| 7fa5723 | 2025-05-07 | UCX: add remote failure error handling | **Port** | ✅ Completed | UCX error handling improvements |
| 6f0e405 | 2025-05-08 | CI: Change default runner labels | Skip | - | CI configuration |
| 16dd334 | 2025-05-07 | UCX: Only send request notification when requests have completed | **Port** | ✅ Completed | UCX performance optimization |
| 48ff7bf | 2025-05-08 | chore: update CODEOWNERS file | Skip | - | Project maintenance |
| 99d4436 | 2025-05-07 | NIXLBench: Bug fixes | Skip | - | Benchmarking tool not included |
| 9135a18 | 2025-05-08 | python: use torch.set_default_device | Skip | - | Python API changes |
| c5311a0 | 2025-05-07 | TEST: fix improper rebase in nixl_test | Skip | - | Test framework changes |
| 7e5328d | 2025-05-08 | UCX: Fix potential memory leak with genNotif() | **Port** | ✅ Completed | UCX memory leak fix |
| 6c1db45 | 2025-05-08 | posix: Updated implementation | Skip | - | POSIX backend not included |
| 1705e82 | 2025-05-08 | pytest: add more fixtures | Skip | - | Python test infrastructure |
| 16227e2 | 2025-05-08 | Refactor UCX code to comply with smart pointers | **Port** | ✅ Completed | UCX smart pointer refactoring |
| f147b1f | 2025-05-08 | TEST: Add data transfer test in gtest | Skip | - | Test framework, AsyncRT has own tests |
| 9545895 | 2025-05-08 | build: Miscellanous fixes | Skip | - | Build system (Meson) |
| ef85fd3 | 2025-05-08 | Agent: small fixes to commWorker | Skip | - | Not UCX-specific, need to check relevance |
| 0155d4e | 2025-05-09 | Add Mooncake backend in NIXL | Skip | - | Mooncake backend not included |
| fc91b32 | 2025-05-08 | Python: dependency and GDS example | Skip | - | Python API changes |
| f0429ba | 2025-05-08 | Agent API: new param metadataLabel | Skip | - | API changes, need to check if applicable |
| a82ffa4 | 2025-05-08 | Docker build and test fixes | Skip | - | Docker infrastructure |
| 9f06b7e | 2025-05-08 | README: link to plugin requirements | Skip | - | Documentation |
| 791ec60 | 2025-05-08 | chore: bump nixl versions to 0.2.1 | Skip | - | Version management |
| bcbb42f | 2025-05-08 | Dockerfile for building EFA wheel | Skip | - | Docker/EFA infrastructure |
| a55250b | 2025-05-09 | UCX: Fix worker address query to enable cuda-ipc | **Port** | ✅ Completed | UCX CUDA IPC fix |
| 4f73441 | 2025-05-09 | fix: add crate metadata for publishing | Skip | - | Rust/crate metadata |
| ca87709 | 2025-05-11 | CI: Add AWS EFA testing infrastructure | Skip | - | CI infrastructure |
| e8257bb | 2025-05-12 | BUILD/TEST: Fix compilation on RedHat8 | Skip | - | Build system specific to upstream |
| 9ea7db2 | 2025-05-12 | Extend UCX engine with multi worker support | **Port** | ✅ Completed | UCX multi-worker support |

### Commits to Port (6 total)

1. **7fa5723** - UCX: add remote failure error handling
2. **16dd334** - UCX: Only send request notification when requests have completed
3. **7e5328d** - UCX: Fix potential memory leak with genNotif()
4. **16227e2** - Refactor UCX code to comply with smart pointers
5. **a55250b** - UCX: Fix worker address query to enable cuda-ipc
6. **9ea7db2** - Extend UCX engine with multi worker support

### Summary

- **Total commits reviewed**: 30
- **Commits to port**: 6 (all UCX-related improvements)
- **Commits ported**: 6 ✅
- **Commits skipped**: 24

### Implementation Notes

The UCX commits from Week 2 introduce significant improvements:

- Error handling and state management for endpoints
- Memory leak fixes
- Smart pointer refactoring for better memory safety
- Performance optimizations
- Multi-worker support for better scalability
- CUDA IPC fixes

These changes are interconnected and build upon each other. The commits
should be ported carefully, ensuring compatibility with AsyncRT's error
handling patterns (ErrorOr vs exceptions) and code style.

### Changes Made

#### 1. 7fa5723 - UCX: add remote failure error handling

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/nixl_types.h`: Added NIXL_ERR_REMOTE_DISCONNECT
  error code
- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Added endpoint state
  management and error handling to nixlUcxEp class
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Implemented error callbacks,
  state management, and updated connect/disconnect methods

**Key changes**:

- Added endpoint state tracking (NULL, CONNECTED, FAILED, DISCONNECTED)
- Implemented UCX error callback for handling remote disconnections
- Added proper error mapping from UCX to NIXL status codes
- Refactored connect/disconnect methods to use new nixlUcxEp interface
- Replaced direct `ep.eph` access with `ep.getHandle()` method

**Adaptations for AsyncRT**:

- Used `assert` instead of `NIXL_ASSERT` macro.
- Maintained compatibility with existing NIXL error handling patterns.
- Added UCS_ERR_NOT_CONNECTED and UCS_ERR_ENDPOINT_TIMEOUT to error mapping
  (from later commit 92a4bbd).
- Fixed nixlUcxWorker::test() to use ucx_status_to_nixl instead of always
  returning NIXL_ERR_BACKEND.
- Fixed nixlUcxWorker::flushEp() to use ucx_status_to_nixl for error cases.
- Added regression test ReceiverFailure to verify remote disconnect handling.

#### 2. 16dd334 - UCX: Only send request notification when requests have completed

- Added deferred notification support to nixlUcxBackendH using std::optional
- Modified postXfer/checkXfer to send notifications only after flush completes

#### 3. 7e5328d - UCX: Fix potential memory leak with genNotif()

- Changed nixlUcxIntReq::amBuffer from raw pointer to std::unique_ptr<std::string>
- Updated notifSendPriv to use std::make_unique and std::move

#### 4. 16227e2 - Refactor UCX code to comply with smart pointers

- Converted UCX context/worker to std::unique_ptr in nixlUcxEngine
- Changed epAddr() and packRkey() to return std::unique_ptr<char[]>
- Removed all manual memory management in favor of RAII

#### 5. a55250b - UCX: Fix worker address query to enable cuda-ipc

- Removed UCP_WORKER_ADDRESS_FLAG_NET_ONLY flag from epAddr()
- Enables CUDA IPC transport for GPU-to-GPU transfers

#### 6. 9ea7db2 - Extend UCX engine with multi worker support

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/plugins/ucx/ucx_backend.h`: Updated data structures
  for multi-worker support
- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Major refactoring to move
  operations to appropriate classes
- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Multi-worker implementation
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Refactored implementation
- `AsyncRT/unittests/NixlTestUcx.cpp`: Updated tests for new API

**Key changes**:

- Refactored nixlUcxEp to be created via constructor instead of connect()
  method.
- Moved memory management operations (memReg, packRkey, memDereg) from Worker
  to Context.
- Moved data transfer operations (sendAm, read, write, flushEp) from Worker to
  Ep.
- Added support for multiple workers configured via 'num_workers' parameter.
- Workers are stored in a vector and selected based on thread ID hash.
- Connection establishment now happens on all workers.
- Converted nixlUcxContext to shared_ptr for proper lifetime management.
- Updated all tests to use ErrorOr pattern instead of raw pointers.

**Adaptations for AsyncRT**:

- Converted all designated initializers to memset + field assignment for C++ compatibility
- Used ErrorOr instead of absl::StatusOr for return types
- Maintained backward compatibility by defaulting to single worker

## Week 3 (May 13-19, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| 05b5f61 | 2025-05-13 | Python: Reimplement GDRAPI / kvlib module | Skip | - | Python bindings, not relevant to C++ AsyncRT |
| 26d6c8a | 2025-05-13 | NVLS: add endpoint estimation helpers | Skip | - | NVLS backend not included in AsyncRT |
| f3e9b42 | 2025-05-13 | Agent: enable cross-process KV block allocs | Skip | - | KV block manager not in AsyncRT scope |
| cf17e23 | 2025-05-13 | CI: use iam-role for ecr | Skip | - | CI configuration |
| 54b1aac | 2025-05-13 | UCX: Check context before create endpoint | **Port** | ✅ Completed | UCX initialization safety |
| a1b82f8 | 2025-05-13 | Agent: Fix python bindings for KV allocation | Skip | - | Python bindings |
| 498fe2a | 2025-05-13 | UCX: Set endpointCount to 0 instead of hardcoded number | **Port** | ✅ Completed | UCX endpoint estimation fix |
| 1c456a5 | 2025-05-13 | CI: Make containers reusable | Skip | - | CI configuration |
| 77f0a51 | 2025-05-13 | NVLS: rename backend files | Skip | - | NVLS backend not included |
| a00d7d9 | 2025-05-13 | NVLS backend initial implementation | Skip | - | NVLS backend not included |
| 92a4bbd | 2025-05-13 | UCX: add UCS_ERR_NOT_CONNECTED in error map | Skip | - | Already incorporated in Week 2 changes |
| 7b4f73f | 2025-05-13 | Python: allow changing default port | Skip | - | Python API changes |
| e9b9302 | 2025-05-13 | Example: update etcd example | Skip | - | Example code |
| b9a3ee5 | 2025-05-13 | Agent: getLocalMD should add sorted flag | Skip | - | API changes, need to check relevance |
| b0a9bb7 | 2025-05-14 | fix: add test check that non-sorted flag is populated | Skip | - | Test framework |
| d36b8de | 2025-05-14 | NVLS: finish bring up | Skip | - | NVLS backend not included |
| ebdb96d | 2025-05-14 | nixl_test: encapsulate gtest names in anon namespace | Skip | - | User noted preference for static functions |
| 48646a9 | 2025-05-14 | UCX: make data path const for thread safe | **Port** | ✅ Completed | Thread safety improvements |
| 957448e | 2025-05-14 | UCX: remove copy and move semantics | Skip | - | User noted already incorporated |
| c7e396f | 2025-05-14 | TEST: Add missing data-structure tests | Skip | - | Test framework |
| d70fa87 | 2025-05-15 | UCX: Refactor progress thread | **Port** | ✅ Completed | Progress thread optimization |
| 4f37f07 | 2025-05-16 | UCX: Add ucx worker to handle | **Port** | ✅ Completed | UCX error handling improvements |
| 1a4a6ae | 2025-05-16 | UCX: fix multi GPU scenario | Skip | - | Not applicable - AsyncRT uses DeviceContext |
| 6f6bfb9 | 2025-05-16 | nixl_test: reduce sleep overhead | Skip | - | Test framework |
| 96f6aee | 2025-05-16 | test: fix test correctness | Skip | - | Test framework |
| 3d88c8f | 2025-05-16 | NVLS: extend test to target >=2 pairs | Skip | - | NVLS backend tests |
| c8a071d | 2025-05-16 | h2o fork | Skip | - | Already at this commit (initial HackToOwn) |

### Commits to Port

1. **54b1aac + 498fe2a** - UCX initialization and endpoint fixes
   (✅ Completed - cd7d019a2)
2. **48646a9** - UCX: make data path const for thread safe
   (✅ Completed - amended to cd7d019a2)
3. **d70fa87** - UCX: Refactor progress thread (✅ Completed)
4. **4f37f07** - UCX: Add ucx worker to handle (✅ Completed)

### Summary

- **Total commits reviewed**: 27
- **Commits to port**: 4 (all UCX-related improvements)
- **Commits ported**: 4 ✅
- **Commits skipped**: 23

### Changes Made

#### 3. d70fa87 - UCX: Refactor progress thread

**Files modified**:

- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Replaced busy-polling with
  poll()
- `AsyncRT/include/AsyncRT/NIXL/plugins/ucx/ucx_backend.h`: Replaced atomic
  bool with control pipe
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Added UCP_FEATURE_WAKEUP support
- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Added getWorker()
  method

**Key changes**:

- Replaced busy-polling loop with poll()-based event-driven approach
- Added control pipe for clean thread cancellation
- Enable UCP_FEATURE_WAKEUP when progress thread is enabled
- Monitor UCX worker file descriptors with poll()

#### 4. 4f37f07 - UCX: Add error handling mode parameter

**Files modified**:

- `AsyncRT/lib/NIXL/plugins/ucx/ucx_plugin.cpp`: Added
  ucx_error_handling_mode parameter
- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Parse error handling mode
  parameter
- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Updated class
  signatures
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Implemented error handling mode
  support
- `AsyncRT/unittests/NixlTestUcx.cpp`: Added comprehensive error handling
  tests

**Key changes**:

- Added configurable error handling mode ("none" or "peer")
- Updated nixlUcxContext to store and pass error handling mode
- Updated nixlUcxEp constructor to use configured error handling mode
- Simplified closeImpl() by removing blocking behavior
- Fixed disconnectNonblocking() to ignore remote disconnect errors
- Added comprehensive error handling tests:
  - XferThenFail for both WRITE and READ operations
  - Basic READ operation test
  - Explicit peer mode configuration test
