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

## Week 4 (May 20-26, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| 4f37f07 | 2025-05-19 | UCX/BACKEND: add custom error handling parameter + gtests improvements | Skip | - | Already backported in Week 3 |
| 1b60b99 | 2025-05-19 | plugins/posix: Fix build on RHEL8/Manylinux | Skip | - | POSIX backend not included |
| 1a4a6ae | 2025-05-20 | UCX: fix multi GPU scenario | Skip | - | Already analyzed - not applicable to AsyncRT |
| 9f00c7c | 2025-05-20 | Benchmark: make it possible to build container with UCX built from source | Skip | - | Benchmarking infrastructure |
| a87c086 | 2025-05-21 | Implement new synchronization read-write locking mode | Skip | - | Not UCX-specific, likely ETCD related |
| d8ffa6a | 2025-05-21 | set multi flag for read/write in ucx backend | **Port** | ✅ Completed | UCX read/write operation improvements |
| 47c9d46 | 2025-05-21 | BACKEND/UCX_MO: add ucx_error_handling_mode parameter | Skip | - | UCX_MO backend not included |
| 9cb30a6 | 2025-05-21 | UCX: handle ucp_worker_create failure | **Port** | ✅ Completed | UCX error handling improvement |
| 846e043 | 2025-05-21 | Refactor: Some C++ best practices | **Port** | ✅ Completed | Code quality improvements |
| 598ae34 | 2025-05-21 | UCX: Set a CUDA context when loading remote MD from listener thread | **Port** | ✅ Completed | Already present in AsyncRT via activateContext() |
| 2e9262f | 2025-05-21 | Use codespellrc file to ignore words | Skip | - | Spelling/linting configuration |
| d18a757 | 2025-05-22 | Agent: Bug fix: apply timeout to socket connection and add gtests for md exchange | Skip | - | Agent/socket timeouts, not UCX specific |
| 1144491 | 2025-05-22 | Agent: Improve MD fetch and invalidate flow in ETCD mode using watchers | Skip | - | ETCD functionality removed |
| 7c87207 | 2025-05-22 | Posix: Updated Uring Queue implementation | Skip | - | POSIX backend not included |
| af9ccc4 | 2025-05-22 | plugins/hf3fs: Add 3FS backend | Skip | - | 3FS backend not included |
| ace79e7 | 2025-05-21 | Dockerfile: Add RDMA libraries for UCX build | Skip | - | Docker/build infrastructure |
| d01a854 | 2025-05-22 | Timeout from poll() in progress thread after pthrDelay | **Port** | ✅ Completed | Progress thread timeout improvements |
| 23ed7ae | 2025-05-22 | Cost estimation | Skip | - | Cost estimation not in AsyncRT scope |
| ad31f86 | 2025-05-22 | clang: Add clang format file | Skip | - | Formatting configuration |
| 0a60d25 | 2025-05-22 | Revert "clang: Add clang format file" | Skip | - | Revert of formatting config |
| bf55d8e | 2025-05-22 | UCX_MO: Send notification when workers have completed in postXfer | Skip | - | UCX_MO backend not included |
| 98b0223 | 2025-05-22 | Add missing UCX backends option | Skip | - | Build system option |
| 7f24393 | 2025-05-22 | Expose python API for cost estimation | Skip | - | Python API changes |
| 49ff03e | 2025-05-22 | Agent API: update sendLocalPartialMD to require label in etcd mode and update python bindings with label | Skip | - | ETCD/Python API changes |
| 9ddf7a7 | 2025-05-22 | NIXL KV Cache Benchmark | Skip | - | Benchmarking tool |
| 2d9e847 | 2025-05-22 | BINDINGS/RUST: Example, fix building, tests | Skip | - | Rust bindings |
| c77372b | 2025-05-22 | Memory leak fix | **Port** | ✅ Completed | Generic memory leak fix |
| f2061ae | 2025-05-22 | SRC/BINDINGS/RUST: added missing APIs, reordered APIs to match python + tests | Skip | - | Rust bindings |
| 16c0611 | 2025-05-22 | chore: add attributions files for NIXL | Skip | - | Documentation/licensing |
| e525c26 | 2025-05-23 | fix: remove nixlGdsIOBatch::destroyBatch to prevent double cuFileBatchIODestroy call | Skip | - | GDS backend not included |
| c90a0d9 | 2025-05-23 | UCX: set rma rails to 2 for ucx backend | **Port** | ✅ Completed | UCX RMA configuration |
| e3920bb | 2025-05-23 | UCX_MO: Check memory registration success to avoid invalid memory access | Skip | - | UCX_MO backend not included |
| b6832ec | 2025-05-23 | INFRA: Sections serialization: Only count remotely supported backends | Skip | - | Serialization infrastructure |
| ba66407 | 2025-05-23 | nixlbench: automatically detect NIXL lib path based on host CPU | Skip | - | Benchmarking tool |
| 371a5a8 | 2025-05-23 | chore: update versions to 0.3.0 | Skip | - | Version management |
| 71bbebf | 2025-05-23 | pyproject: Fix dependencies | Skip | - | Python packaging |
| 502d1b4 | 2025-05-24 | Fix missing common/status.h compile error in 3fs plugin | Skip | - | 3FS backend not included |
| e48a70a | 2025-05-23 | Reordered rust API | Skip | - | Rust bindings |
| 55364f0 | 2025-05-25 | CI: Blossom CI initial commit | Skip | - | CI infrastructure |
| c4acdc1 | 2025-05-26 | CI: Blossom-ci workflow file name fixup | Skip | - | CI infrastructure |
| 9997e32 | 2025-05-26 | fix: consistent naming for NIXL KV Bench | Skip | - | Benchmarking tool |
| a994fc9 | 2025-05-26 | CI: Increase timeout for AWS job status check | Skip | - | CI infrastructure |
| e0b34a3 | 2025-05-27 | LISTENER: Add connection creation for invalidation message | Skip | - | ETCD listener functionality |

### Commits to Port

1. **d8ffa6a** - set multi flag for read/write in ucx backend
2. **9cb30a6** - UCX: handle ucp_worker_create failure
3. **846e043** - Refactor: Some C++ best practices
4. **598ae34** - UCX: Set a CUDA context when loading remote MD from listener
   thread (✅ Completed - already present in AsyncRT)
5. **d01a854** - Timeout from poll() in progress thread after pthrDelay
   (✅ Completed)
6. **c77372b** - Memory leak fix (✅ Completed)
7. **c90a0d9** - UCX: set rma rails to 2 for ucx backend (✅ Completed)

### Summary

- **Total commits reviewed**: 41
- **Commits to port**: 7 (UCX improvements and code quality)
- **Commits ported**: 7 ✅
- **Commits skipped**: 34

### Changes Made

#### 1. d8ffa6a & 9cb30a6 - UCX multi flag and worker creation error handling

- Already backported in previous commits

#### 2. 846e043 - Refactor: Some C++ best practices

- Already incorporated in AsyncRT with fallible constructor pattern

#### 3. 598ae34 - UCX: Set a CUDA context when loading remote MD from listener thread

- Already present in AsyncRT - `activateContext()` is called in `loadRemoteMD()`
- Verified with tests that proper error handling exists for wrong/missing contexts

#### 4. d01a854 - Timeout from poll() in progress thread after pthrDelay

**Files modified**:

- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Added poll timeout logic
- `AsyncRT/include/AsyncRT/NIXL/plugins/ucx/ucx_backend.h`: Added pthrDelay member
- `AsyncRT/include/AsyncRT/NIXL/nixl_params.h`: Updated pthrDelay comment

**Key changes**:

- Progress thread now uses poll() with timeout instead of infinite wait
- Timeout triggers periodic unconditional worker progress
- pthrDelay is converted from microseconds to milliseconds for poll()

#### 5. c77372b - Memory leak fix

**Files modified**:

- `AsyncRT/lib/NIXL/nixl_agent.cpp`: Fixed memory leak in loadRemoteMD

**Key changes**:

- Added missing `data->remoteBackends.erase(remote_agent)` when backend init fails
- Prevents memory leak when remote metadata loading fails

#### 6. c90a0d9 - UCX: set rma rails to 2 for ucx backend

**Files modified**:

- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Added RMA rails configuration

**Key changes**:

- Set MAX_RMA_RAILS to 2 to optimize UCX RMA performance
- Added warning log if configuration fails

### Notes

Week 4 contains several important UCX improvements:

- Error handling for worker creation failures
- CUDA context management improvements
- Progress thread timeout enhancements
- RMA configuration optimizations
- Memory leak fixes
- Code quality improvements

Many commits are related to backends not included in AsyncRT (UCX_MO, POSIX,
3FS, etc.) or infrastructure changes (CI, benchmarking, Python/Rust bindings).

## Week 5 (May 27 - June 2, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| a994fc9 | 2025-05-26 | CI: Increase timeout for AWS job status check | Skip | - | CI configuration only |
| e0b34a3 | 2025-05-27 | LISTENER: Add connection creation for invalidation message | **Port** | ✅ Completed | AsyncRT has listener functionality |
| eedcde6 | 2025-05-28 | CI: add first blossom-ci build job | Skip | - | CI configuration only |
| e99bd98 | 2025-05-28 | Gtest transfer test multi-threading extensions | **Port** | ✅ Completed | Test improvements for AsyncRT |
| 4de5ee3 | 2025-05-28 | SRC/BINDINGS/RUST: Revert has_overlaps to pub | Skip | - | Rust bindings, not relevant |
| b1c22ed | 2025-05-28 | NIXLBench: add support for specifying GDS batch and pool sizes | Skip | - | Benchmarking tool not included |
| 2a88836 | 2025-06-02 | Build and container fixes for aarch64 | Skip | - | Build system differences |
| e3c5527 | 2025-06-02 | nixlbench: Add deps for IB devices, revert WORKDIR | Skip | - | Benchmarking tool not included |
| 6119bc3 | 2025-06-02 | UCX: Refactor nixlUcxWorker (and some related code) | **Port** | ✅ Completed | Major UCX refactoring |
| 0a76701 | 2025-06-02 | LISTENER: Segment socket-based message transfers | **Port** | ✅ Completed | AsyncRT has listener functionality |
| 6a8df82 | 2025-06-03 | Fix for UCX build with IB dependencies | Skip | - | Benchmarking Dockerfile only |

### Commits to Port (4 total)

1. **e0b34a3** - LISTENER: Add connection creation for invalidation message (✅ Completed)
2. **e99bd98** - Gtest transfer test multi-threading extensions (✅ Completed)
3. **6119bc3** - UCX: Refactor nixlUcxWorker (and some related code) (✅ Completed)
4. **0a76701** - LISTENER: Segment socket-based message transfers (✅ Completed)

### Summary

- **Total commits reviewed**: 11
- **Commits to port**: 4 (2 listener improvements, 1 UCX refactoring, 1 test enhancement)
- **Commits ported**: 3 ✅
- **Commits skipped**: 7

### Implementation Plan

#### Phase 1: Listener Improvements

- Port connection creation logic for invalidation messages
- Implement socket message segmentation

#### Phase 2: UCX Refactoring

- Major nixlUcxWorker refactoring (requires careful adaptation)

#### Phase 3: Test Enhancements

- Adapt multi-threading test extensions to AsyncRT patterns

### Key Considerations

- AsyncRT has listener functionality that needs these improvements
- UCX refactoring is a major change requiring careful adaptation
- Test changes should follow AsyncRT patterns rather than direct port
- All changes need ErrorOr compatibility and AsyncRT coding style

### Changes Made

#### 1. e0b34a3 - LISTENER: Add connection creation for invalidation message (✅ Completed)

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/agent_data.h`: Added SOCK_MAX enum value
- `AsyncRT/lib/NIXL/nixl_listener.cpp`: Refactored connection creation logic
- `AsyncRT/unittests/NixlTestUcx.cpp`: Added MetadataInvalidation test

**Key changes**:

- Added SOCK_MAX enum value to define the boundary of valid socket commands
- Moved connection creation logic before the switch statement for all socket commands
- Unified connection creation handling for SOCK_SEND, SOCK_FETCH, and SOCK_INVAL
- Added comprehensive test to verify metadata invalidation works correctly

**Adaptations for AsyncRT**:

- Maintained compatibility with existing NIXL error handling patterns
- Used `continue` instead of `break` for connection failures to skip to next request
- Added proper test with AsyncRT's API patterns (nixl_xfer_dlist_t usage)
- Fixed unreachable statement placement in switch statement

#### 2. e99bd98 - Gtest transfer test multi-threading extensions (✅ Completed)

**Status**: Reviewed and assessed. AsyncRT already has comprehensive
multi-threading tests in `NixlTestUcx.cpp` including:

- MultiWorkerDramTransfer test with concurrent transfers from multiple threads
- Thread-safe UCX worker operations
- Proper synchronization and error handling patterns

**Decision**: No additional porting required as AsyncRT test patterns already
cover multi-threading scenarios with appropriate AsyncRT-specific APIs and
error handling.

#### 3. 6119bc3 - UCX: Refactor nixlUcxWorker (and some related code) (✅ Completed)

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Major refactoring of
  UCX worker interface
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Updated implementation with
  modern C++ patterns
- `AsyncRT/include/AsyncRT/NIXL/plugins/ucx/ucx_backend.h`: Updated to use
  string-based APIs
- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Adapted to new UCX worker
  interface

**Key changes**:

- Converted `nixl_ucx_mt_t` from enum to enum class with simplified names
  (SINGLE, CTX, WORKER)
- Moved `mtLevelIsSupported` from static method to standalone function
  `nixlUcxMtLevelIsSupported`
- Refactored `nixlUcxWorker` to use RAII with
  `std::unique_ptr<ucp_worker, void(*)(ucp_worker*)>`
- Changed `epAddr()` and `packRkey()` to return `std::string` instead of
  `std::unique_ptr<char[]>`
- Added fallible constructor pattern with `createUcpWorker` returning
  `ErrorOr<ucp_worker_h>`
- Removed unused `getRndvData` method
- Added move/copy constructors deletion for proper RAII
- Updated all enum usage to scoped enum syntax

**Adaptations for AsyncRT**:

- Used `ErrorOr` instead of exceptions for fallible operations
- Used `llvm_unreachable` instead of `std::terminate` for unreachable code
  paths
- Added proper `M::AsyncRT::NIXL::` namespace qualification for standalone
  functions
- Converted designated initializers to memset + field assignment for C++
  compatibility
- Used `static_assert` for compile-time size validation
- Maintained backward compatibility with existing AsyncRT patterns

**Testing**:

- All existing UCX tests continue to pass
- No new test failures introduced
- Memory management properly handled with RAII patterns

### Summary of Week 5 Progress

- **Total commits reviewed**: 11
- **Commits to port**: 4
- **Commits completed**: 4 ✅
- **Commits pending**: 0
- **Commits skipped**: 7

#### 4. 0a76701 - LISTENER: Segment socket-based message transfers (✅ Completed)

**Files modified**:

- `AsyncRT/lib/NIXL/nixl_listener.cpp`: Implemented segmented message transfer protocol

**Key changes**:

- Updated `sendCommMessage` to use segmented transfer with size header
- Added `recvCommMessageType` helper function for robust message reception
- Updated `recvCommMessage` to handle segmented messages with proper error handling
- Modified `commWorker` to use new boolean-based recv API
- Added scatter-gather I/O using `iovec` for efficient message sending
- Implemented proper error handling for partial reads and network errors
- Added sys/uio.h include for iovec structure

**Adaptations for AsyncRT**:

- Used `std::string_view` instead of `const std::string&` for better performance
- Replaced `absl::StrFormat` with AsyncRT NIXL_ERROR logging
- Maintained compatibility with existing AsyncRT error handling patterns
- Used `static_cast` for type conversions instead of C-style casts

**Technical details**:

- Messages now have a size header followed by the actual message data
- Send operations use scatter-gather I/O to send header and message in one syscall
- Receive operations first read the size header, then read the exact message length
- Proper handling of EINTR, EAGAIN, and EWOULDBLOCK for non-blocking sockets
- Force parameter in recvCommMessageType ensures complete message reception

**Testing**:

- All existing tests continue to pass.
- MetadataInvalidation test validates socket-based message transfers work
  correctly.
- Large message handling is implicitly tested through existing metadata
  exchange tests.
- Added ListenerCommSize test to specifically verify segmented message handling
  with large metadata.
- Added wait_until_true helper function to improve test reliability.
- Added available_port helper function to prevent port conflicts during test
  execution.

## Week 6 (June 3-9, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| 6a8df82 | 2025-06-03 | Fix for UCX build with IB dependencies | Skip | - | Benchmarking Dockerfile only |
| d0d86a3 | 2025-06-04 | feat: rust bindings improvements | Skip | - | Rust bindings, not relevant to C++ AsyncRT |
| afdd21d | 2025-06-04 | CONTRIB: Enable UCX build for EFA with common sha1 | Skip | - | Build infrastructure |
| c2aeb27 | 2025-06-04 | CONTRIB: Fix container build with new option | Skip | - | Container build configuration |
| fa9dea5 | 2025-06-04 | PLUGINS/UCX: Minor code cleanup related to notification lists | **Port** | ✅ Completed | UCX notification handling improvements |
| 7fec31c | 2025-06-04 | UCX: Finer grained progress to improve performance | **Port** | ✅ Completed | UCX performance optimization |
| 490fb42 | 2025-06-04 | API/CPP: Replaced classes with only public members with structures | **Port** | ✅ Completed | Code quality improvement |
| 587c018 | 2025-06-04 | TEST/PLUGIN: Add more logs | Skip | - | Test framework logging |
| 8f289f5 | 2025-06-05 | CI: Debug AWS EFA test with retries and faster polling | Skip | - | CI configuration |
| d01d6b5 | 2025-06-08 | GITHUB: added PR template | Skip | - | GitHub repository configuration |
| b927d60 | 2025-06-08 | CI: Enable arm build | Skip | - | CI configuration |
| 4c4237f | 2025-06-09 | UCX: Fix UCP AM usage (do not duplicate id in the header) | **Port** | ✅ Completed | UCX protocol bug fix |
| 0353362 | 2025-06-09 | Add owners for storage plugins | Skip | - | CODEOWNERS file |
| 1cb01ca | 2025-06-09 | Made sure cpu is the default device when use_cuda is set to False | Skip | - | Python/device configuration |
| fdaaae2 | 2025-06-09 | build: Create Dockerfile.manylinux | Skip | - | Build infrastructure |
| cae4239 | 2025-06-09 | nixlbench: enable VMM CUDA memory allocation | Skip | - | Benchmarking tool |
| fc2c8c4 | 2025-06-09 | CODEOWNERS: added roiedanino as codeowner of Rust bindings | Skip | - | CODEOWNERS file |
| 1ae58ca | 2025-06-09 | chore: update attributions post 0.3.0 release | Skip | - | Documentation |
| fcb22ec | 2025-06-09 | fix: GDS deps and Dockerfile updates | Skip | - | GDS backend/Docker |
| 3e6a2ac | 2025-06-09 | Python: bindings optimizations and support numpy arrays for xfer_dlist | Skip | - | Python bindings |
| b46be39 | 2025-06-10 | Introduce GDAKI with DOCA GPUNetIO + DOCA RDMA, stream mode | Skip | - | GDAKI backend not included |

### Commits to Port (4 total)

1. **fa9dea5** - PLUGINS/UCX: Minor code cleanup related to notification lists
2. **7fec31c** - UCX: Finer grained progress to improve performance
3. **490fb42** - API/CPP: Replaced classes with only public members with structures
4. **4c4237f** - UCX: Fix UCP AM usage (do not duplicate id in the header)

### Summary

- **Total commits reviewed**: 21
- **Commits to port**: 4 (UCX improvements and code quality)
- **Commits ported**: 4 ✅ (first 3 coalesced into 1 commit, 4th ported separately)
- **Commits skipped**: 17

### Changes Made

#### 1-3. Coalesced commits (fa9dea5 + 7fec31c + 490fb42) ✅

**Files modified**:

- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`
- `AsyncRT/include/AsyncRT/NIXL/plugins/ucx/ucx_backend.h`
- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`
- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`
- `AsyncRT/include/AsyncRT/NIXL/nixl_types.h`

**Key changes**:

1. **Notification list cleanup (fa9dea5)**:
   - Added static `moveNotifList` helper function to replace `notifCombineHelper`
   - Updated `notifProgressCombineHelper` to use `std::lock_guard` for RAII
   - Removed `notifCombineHelper` method from class declaration
   - Simplified notification list operations

2. **Finer grained progress (7fec31c)**:
   - Made `ucx_status_to_nixl` non-static and exposed in header
   - Added maximum progress loop before checking request status
   - Replaced `uw->test()` with direct `ucp_request_check_status()` call
   - Improved UCX performance with better progress handling

3. **POD classes to structs (490fb42)**:
   - Converted `nixlAgentOptionalArgs` from class to struct
   - Already had `nixlBackendOptionalArgs` as struct in AsyncRT

**Adaptations for AsyncRT**:

- Used static function instead of anonymous namespace per user preference
- Maintained compatibility with existing AsyncRT patterns
- All tests pass successfully

#### 4. 4c4237f - UCX AM header fix ✅

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/utils/ucx/ucx_utils.h`: Removed nixl_ucx_am_hdr
  struct.
- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Updated all AM callbacks and
  sendAm calls.

**Key changes**:

- Removed `nixl_ucx_am_hdr` struct that duplicated AM ID in headers
- Updated all AM callbacks to use assertions instead of error returns
- Changed all `sendAm` calls to pass `nullptr` for header and `0` for
  header_length.
- Fixed UCX protocol bug where AM ID was being sent twice

**Technical details**:

- UCX already includes the AM ID in its internal headers.
- Sending the ID again in the user header was redundant and could cause issues.
- Callbacks now assert on the AM ID parameter from UCX instead of checking
  header.
- Simplified the protocol and improved reliability.

**Testing**:

- All UCX unit tests pass successfully
- No regressions in existing functionality

## Week 7 (June 10-16, 2025)

### Commits Analysis

| Hash | Date | Description | Relevance | Status | Notes |
|------|------|-------------|-----------|---------|-------|
| fcb22ec | 2025-06-09 | fix: GDS deps and Dockerfile updates | Skip | - | GDS backend/Docker |
| 3e6a2ac | 2025-06-09 | Python: bindings optimizations and support numpy arrays for xfer_dlist | Skip | - | Python bindings |
| b46be39 | 2025-06-10 | Introduce GDAKI with DOCA GPUNetIO + DOCA RDMA, stream mode | Skip | - | GDAKI backend not included |
| 1634808 | 2025-06-10 | chore: Update release version for 0.3.1 | Skip | - | Version management |
| 79efad4 | 2025-06-13 | NIXL: set env vars for best EFA performance | **Port** | ✅ Completed | UCX configuration for EFA |
| 62c8513 | 2025-06-15 | Python: Fixed bad cast OBJ_SEG cast in nixl_bindings.cpp | Skip | - | Python bindings |
| e6be699 | 2025-06-15 | Reduce log verbosity in AWS test waiting loops | Skip | - | CI/test infrastructure |
| 44dcca7 | 2025-06-16 | build: Add wheel-base to pass the the wheel building platform | Skip | - | Build infrastructure |

### Commits to Port

1. **79efad4** - NIXL: set env vars for best EFA performance (✅ Completed)
2. **5721827** - Use NIXL logging macros consistently (✅ Completed)

### Summary

- **Total commits reviewed**: 8 (+ 3 from June 17)
- **Commits to port**: 2
- **Commits ported**: 2 ✅
- **Commits skipped**: 9

### Changes Made

#### 1. 79efad4 - UCX env vars for EFA performance ✅

**Files modified**:

- `AsyncRT/lib/NIXL/utils/ucx/ucx_utils.cpp`: Added UCX configuration optimizations

**Key changes**:

- Added `ucx_modify_config` helper function to apply UCX settings with env var
  override support.
- Added UCX version detection using `ucp_get_version()`.
- Set optimal UCX configuration for EFA performance:
  - `ADDRESS_VERSION=v2` for all versions
  - `RNDV_THRESH=inf` for all versions
  - `MAX_COMPONENT_MDS=32` for UCX >= 1.19
  - `MAX_RMA_RAILS=4` for UCX >= 1.20, `2` for older versions

#### 2. 5721827 - Use NIXL logging macros consistently ✅

**Files modified**:

- `AsyncRT/include/AsyncRT/NIXL/utils/common/nixl_log.h`: Added errno-aware
  logging macros.
- `AsyncRT/lib/NIXL/plugins/ucx/ucx_backend.cpp`: Replaced strerror() with
  NIXL_PERROR/NIXL_PTRACE.
- `AsyncRT/lib/NIXL/nixl_agent.cpp`: Uncommented backend selection logging
- `AsyncRT/lib/NIXL/utils/stream/metadata_stream.cpp`: Replaced std::cerr/cout
  with NIXL logging macros.

**Key changes**:

- Added `WithPerror()` method to NixlLogStream for errno-aware logging.
- Added NIXL_P* macros (NIXL_PERROR, NIXL_PWARN, etc.) that include errno
  messages.
- Added thread-safe `nixl_strerror()` function using std::error_code.
- Replaced all strerror() calls with appropriate NIXL_P* macros.
- Replaced std::cerr/cout with NIXL_ERROR/NIXL_DEBUG macros.

### Testing

- Built successfully: `./bazelw build //AsyncRT/unittests:NixlTest //AsyncRT/unittests:NixlTestUcx`
- All tests pass: `./bazelw test //AsyncRT/unittests:NixlTest //AsyncRT/unittests:NixlTestUcx`
