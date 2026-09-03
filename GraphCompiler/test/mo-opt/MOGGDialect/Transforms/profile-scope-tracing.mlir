// RUN: mo-opt %s -mogg-add-try-except-blocks 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: mo-opt %s -mogg-add-try-except-blocks="enable-profile-scope-tracing=true" 2>&1 | FileCheck %s --check-prefix=ON
// RUN: mo-opt %s -mogg-add-try-except-blocks="enable-profile-scope-tracing=true profile-scope-tracing-trace-unlabelled=true" 2>&1 | FileCheck %s --check-prefix=TRACE

// OFF-LABEL: mogg.kernel.stub @region_952
// OFF-NOT: mogg.profile_scope

// ON-LABEL: mogg.kernel.stub @region_952
// ON: mogg.profile_scope.begin "[draft_forward] region_952"
// ON: mogg.profile_scope.end

// TRACE-LABEL: mogg.kernel.stub @region_952
// TRACE: mogg.profile_scope.begin "[draft_forward] region_952"
// TRACE: mogg.profile_scope.end

mogg.kernel.stub @region_952 (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  %0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %3 = mogg.tensor.__init__(%0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %4 = mogg.tensor.__init__(%1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_mogg %3, %4, %2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %5 = mogg.async.ready : !mef.chain
  mogg.output %5 : !mef.chain
}

mogg.kernel.stub @cast_mogg (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// OFF-LABEL: mogg.kernel.stub @unlabeled_region
// OFF-NOT: mogg.profile_scope

// ON-LABEL: mogg.kernel.stub @unlabeled_region
// ON-NOT: mogg.profile_scope.begin "unlabeled_region"

// TRACE-LABEL: mogg.kernel.stub @unlabeled_region
// TRACE: mogg.profile_scope.begin "unlabeled_region"
// TRACE: mogg.profile_scope.end

mogg.kernel.stub @unlabeled_region (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  %0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %3 = mogg.tensor.__init__(%0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %4 = mogg.tensor.__init__(%1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_mogg_unlabeled %3, %4, %2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>>
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %5 = mogg.async.ready : !mef.chain
  mogg.output %5 : !mef.chain
}

mogg.kernel.stub @cast_mogg_unlabeled (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// A host-side bookkeeping op with no device dispatch of its own (here,
// mogg.tensor.__init__) must never split two device-dispatching ops that
// share the same profile_scope layer into separate runs, even if that
// bookkeeping op happens to carry a stale/foreign layer -- e.g. a CSE'd
// cached-buffer lookup shared across non-adjacent profile_scope() call
// sites, whose location was never re-attributed to the step it's actually
// read from here. One continuous range should cover both calls, not two.
#glueStepA = loc(#mogg.profile_scope<"draft_step_1", #mogg.profile_scope<"draft_forward", loc(unknown)>>)
#glueStepForeign = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown)>>)

// ON-LABEL: mogg.kernel.stub @region_glue_mismatch
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_1] region_glue_mismatch"
// ON: mogg.call.stub @cast_step
// ON-NOT: mogg.profile_scope.end
// ON: mogg.tensor.__init__
// ON-NOT: mogg.profile_scope.begin
// ON: mogg.call.stub @cast_step
// ON: mogg.profile_scope.end

mogg.kernel.stub @region_glue_mismatch (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  %g0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %g1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %g2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %g3 = mogg.tensor.__init__(%g0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %g4 = mogg.tensor.__init__(%g1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_step %g3, %g4, %g2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#glueStepA)
  %g5 = mogg.tensor.__init__(%g0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}> loc(#glueStepForeign)
  mogg.call.stub @cast_step %g3, %g4, %g2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#glueStepA)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %g6 = mogg.async.ready : !mef.chain
  mogg.output %g6 : !mef.chain
}

mogg.kernel.stub @cast_step (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// MergeIdenticalKernelStubs can canonicalize structurally-identical, non-nested
// profile scope call sites (e.g. the same matmul kernel invoked from
// draft_forward and target_forward) onto one shared surviving KernelStubOp.
// Every merged duplicate's location is fused onto the stub's own location.
// Locations of the ops *inside* that body still carry an arbitrary
// original, single-scope location
// In-body bracketing must be suppressed in this case, otherwise
// the generated range would show one scope and the ops inside
// would be labeled with a different scope.
#shared_a = #mogg.profile_scope<"draft_forward", loc(unknown)>
#shared_b = #mogg.profile_scope<"target_forward", loc(unknown)>

// ON-LABEL: mogg.kernel.stub @region_shared
// ON-NOT: mogg.profile_scope

mogg.kernel.stub @region_shared (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  %sh0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %sh1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %sh2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %sh3 = mogg.tensor.__init__(%sh0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %sh4 = mogg.tensor.__init__(%sh1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_shared %sh3, %sh4, %sh2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#shared_a)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %sh5 = mogg.async.ready : !mef.chain
  mogg.output %sh5 : !mef.chain
} loc(fused[#shared_a, #shared_b])

mogg.kernel.stub @cast_shared (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.output
}

