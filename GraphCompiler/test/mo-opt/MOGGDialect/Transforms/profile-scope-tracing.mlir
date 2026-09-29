// RUN: mo-opt %s -mogg-add-try-except-blocks 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: mo-opt %s -mogg-add-profile-scope-tracing 2>&1 | FileCheck %s --check-prefix=ON
// RUN: mo-opt %s -mogg-add-profile-scope-tracing="profile-scope-tracing-trace-unlabelled=true" 2>&1 | FileCheck %s --check-prefix=TRACE

// OFF-LABEL: mogg.kernel.stub @region_952
// OFF-NOT: mogg.profile_scope

// ON-LABEL: mogg.kernel.stub @region_952
// ON: mogg.profile_scope.begin "[draft_forward] region_952"
// ON: mogg.profile_scope.end

// TRACE-LABEL: mogg.kernel.stub @region_952
// TRACE: mogg.profile_scope.begin "[draft_forward] region_952"
// TRACE: mogg.profile_scope.end

mogg.kernel.stub @region_952 (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @cast_mogg (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// ON-LABEL: mogg.kernel.stub @unlabeled_region
// ON-NOT: mogg.profile_scope.begin "unlabeled_region"

// TRACE-LABEL: mogg.kernel.stub @unlabeled_region
// TRACE: mogg.profile_scope.begin "unlabeled_region"
// TRACE: mogg.profile_scope.end

mogg.kernel.stub @unlabeled_region (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @cast_mogg_unlabeled (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @region_glue_mismatch (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @cast_step (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @region_shared (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
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

mogg.kernel.stub @cast_shared (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// Color is region-only: a single consistent color on every op in a run
// resolves to a `color` attribute on the begin op.
// ON-LABEL: mogg.kernel.stub @region_color
// ON: mogg.profile_scope.begin "[colored] region_color" {color = "orange"} : !mogg.profile_scope_range

mogg.kernel.stub @region_color (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  %c0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %c1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %c2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %c3 = mogg.tensor.__init__(%c0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %c4 = mogg.tensor.__init__(%c1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_mogg_color %c3, %c4, %c2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"colored", loc(unknown), color = "orange">)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %c5 = mogg.async.ready : !mef.chain
  mogg.output %c5 : !mef.chain
}

mogg.kernel.stub @cast_mogg_color (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// Disagreeing colors on ops in the same run resolve to no color attribute,
// rather than guessing one.
// ON-LABEL: mogg.kernel.stub @region_color_conflict
// ON: mogg.profile_scope.begin "[conflict] region_color_conflict" : !mogg.profile_scope_range

mogg.kernel.stub @region_color_conflict (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  %x0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %x1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %x2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %x3 = mogg.tensor.__init__(%x0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %x4 = mogg.tensor.__init__(%x1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_mogg_conflict1 %x3, %x4, %x2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"conflict", loc(unknown), color = "orange">)
  mogg.call.stub @cast_mogg_conflict2 %x3, %x4, %x2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"conflict", loc(unknown), color = "blue">)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %x5 = mogg.async.ready : !mef.chain
  mogg.output %x5 : !mef.chain
}

mogg.kernel.stub @cast_mogg_conflict1 (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}

mogg.kernel.stub @cast_mogg_conflict2 (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}

// ---

// Two adjacent leaf runs nested under the same outer scope get an ancestor
// range bracketing both, in addition to each step's own narrower range.
// The color on the outer scope alone still resolves on both the ancestor and
// the leaf ranges because color propagates through every active scope layer.
#step0 = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown), color = "orange">>)
#step1 = loc(#mogg.profile_scope<"draft_step_1", #mogg.profile_scope<"draft_forward", loc(unknown), color = "orange">>)

// ON-LABEL: mogg.kernel.stub @region_ancestor
// ON: mogg.profile_scope.begin "[draft_forward] region_ancestor" {color = "orange"}
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_0] region_ancestor" {color = "orange"}
// ON: mogg.call.stub @cast_step
// ON: mogg.profile_scope.end
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_1] region_ancestor" {color = "orange"}
// ON: mogg.call.stub @cast_step
// ON: mogg.profile_scope.end
// ON: mogg.profile_scope.end

mogg.kernel.stub @region_ancestor (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  %a0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %a1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %a2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %a3 = mogg.tensor.__init__(%a0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %a4 = mogg.tensor.__init__(%a1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_step %a3, %a4, %a2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step0)
  mogg.call.stub @cast_step %a3, %a4, %a2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step1)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %a5 = mogg.async.ready : !mef.chain
  mogg.output %a5 : !mef.chain
}

// ---

// A single nested run with no sibling under the same outer scope gets no
// ancestor range -- it would just duplicate the leaf range's exact span under
// a less specific name.
#step_only = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown)>>)

// ON-LABEL: mogg.kernel.stub @region_ancestor_single
// ON-NOT: mogg.profile_scope.begin "[draft_forward] region_ancestor_single"
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_0] region_ancestor_single"

mogg.kernel.stub @region_ancestor_single (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain fusionInfo = "mo.cast" attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  %s0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %s1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %s2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %s3 = mogg.tensor.__init__(%s0) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>
  %s4 = mogg.tensor.__init__(%s1) {io = 0 : i32, mut = true} : (!mgp.buffer) -> !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>
  mogg.call.stub @cast_step_single %s3, %s4, %s2 : !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step_only)
  mogg.async.__del__(%arg0, %arg1, %arg2, %arg3) : (!mogg.async<!mef.chain>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.buffer>, !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> ()
  %s5 = mogg.async.ready : !mef.chain
  mogg.output %s5 : !mef.chain
}

mogg.kernel.stub @cast_step_single (%arg0: !mo.tensor<[10], f32, {io = 1 : i32, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = 0 : i32, mut = true}>, %arg2: !mogg.context) fusionInfo = "mo.cast" attributes {abi = 0 : i32, constantResources = [], device = "cpu", isAsync = true, sym_visibility = "private"} {
  mogg.output
}
