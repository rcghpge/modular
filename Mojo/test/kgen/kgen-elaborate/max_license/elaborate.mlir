// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK %s

// CHECK-LABEL: kgen.func @"metadata,x=16,y=16,z=8"
// CHECK-SAME: fnArgAttrs = [{}, {nvvm.maxntid = #pop.array<16, 16, 8>
// CHECK-SAME: fnAttrs = {nvvm.maxntid = #pop.array<16, 16, 8>
kgen.generator @metadata<x: i32, y: i32, z: i32>(%a: i32, %b: i32) attributes {
  fnArgAttrs = [
    [],
    ["nvvm.maxntid",  #pop.array<x, y, z> : !pop.array<3, i32>]
  ],
  fnAttrs = [
    "nvvm.maxntid",  #pop.array<x, y, z> : !pop.array<3, i32>
  ]
}{
  hlcf.return
}

kgen.generator @kernel(%a: i32, %b: i32) {
  kgen.call @metadata<:i32 16, :i32 16, :i32 8>(%a, %b) : (i32, i32) -> ()
  hlcf.return
}

// -----

// COM: Given a compile_offload op over a capturing function FOO,
// verify that the populate captures function is generated and accessible
// for calls via the compile_offload_closure operator.
kgen.generator @HELLO<x: () capturing -> index>() capturing -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  %0 = kgen.call_param[() capturing -> index: x]()
  hlcf.return %none : !kgen.none
}

kgen.generator @FOO() capturing -> index {
  %0 = pop.compiler.global_load "CAPTURE_0" : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  hlcf.return %1 : index
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry(%arg0: !kgen.pointer<none>) {
  %0 = pop.stack_allocation 1 x index marked
  pop.compiler.global_store "CAPTURE_0", %0 : !kgen.pointer<index>
  kgen.param.declare *"foo()": () capturing -> index = <@FOO>
  // CHECK: [[STR:%.*]] = kgen.param.constant: string = <"; Mod{{.*}}">
  // CHECK-NEXT: [[STR1:%.*]] = kgen.param.constant: string = <"{{.*}}">
  // CHECK-NEXT: [[NUM_CAPTURES:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: [[CAPTURE_SIZES:%.*]] = pop.stack_allocation 1 x i64
  // CHECK-NEXT: [[INDEX0:%.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: [[GEP0:%.*]] = pop.offset [[CAPTURE_SIZES]][[[INDEX0]]] : !kgen.pointer<i64>
  // CHECK-NEXT: [[CAPTURE_SIZE:%.*]] = kgen.param.constant: i64 = <8>
  // CHECK-NEXT: pop.store [[CAPTURE_SIZE]], [[GEP0]] : !kgen.pointer<i64>
  // CHECK-NEXT: [[OPAQUE_CAPTURE_SIZES:%.*]] = pop.pointer.bitcast [[CAPTURE_SIZES]] : !kgen.pointer<i64> to !kgen.pointer<none>
  // CHECK: kgen.struct.create([[STR]], [[STR1]], [[NUM_CAPTURES]], [[OPAQUE_CAPTURE_SIZES]]) : !kgen.struct<(string, string, index, pointer<none>)>

  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  %1 = kgen.compile_offload<nvptx, 2, "", "",
                            :() capturing -> !kgen.none @HELLO<:() capturing -> index *"foo()">>
                            : !kgen.struct<(string, index)>
  // CHECK-NEXT: kgen.call @"HELLO,x=FOO_populate_captures"(%arg0) : (!kgen.pointer<none>) -> !kgen.none
  kgen.param.declare x: (!kgen.pointer<none>) -> !kgen.none = <#kgen.compile_offload_closure<
    nvptx, #kgen.symbol.constant<@HELLO<:() capturing -> index *"foo()">> : !kgen.generator<() capturing -> !kgen.none>>>
  %2 = kgen.call_param[(!kgen.pointer<none>) -> !kgen.none: x](%arg0)
  hlcf.return
}

// COM: populate_captures must be an inlined
// CHECK: kgen.func @"HELLO,x=FOO_populate_captures"(%arg0: !kgen.pointer<none>) -> !kgen.none always_inline {
// CHECK-NEXT: [[V0:%.*]] = pop.compiler.global_load "CAPTURE_0" : !kgen.pointer<index>
// CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x pointer<index>
// CHECK-NEXT: pop.store [[V0]], [[V1]] : !kgen.pointer<pointer<index>>
// CHECK-NEXT: [[V2:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
// CHECK-NEXT: [[IDX0:%.*]] = kgen.param.constant = <0>
// CHECK-NEXT: [[V3:%.*]] = pop.offset [[V2]][[[IDX0]]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V1]] : !kgen.pointer<pointer<index>> to !kgen.pointer<none>
// CHECK-NEXT: pop.store [[V4]], [[V3]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT: [[NONE:%.*]] = kgen.param.constant: none = <#kgen.none>
// CHECK-NEXT: hlcf.return [[NONE]] : !kgen.none

// -----

// The offload target-attribute dictionary is forwarded from the generator
// onto the instantiated function, where the target backend reads it.

// CHECK-LABEL: kgen.func @target_attrs_forwarding
// CHECK-SAME: kgen.target_attrs = {tpu.grid_dims = [2, 4]}
kgen.generator @target_attrs_forwarding() -> index attributes {"kgen.target_attrs" = {tpu.grid_dims = [2, 4]}} {
  %c = kgen.param.constant = <42>
  hlcf.return %c : index
}
kgen.generator @main() -> index {
  %v = kgen.call @target_attrs_forwarding() : () -> index
  hlcf.return %v : index
}
