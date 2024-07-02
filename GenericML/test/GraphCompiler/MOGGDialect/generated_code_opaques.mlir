// RUN: mo-opt --jit-compile-kernels="dump-stub=true kernel-library-paths="%extra_test_lib"" %s -o %t 2>&1 | FileCheck %s


module {
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,stdlib::builtin::int::Int)"(!kgen.pointer<struct<(index) memoryOnly>> init_self, index) -> !kgen.none
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none
  mgp.model @graph : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.check_runtime %0, %1[0], <host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "arm64-apple-darwin22.6.0", arch = "apple-m2", features = "+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+ras,+rcpc,+rdm,+sha2,+sha3">>]>
    %3 = mgp.runtime.create %2, %1[1]
    mef.output %3, %1 : !mef.chain, !mgp.context
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.opaque) {
    %0:2 = mogg.generic.compiled.region (%arg0) : (%arg2: !mef.chain) -> (!mef.chain, !mgp.opaque){
      %1:2 = mogg.get_async_vals() : () -> (!kgen.pointer<none>, !kgen.pointer<none>)
      %2 = mogg.rt_placeholder : !kgen.pointer<none>
      %3 = mogg.stack_allocate_opaques : !mgp.opaque
      %4 = "mogg.to_kgen"(%3) {typeMethodsDict = {__del__ = #kgen.symbol.constant<@"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"> : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>, __moveinit__ = #kgen.symbol.constant<@"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"> : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>}} : (!mgp.opaque) -> !kgen.pointer<struct<(index) memoryOnly>>
      %index42 = kgen.param.constant = <42>
      %5 = kgen.call @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,stdlib::builtin::int::Int)"(%4, %index42) : (!kgen.pointer<struct<(index) memoryOnly>> init_self, index) -> !kgen.none
      %6 = mogg.to_mgp_from_opaque(%4) : (!kgen.pointer<struct<(index) memoryOnly>>) -> !mgp.opaque
      mogg.create_async_mojo_value(%6, %1#1, %2) : (!mgp.opaque, !kgen.pointer<none>, !kgen.pointer<none>) -> ()
      mogg.output %6 : !mgp.opaque
    } {canFuseIntoOther = false, inOutSignature = #M.inout_sig<"i">, isAsync = false, isKGEN = true, summary = "custom__test_make_custom_int:5"}
    mef.output %arg0, %0#1 : !mef.chain, !mgp.opaque
  }
}


// CHECK-LABEL: kgen.generator export @__gen__
//  CHECK-SAME: ([[ARG0:%.*]]: !kgen.pointer<pointer<none>>, [[ARG1:%.*]]: !kgen.pointer<none>, [[ARG2:%.*]]: !kgen.pointer<none>) {
//  CHECK-NEXT: kgen.param.declare simd_width = <1>

// COM: Load the async value pointer.
// CHECK-NEXT: [[INDEX1:%.*]] = kgen.param.constant = <1>
// CHECK-NEXT: [[V0:%.*]] = pop.offset [[ARG0]][[[INDEX1]]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] align<8> : !kgen.pointer<pointer<none>>

// COM: Do the stack allocation for the custom kernel output.
// CHECK-NEXT: [[V2:%.*]] = pop.stack_allocation 1 x struct<(index) memoryOnly>
// CHECK-NEXT: [[V3:%.*]] = kgen.rebind [[V2]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<struct<(index) memoryOnly>>

// COM: Call the custom kernel that creates the custom mojo value.
// CHECK-NEXT: [[INDEX42:%.*]] = kgen.param.constant = <42>
// CHECK-NEXT: [[V4:%.*]] = kgen.call @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,stdlib::builtin::int::[[INT:[A-Za-z0-9_]*]])"([[V3]], [[INDEX42]]) : (!kgen.pointer<struct<(index) memoryOnly>> init_self, index) -> !kgen.none

// COM: Get the move / del function pointers
// CHECK-NEXT: [[V5:%.*]] = kgen.create_closure[(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none: @"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"]()
// CHECK-NEXT: [[V6:%.*]] = kgen.create_closure[(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none: @"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"]()

// COM: Get the sizeof and alignof properties needed to allocate heap memory for the mojo value.
// CHECK-NEXT: [[INDEX:%.*]] = kgen.param.constant = <get_sizeof(struct<(index) memoryOnly>, current_target())>
// CHECK-NEXT: [[INDEX_0:%.*]] = kgen.param.constant = <get_alignof(struct<(index) memoryOnly>, current_target())>

// COM: bitcast mov / del the expeced type for create_mojo_value_async.
// CHECK-NEXT: [[V7:%.*]] = pop.pointer.bitcast [[V5]] : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> to !kgen.signature<(!kgen.pointer<none>, !kgen.pointer<none>) -> !kgen.none>
// CHECK-NEXT: [[V8:%.*]] = pop.pointer.bitcast [[V6]] : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> to !kgen.signature<(!kgen.pointer<none>) -> !kgen.none>

// COM: Cast / rebind ptr / runtime / async value to the correct types for create_mojo_value_async.
// CHECK-NEXT: [[V9:%.*]] = pop.pointer.bitcast [[V1]] : !kgen.pointer<none> to !kgen.pointer<none>
// CHECK-NEXT: [[V10:%.*]] = pop.pointer.bitcast [[ARG1]] : !kgen.pointer<none> to !kgen.pointer<none>
// CHECK-NEXT: [[V11:%.*]] = pop.pointer.bitcast [[V3]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<none>

// COM: call create_mojo_value_async
// CHECK-NEXT: [[V12:%.*]] = kgen.call @create_mojo_value_async([[V11]], [[V9]], [[V10]], [[INDEX]], [[INDEX_0]], [[V8]], [[V7]]) : (!kgen.pointer<none>, !kgen.pointer<none>, !kgen.pointer<none>, index, index, !kgen.signature<(!kgen.pointer<none>) -> !kgen.none>, !kgen.signature<(!kgen.pointer<none>, !kgen.pointer<none>) -> !kgen.none>) -> !kgen.none
// CHECK-NEXT: kgen.return
