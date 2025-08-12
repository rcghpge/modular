// RUN: mo-opt %s -hash-module-contents -allow-unregistered-dialect -split-input-file | FileCheck %s

// COM: Identical functions should hash to the same value, as should their
// COM: transitive dependencies.

// CHECK-LABEL @foo1
// CHECK: semantic_hash = "[[FOO_HASH:.*]]"
kgen.generator @foo_1(%arg0: si32, %arg1: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL @foo2
// CHECK: semantic_hash = "[[FOO_HASH]]"
kgen.generator @foo_2(%arg0: si32, %arg1: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL @bar1
// CHECK: semantic_hash = "[[BAR_HASH:.*]]"
kgen.generator @bar_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL @bar2
// CHECK: semantic_hash = "[[BAR_HASH]]"
kgen.generator @bar_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_2(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// -----

// COM: Swapping the callsite of functions should change the hash
// COM: This ensures we're properly factoring in the callee's hash on a
// COM: per-callsite basis.

// CHECK-LABEL @foo
kgen.generator @foo(%arg0: si32, %arg1: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL @bar
kgen.generator @bar(%arg0: si32, %arg1: si32) -> si32 {
  kgen.return %arg1 : si32
}

// CHECK-LABEL: @test_1
// CHECK: semantic_hash = "[[HASH:.*]]"
kgen.generator @test_1(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @foo(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @bar(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x, %y : si32, si32
}

// CHECK-LABEL: @test_2
// CHECK-NOT: semantic_hash = "[[HASH]]"
kgen.generator @test_2(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @bar(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @foo(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x, %y : si32, si32
}

// -----

// COM: Equivalent recursive functions could hash to the same value, as should
// COM: equivalent callers.
// COM: Improving this is future work.

// CHECK-LABEL: @recursion_1
// CHECK-SAME: semantic_hash = "[[REC_HASH:.*]]"
kgen.generator @recursion_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL: @recursion_2
// CHECK-NOT: semantic_hash = "[[REC_HASH]]"
kgen.generator @recursion_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL: @test_1
// CHECK-SAME: semantic_hash = "[[CALLER_HASH:.*]]"
kgen.generator @test_1(%arg0: si32) -> si32 {
  %x = kgen.call @recursion_1(%arg0, %arg0) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL @test_2
// CHECK-NOT: semantic_hash = "[[CALLER_HASH]]"
kgen.generator @test_2(%arg0: si32) -> si32 {
  %x = kgen.call @recursion_2(%arg0, %arg0) : (si32, si32) -> si32
  kgen.return %x : si32
}


// -----

// COM: Equivalent recursive functions should hash to the same value, as should
// COM: equivalent callers.

kgen.generator @scc_input() {
  "some_op" () : () -> ()
  kgen.return
}

// CHECK-LABEL: @recursion_1
// CHECK: semantic_hash = "[[REC_HASH_1:.*]]"
kgen.generator @recursion_1(%arg0: si32, %arg1: si32) -> (si32, si32) {
  kgen.call @scc_input() : () -> ()

  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_3(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x, %y : si32, si32
}

// CHECK-LABEL: @recursion_2
// CHECK: semantic_hash = "[[REC_HASH_2:.*]]"
kgen.generator @recursion_2(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @recursion_3(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x, %y : si32, si32
}

// CHECK-LABEL: @recursion_3
// CHECK-NOT: semantic_hash = "[[REC_HASH_1]]"
// CHECK-NOT: semantic_hash = "[[REC_HASH_2]]"
kgen.generator @recursion_3(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x, %y : si32, si32
}

// -----

// CHECK-LABEL: @foo_recursive_1
// CHECK: semantic_hash = "[[HASH1:.*]]"
kgen.generator @foo_recursive_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_recursive_2(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL: @foo_recursive_2
// CHECK: semantic_hash = "[[HASH2:.*]]"
kgen.generator @foo_recursive_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_recursive_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL: @bar_recursive_1
// CHECK: semantic_hash = "[[HASH1]]"
kgen.generator @bar_recursive_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @bar_recursive_2(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// CHECK-LABEL: @bar_recursive_2
// CHECK: semantic_hash = "[[HASH2]]"
kgen.generator @bar_recursive_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @bar_recursive_1(%arg0, %arg1) : (si32, si32) -> si32
  kgen.return %x : si32
}

// -----

// Ensure that functions referred to in "parameter" positions are factored
// into the hash. This occurs commonly in Mojo code with functions like
// `ctx.enqueue_function[my_func]()`. There is no direct call to `my_func` in
// this instance.

// CHECK-LABEL: @parameter_function_1
// CHECK: semantic_hash = "[[FUNC_HASH:.*]]"
kgen.generator @parameter_function_1(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %x = pop.add %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %x : !pop.scalar<si32>
}

// CHECK-LABEL: @parameter_function_2
// CHECK-NOT: semantic_hash = "[[FUNC_HASH]]"
kgen.generator @parameter_function_2(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %x = pop.sub %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %x : !pop.scalar<si32>
}

kgen.generator @enqueue_function<callable: (!pop.scalar<si32>) -> (!pop.scalar<si32>)>() -> !pop.scalar<si32> {
  %c0 = kgen.param.constant : !pop.scalar<si32> = <1>
  %callee = kgen.param.constant : (!pop.scalar<si32>) -> !pop.scalar<si32> = <callable>
  %call = kgen.call_indirect %callee(%c0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
  kgen.return %call : !pop.scalar<si32>
}

// CHECK-LABEL: @main_1
// CHECK: semantic_hash = "[[MAIN_HASH:.*]]"
kgen.generator @main_1() {
  kgen.call @enqueue_function<:(!pop.scalar<si32>) -> !pop.scalar<si32> @parameter_function_1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: @main_2
// CHECK-NOT: semantic_hash = "[[MAIN_HASH]]"
kgen.generator @main_2() {
  kgen.call @enqueue_function<:(!pop.scalar<si32>) -> !pop.scalar<si32> @parameter_function_2>() : () -> ()
  kgen.return
}
