// RUN: mo-opt %s --mo-merge-adjacent-parallels --verify-each | FileCheck %s

// Unit test for the merge-adjacent-parallels pass in isolation.  Tests the
// full create -> canonicalize -> merge pipeline live in
// create-parallel-from-unbundle.mlir and the integration suite.

// ============================================================================
// Simple merge: B consumes A's single bundle output directly (no buffers,
// no chain).  Merged op has the concatenated body and yields both values.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_simple
mo.graph @merge_simple(%arg0: !mo.tensor<[3], f32, gpu:0>,
                       %arg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                    !mo.tensor<[3], f32, gpu:1>]>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %a = mo.parallel (%arg) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %c = mo.parallel (%arg) in (%a
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // Exactly one merged parallel with both ops in sequence.  B's block arg
  // is replaced by A's yield operand (%r), and A's yield is dropped since
  // it had no external users beyond the now-absorbed B.
  // CHECK:     mo.parallel
  // CHECK:       %[[R:[a-zA-Z0-9_]+]] = mo.relu
  // CHECK:       %[[N:[a-zA-Z0-9_]+]] = mo.negative(%[[R]])
  // CHECK:       mo.yield %[[N]] :
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
}

// ============================================================================
// Non-shared consumer input bundle is defined between producer and consumer.
// The pass does not hoist it: the merged op is inserted at the producer slot
// and may reference that bundle before its defining op (allowed in mo.graph).
// ============================================================================

// CHECK-LABEL: mo.graph @merge_with_non_shared_between
mo.graph @merge_with_non_shared_between(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %r0: !mo.tensor<[3], f32, gpu:0>,   %r1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                    !mo.tensor<[3], f32, gpu:1>]>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %a = mo.parallel (%arg) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x : !mo.tensor<[3], f32, gpu:0>
  }
  // This bundle is constructed AFTER %a but before %c; merged parallel is
  // still inserted before %a and takes it as an operand.
  %residuals = mo.tensor.bundle(%r0, %r1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %c = mo.parallel (%arg_a, %arg_r) in (%a
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>,
                                           %residuals
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %s = mo.add(%arg_a, %arg_r) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %s : !mo.tensor<[3], f32, gpu:0>
  }

  // One merged parallel; residuals bundle may appear after it in the block.
  // CHECK:     mo.tensor.bundle
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:     mo.tensor.bundle
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
}

// ============================================================================
// Buffers + chain on BOTH sides: merge requires identical buffer lists and
// strict chain threading (B.inChain == A.outChain, no other users).
// ============================================================================

// CHECK-LABEL: mo.graph @merge_chain_both
mo.graph @merge_chain_both(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                    !mo.tensor<[3], f32, gpu:1>]>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x : !mo.tensor<[3], f32, gpu:0>
  }
  %c:2 = mo.parallel (%arg) in (%a#0
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%a#1)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // One merged parallel that keeps A's buffers + inChain; outChain is a
  // single trailing result.
  // CHECK: %[[M:[a-zA-Z0-9_]+]]:{{[0-9]+}} = mo.parallel
  // CHECK-SAME: buffers(%{{.*}} : !mo.buffer<[1], ui8, gpu:0>, %{{.*}} : !mo.buffer<[1], ui8, gpu:1>)
  // CHECK:   mo.relu
  // CHECK:   mo.negative
  // CHECK:   mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c#0, %c#1
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>,
        !mo.chain
}

// ============================================================================
// Buffer mismatch on BOTH sides: merge must be rejected, two parallels
// remain.
// ============================================================================

// CHECK-LABEL: mo.graph @no_merge_buffer_mismatch
mo.graph @no_merge_buffer_mismatch(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sigA0: !mo.buffer<[1], ui8, gpu:0>, %sigA1: !mo.buffer<[1], ui8, gpu:1>,
    %sigB0: !mo.buffer<[1], ui8, gpu:0>, %sigB1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                    !mo.tensor<[3], f32, gpu:1>]>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%sigA0 : !mo.buffer<[1], ui8, gpu:0>,
              %sigA1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x : !mo.tensor<[3], f32, gpu:0>
  }
  %c:2 = mo.parallel (%arg) in (%a#0
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%sigB0 : !mo.buffer<[1], ui8, gpu:0>,
              %sigB1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%a#1)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // Different buffers on each side — the merge must bail; both parallels
  // remain.
  // CHECK: mo.parallel
  // CHECK: mo.parallel

  mo.output %c#0, %c#1
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>,
        !mo.chain
}

// ============================================================================
// One-sided buffers + chain: producer has buffers+chain, consumer doesn't.
// Merged op inherits producer's buffers+chain.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_chain_producer_only
mo.graph @merge_chain_producer_only(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                    !mo.tensor<[3], f32, gpu:1>]>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x : !mo.tensor<[3], f32, gpu:0>
  }
  %c = mo.parallel (%arg) in (%a#0
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // Merged op carries A's buffers and chain.
  // CHECK:     mo.parallel
  // CHECK-SAME: buffers(%{{.*}} : !mo.buffer<[1], ui8, gpu:0>, %{{.*}} : !mo.buffer<[1], ui8, gpu:1>)
  // CHECK:       mo.relu
  // CHECK:       mo.negative
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c, %a#1
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>,
        !mo.chain
}
