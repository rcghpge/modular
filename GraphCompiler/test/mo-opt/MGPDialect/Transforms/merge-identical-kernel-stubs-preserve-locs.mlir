// RUN: mo-opt %s -mogg-merge-identical-kernel-stubs -mlir-print-debuginfo 2>&1 | FileCheck %s

// Test that MergeIdenticalKernelStubs preserves source note locations from
// both stubs when merging duplicates. Without location merging, the losing
// stub's Python call site metadata would be silently dropped.

// func_a and func_b are structurally identical (same body, same attributes)
// but carry different source note locations. After merging, the surviving
// stub should have a FusedLoc combining both original source notes.

// Verify call sites are rewritten to the surviving stub.
// CHECK: mogg.call.stub @func_a
// CHECK: mogg.call.stub @func_a

// The individual source note locs survive in the #loc alias table.
// CHECK-DAG: loc(fused<{{.*}}call_site_a
// CHECK-DAG: loc(fused<{{.*}}call_site_b

// A merged fused loc references both (the bare fused[] without metadata).
// This only exists if the merge pass combined the locations.
// CHECK: loc(fused[

mogg.kernel.stub @func_a private()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_a">>} {
  mogg.output
} loc(fused<"  File \22model.py\22, line 10, in call_site_a\n">["model.py":10:0])

mogg.kernel.stub @func_b private()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_b">>} {
  mogg.output
} loc(fused<"  File \22model.py\22, line 20, in call_site_b\n">["model.py":20:0])

mogg.kernel.stub @caller private()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call.stub @func_a
  mogg.call.stub @func_b
  mogg.output
}
