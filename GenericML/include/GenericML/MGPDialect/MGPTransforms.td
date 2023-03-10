//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"
include "MGPOps.td"

//===----------------------------------------------------------------------===//
// PeepholeTensorExtract
//===----------------------------------------------------------------------===//

def PeepholeTensorExtract : Pass<"peephole-tensor-extract", "MGP::ModelOp"> {
  let summary = "Removes redundant tensor extract operations from the graph.";
  let description = [{
    If a tensor is created only to be extracted then we 
    should elide both and just refer to the underlying
    buffer and spec.
  }];
}

//===----------------------------------------------------------------------===//
// LookupCachedConstantsPass
//===----------------------------------------------------------------------===//

def LookupCachedConstantsPass : Pass<"lookup-cached-constants", "MGP::ModelOp"> {
  let summary = "Replace uses of #cache.constant_hash<TAG> "
                "with mgp.cas.lookup, mgp.cpu.buffer.cache and "
                "mgp.cpu.buffer.get_cached";
  let description = [{
    This pass replace uses of #cache.constant_hash<"TAG> with mgp.cas.lookup,
    mgp.cpu.buffer.cache and mgp.cpu.buffer.get_cached.

    For example:
    ```mlir
    %buf = mgp.cpu.buffer.constant #cache.constant_hash<"tag"> : !mgp.cpu.buffer<256>
    ```
    becomes, in the 'init' block:
    ```mlir
    %ch1 = mgp.cas.create_instance %ch0, %ctx[0] at "cacheDir"
    %buf = mgp.cas.lookup %ch1, %ctx[0], "tag" : !mgp.cpu.buffer<256>
    %ch2 = mgp.cpu.buffer.cache %ch1, %ctx[5], %buf : !mgp.cpu.buffer<256>
    ```
    and in the 'execute' block:
    ```mlir
    %buf = mgp.cpu.buffer.get_cached %ch, %ctx[5] : !mgp.cpu.buffer<256>
    ```
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["Cache::CacheDialect"];
}

//===----------------------------------------------------------------------===//
// StripBufferAttributesPass
//===----------------------------------------------------------------------===//

def StripBufferAttributesPass : Pass<"strip-buffer-attributes", "MGP::ModelOp"> {
  let summary = "Removes numOutputBuffers attributes.";
  let description = [{
    Removes numOutputBuffers attributes. These may be added during lowering
    to help downstream passes distinguish between input and output buffer
    arguments, but are not needed thereafter.
  }];
}

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

def ConcatInPlacePass : Pass<"concat-in-place", "MGP::ModelOp"> {
  let summary = "Replaces mgp.cpu.buffer.concat with "
                "mcp.cpu.buffer.split.mutable_borrow.";
  let description = [{
    This pass replaces mgc.cpu.buffer.concat with
    mcp.cpu.buffer.split.mutable_borrow, provided all writers to the buffer
    operands to the concat can be rewritten to instead write to the buffer
    results of split.

    For example:
    ```mlir
    %in1_buf = mgp.cpu.buffer.alloc.static ..., <size1>
    %in1_ch = ...
    %in2_spec = ...
    %in2_buf = mgp.cpu.buffer.alloc.spec ..., %in2_spec
    %in2_ch = ...
    %out_spec = ...
    %out_buf = mgp.cpu.buffer.alloc.spec ..., %out_spec
    %ch1 = grt.chain.merge %in1_ch, %in2_ch
    %ch2 = mgp.cpu.buffer.concat %ch1, %out_buf, %in1_buf, %in2_buf : ...
    // where these are the only uses of %in1_buf and %in2_buf
    ```
    is rewritten to:
    ```mlir
    %out_spec = ...
    %out_buf = mgp.cpu.buffer.alloc.spec ..., %out_spec
    %in2_spec = ...
    %in2_size = mgp.cpu.tensor_spec.size %in2_spec
    %in1_buf, %in2_buf = mgp.cpu.buffer.split %out_buf, %in2_size : ...
    %in1_ch = ...
    %in2_ch = ...
    %ch2 = grt.chain.merge %in1_ch, %in2_ch
    ```
  }];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
