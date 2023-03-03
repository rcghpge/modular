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

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
