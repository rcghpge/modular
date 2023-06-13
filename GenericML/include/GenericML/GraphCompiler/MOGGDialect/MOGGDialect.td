//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD

include "mlir/IR/DialectBase.td"
//include "GenericML/GraphCompiler/MODialect/MODialect.td"

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

def MOGGDialect : Dialect {
  let name = "mogg";

  let summary = "Modular operations graph generators.";
  let description = [{
    The MOGG Dialect is intended to be a dialect that represents increased
    specification of the algorithms a kernel is implemented with, which enables
    fusion optimizations.

    As a concrete example, there are two implementations of sgemm in mlas: one
    which assumes nothing about the input, and another that assumes input
    tensors are packed with zeros to hit a clean multiple of SIMD width. To
    support these two kernels, we would need to implement two mogg ops:

    - `mogg.matmul`, which would directly lower to mlas.sgemm(x) or kgen.sgemm(x)
    - `mogg.packed_matmul`, which would lower to mlas.packed.sgemm(mlas.pack(x))
      or kgen.packed.sgemm(kgen.pack(x)) It may not be clear which of these
      interfaces is fastest for a given situation. Thus, the lowering from
      MO->MOGG will be a tree-based search that reuses the search and
      elaboration infrastructure developed for KGEN.

    Once we have selected a kernel interface by lowering to MOGG, we have
    sufficient information about the numerics of the kernels to decide if fusion
    will be a numerically safe optimization to perform. MOGG ops will contain
    regions indicating possible fusion opportunities, and an initially
    rule-based fusion pass will fuse kernels by embedding them in the output
    region of the root kernel.

    This fused kernel will then be lowered to KGEN for kernel generation and
    comparison of KGEN and hand-tuned kernel implementations of a given
    numerically specified kernel.

    Once kernels have been selected, MOGG will lower those kernels, generally in
    buffer semantics to the MO dialect for integration with framework kernels,
    memory optimizaiton, and lowering to MEF.
  }];

  let cppNamespace = "::M::MOGG";
  let dependentDialects = ["::M::MO::MODialect"];

  let extraClassDeclaration = [{
  private:
    /// Register all dialect ops.
    void registerOperations();
    /// Register all dialect types.
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
