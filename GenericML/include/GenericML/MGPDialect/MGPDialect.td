//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDIALECT_TD
#define GENERICML_MGPDIALECT_MGPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MGPDialect
//===----------------------------------------------------------------------===//

def MGPDialect : Dialect {
  let name = "mgp";

  let summary = "Buffers, contexts and other framework/kernel independent glue.";
  let description = [{
    The MGP dialect is intended to support buffers, contexts and other base
    functionality needed by graphs containing a mix of framework-specific ops
    (eg tf.add), computational library ops (eg mlas.sgemm), and kernels
    generated from KGEN specifications.

    (Rough design is currently in
    https://docs.google.com/document/d/10oh6cOLdFoFKrlDTbJNVI4pi5vjOoUCZ1AVG2VUTAJ8/edit
    which will be moved into GenericML/docs/mgp.md as things firm up.)
  }];

  let cppNamespace = "::M::MGP";
  let dependentDialects = ["::M::BEF::BEFDialect", "::M::MDialect"];

  let extraClassDeclaration = [{
   private:
    /// Register all dialect types.
    void registerTypes();
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect ops.
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_MGPDIALECT_MGPDIALECT_TD
