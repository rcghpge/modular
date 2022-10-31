//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPDIALECT_TD
#define GENERICML_MOPDIALECT_MOPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MOPDialect
//===----------------------------------------------------------------------===//

def MOPDialect : Dialect {
  let name = "mop";

  let summary = "Buffers, contexts and other framework/kernel independent glue.";
  let description = [{
    The MOP dialect is intended to support buffers, contexts and other base
    functionality needed by graphs containing a mix of framework-specific ops
    (eg tf.add), computational library ops (eg mlas.sgemm), and kernels
    generated from KGEN specifications.

    (Rough design is currently in
    https://docs.google.com/document/d/10oh6cOLdFoFKrlDTbJNVI4pi5vjOoUCZ1AVG2VUTAJ8/edit
    which will be moved into GenericML/docs/mop.md as things firm up.)
  }];

  let cppNamespace = "::M::MOP";
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

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_MOPDIALECT_MOPDIALECT_TD
