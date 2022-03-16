//===- BEFTypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines BEF dialect types, like !bef.unknown.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"

// Base class for the types in this dialect.
class BEFType<string mnemonic, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type"> :
    TypeDef<BEFDialect, mnemonic, traits, baseCppClass>;

//===----------------------------------------------------------------------===//
// UnknownType
//===----------------------------------------------------------------------===//

def UnknownType : BEFType<"Unknown"> {
  let mnemonic = "unknown";
  let summary = "an unknown type";
  let description = [{
    This type is primarily used by MLIR2BEF to reflect parsed BEF files that
    don't have concrete metadata about types in the BEF file.
  }];
}
