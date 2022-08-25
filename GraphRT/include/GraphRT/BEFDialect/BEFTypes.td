//===- BEFTypes.td --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines BEF dialect types, like !bef.unknown and !bef.chain.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "GraphRT/BEFDialect/BEFDialect.td"

// Base class for the types in this dialect.
class BEFType<string mnemonic, list<Trait> traits = [],
              string baseCppClass = "Type"> :
    TypeDef<BEFDialect, mnemonic, traits, baseCppClass>;

//===----------------------------------------------------------------------===//
// ChainType
//===----------------------------------------------------------------------===//

def ChainType : BEFType<"Chain"> {
  let mnemonic = "chain";
  let summary = "empty value used for control dependence tracking";
  let description = [{
    This type is used to model dependencies between side-effecting operations,
    by turning these side effects into explicitly modeled values.
  }];
}

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
