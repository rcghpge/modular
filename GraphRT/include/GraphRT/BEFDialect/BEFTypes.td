//===- BEFTypes.td - BEF Type definitions ------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines BEF dialect types, like !bef.chain and !bef.unknown
//
//===----------------------------------------------------------------------===//

def ChainType : TypeDef<BEFDialect, "Chain"> {
  let mnemonic = "chain";
  let summary = "empty value used for control dependence tracking";
  let description = [{
    This type is used to model dependences between side-effecting operations,
    by turning these side effects into explicitly modeled values.
  }];
}

def UnknownType : TypeDef<BEFDialect, "Unknown"> {
  let mnemonic = "unknown";
  let summary = "an unknown type";
  let description = [{
    This type is primarily used by MLIR2BEF to reflect parsed BEF files that
    don't have concrete metadata about types in the BEF file.
  }];
}
