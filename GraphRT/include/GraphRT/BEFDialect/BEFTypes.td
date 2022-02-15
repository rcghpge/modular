//===- BEFTypes.td - BEF Type definitions ------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines BEF::UnknownType.
//
//===----------------------------------------------------------------------===//

def UnknownType : TypeDef<BEFDialect, "Unknown"> {
  let mnemonic = "unknown";
  let summary = "an unknown type";
  let description = [{
    This type is primarily used by MLIR2BEF to reflect parsed BEF files that
    don't have concrete metadata about types in the BEF file.
  }];
}
