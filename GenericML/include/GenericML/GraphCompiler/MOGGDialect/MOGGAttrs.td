//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.td"

class MOGGAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOGGDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// Base class for fusion group attributes that have children and a fusionName.
class MOGGFusionGroupAttr<string attrName, string attrMnemonic,
                          string name>
    : MOGGAttr<attrName, attrMnemonic> {
  let extraClassDeclaration = [{
    static constexpr llvm::StringLiteral getFusionName() { return "}] # name # [{"; }
  }];
}

//===----------------------------------------------------------------------===//
// FusionInfo: Leaf Attributes
//===----------------------------------------------------------------------===//

def MOGG_FusionLeafAttr : MOGGAttr<"FusionLeaf", "fusion_leaf"> {
  let summary = "An attribute that represents a single operation info.";
  let description = [{
    This attribute holds information for a single leaf op in a fusion info tree.
  }];

  let parameters = (ins "mlir::StringAttr":$opName,
                        OptionalParameter<"uint32_t">:$lineNumber);
  let assemblyFormat = "`<` $opName (`,` `line` `=` $lineNumber^)? `>`";
}

//===----------------------------------------------------------------------===//
// FusionInfo: Group Attributes
//===----------------------------------------------------------------------===//

def MOGG_FusionSequenceAttr : MOGGFusionGroupAttr<"FusionSequence", "fusion_sequence", "Sequence"> {
  let summary = "An attribute that represents sequence fusion info.";
  let description = [{
    This attribute holds information for sequence fusion.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionElementwiseAttr : MOGGFusionGroupAttr<"FusionElementwise", "fusion_elementwise", "Elementwise"> {
  let summary = "An attribute that represents elementwise fusion info.";
  let description = [{
    This attribute holds information for elementwise fusion.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionEpilogueAttr : MOGGFusionGroupAttr<"FusionEpilogue", "fusion_epilogue", "Epilogue"> {
  let summary = "An attribute that represents epilogue fusion info.";
  let description = [{
    This attribute holds information for epilogue fusion.
    The operandIndices tracks which outputs were fused.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children,
                        ArrayRefParameter<"uint32_t">:$operandIndices);
  let assemblyFormat = "`<` `[` $children `]` `,` `operandIndices` `=` `[` $operandIndices `]` `>`";
}

def MOGG_FusionPrologueAttr : MOGGFusionGroupAttr<"FusionPrologue", "fusion_prologue", "Prologue"> {
  let summary = "An attribute that represents prologue fusion info.";
  let description = [{
    This attribute holds information for prologue fusion.
    The operandIndices tracks which inputs were fused.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children,
                        ArrayRefParameter<"uint32_t">:$operandIndices);
  let assemblyFormat = "`<` `[` $children `]` `,` `operandIndices` `=` `[` $operandIndices `]` `>`";
}

def MOGG_FusionViewAttr : MOGGFusionGroupAttr<"FusionView", "fusion_view", "View"> {
  let summary = "An attribute that represents view fusion info.";
  let description = [{
    This attribute holds information for view fusion.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionAffineAttr : MOGGFusionGroupAttr<"FusionAffine", "fusion_affine", "Affine"> {
  let summary = "An attribute that represents affine fusion info.";
  let description = [{
    This attribute holds information for affine view fusion.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionMutableLoadAttr : MOGGFusionGroupAttr<"FusionMutableLoad", "fusion_mutable_load", "MutableLoad"> {
  let summary = "An attribute that represents mutable load fusion info.";
  let description = [{
    This attribute holds information for mutable load fusion.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr - the root in the fusion tree info stored on ops
//===----------------------------------------------------------------------===//

def MOGG_FusionInfoAttr : MOGGAttr<"FusionInfo", "fusion"> {
  let summary = "An attribute that holds the fusion info tree.";
  let description = [{
    This attribute wraps a fusion info tree and provides a pretty-printed
    format using function-call syntax:

    ```mlir
    fusionInfo = Sequence(View(mo.slice:42), Epilogue(mo.matmul, Elementwise(mo.relu, mo.add), operandIndices=[0]))
    ```

    The `root` parameter can be any of:
    - FusionLeafAttr: single operation
    - FusionSequenceAttr: soft fusion (multiple kernels)
    - FusionElementwiseAttr: hard elementwise fusion
    - FusionEpilogueAttr: epilogue fusion (with optional operand index)
    - FusionPrologueAttr: prologue fusion (with optional operand index)
    - FusionViewAttr: view fusion
    - FusionAffineAttr: affine view fusion
    - FusionMutableLoadAttr: mutable load fusion
  }];

  let parameters = (ins "mlir::Attribute":$root);

  let assemblyFormat = "`<` $root `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Flatten the fusion info tree to a string.
    std::string toString(bool includeLineNumbers = false) const;

    /// Return a multi-line, indented tree visualization of the fusion info.
    std::string toPrettyTree() const;
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD
