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
include "mlir/IR/EnumAttr.td"
include "GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/MOGGDialect/MOGGEnums.td"
include "KGEN/LITDialect/LITTypes.td"

/// Base class for MOGG dialect location attributes: dialect-provided
/// attributes that subclass `mlir::LocationAttr` (see `LocationAttrDef` in
/// `mlir/IR/AttrTypeBase.td`), so `mlir::isa<LocationAttr>` recognizes them
/// and they can be used anywhere an `mlir::Location` is expected.
class MOGGLocationAttr<string attrName, string attrMnemonic,
                       list<Trait> traits = []>
    : LocationAttrDef<MOGGDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class MOGGAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOGGDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// ProfileScopeLocationAttr
//===----------------------------------------------------------------------===//

def MOGG_ProfileScopeLocationAttr
    : MOGGLocationAttr<"ProfileScopeLocation", "profile_scope"> {
  let summary = "A `Graph.profile_scope()` scope-name annotation on a Location";
  let description = [{
    `Graph.profile_scope(name)` wraps the MLIR location of every op created
    inside its `with` block in one of these per active scope, nesting
    outward: the *last*-entered (innermost) scope ends up as the outermost
    `ProfileScopeLocationAttr` layer. Riding on `Location` rather than a
    magic string prefix keeps this strongly typed and backend-neutral.
  }];

  let parameters = (ins "mlir::StringAttr":$name,
                        "mlir::LocationAttr":$childLoc,
                        OptionalParameter<"mlir::StringAttr">:$color);
  let assemblyFormat = "`<` $name `,` $childLoc (`,` `color` `=` $color^)? `>`";
}

//===----------------------------------------------------------------------===//
// TensorArgContractAttr -- the trait-lane tensor-argument contract
//===----------------------------------------------------------------------===//

def MOGG_TensorArgContractAttr : MOGGAttr<"TensorArgContract", "tensor_arg"> {
  let summary = "Distilled trait contract for a trait-bound tensor argument";
  let description = [{
    The single source of truth carried by the trait lane for one trait-bound
    tensor argument (`#mogg.tensor_arg<{role, fused, access, capabilities}>`).
    It mirrors the per-argument `DictionaryAttr` baked onto
    `KernelFunctionAttr.tensor_arg_contracts` at import
    (`tensorArgContractToDict` / `dictToTensorArgContract` in
    `KernelDeclAdaptor.cpp`), and rides on the `mogg.tensor_arg.__init__` op in
    place of the legacy `{io, mut}` carried by `mogg.tensor.__init__`.

      - `role`         : `input` / `output` / `mutable_input`. `mut` is not
                         stored separately -- an argument is mutable iff its
                         role is not `input` (D2).
      - `fused`        : whether the argument is a fused (functor-backed)
                         operand.
      - `access`       : the SET of element-access forms (a bit enum, empty
                         set `none`, or any combination of `load` / `store` /
                         `transform`). A non-fused dense argument may carry
                         several (e.g. `load|store` for in-place read-modify-
                         write); a fused argument carries exactly one.
      - `capabilities` : the subset of `access` for which the callable
                         capability leaf (e.g. `Loadable`) was required, rather
                         than only the access marker (e.g. `LoadAccess`).
      - `bound`        : the deepest tensor-argument trait bound written on
                         a shape-function tensor argument (`tensor`,
                         `dense`, or `tile`). Every execute-lane argument
                         carries `unspecified`. Printing elides `bound` at
                         `unspecified`. Every existing execute-lane attr
                         then still prints exactly as before.

    Per D4, this contract is the only home for the role/access information: it
    is not stamped onto the result `!mo.tensor` type, and no `MO_IOKind` case is
    added.
  }];

  let parameters = (ins
    EnumParameter<MOGG_TensorArgRole>:$role,
    "bool":$fused,
    EnumParameter<MOGG_TensorArgAccess>:$access,
    EnumParameter<MOGG_TensorArgAccess>:$capabilities,
    DefaultValuedParameter<"::M::MOGG::TensorArgBoundKind",
        "::M::MOGG::TensorArgBoundKind::unspecified">:$bound
  );

  let assemblyFormat = [{
    `<` `{` `role` `=` $role `,` `fused` `=` $fused `,` `access` `=` $access
        `,` `capabilities` `=` $capabilities (`,` `bound` `=` $bound^)? `}` `>`
  }];

  let extraClassDeclaration = [{
    /// `mut` is derived, never stored: an argument is mutable iff its role is
    /// not `input` (mirrors `TensorArgContract::isMutable`).
    bool isMutable() const {
      return getRole() != ::M::MOGG::TensorArgRoleKind::input;
    }
  }];
}

//===----------------------------------------------------------------------===//
// KernelFunctionAttr
//===----------------------------------------------------------------------===//

def MOGG_KernelFunctionAttr : MOGGAttr<"KernelFunction", "kernel_function"> {
  let summary = "Bundles a Mojo function signature with its fn-level "
                "parameter declarations and cached slot-name mappings";
  let description = [{
    Holds, for one function variant of a Mojo kernel
    (`execute` / `shape` / `elementwise` / `update_view`):

      - `signature`     : the LIT `FnTypeGeneratorType` for the function.
      - `params`        : the def-level `ParamDeclAttr` list for this
                          function (each entry has `getName()` and
                          `getType()`).
      - `arg_params`    : per-argument `{struct-slot-name -> bound-value}`
                          `DictionaryAttr`s. Each entry corresponds to one
                          argument of the function; a `UnitAttr` placeholder
                          is used for arguments whose struct decl is not
                          present in the loaded package (e.g. tensors and
                          device-context types live in dependency packages
                          we do not load). User-defined parameterised
                          opaque structs are reliably resolved.
      - `result_params` : same shape, applied to result types. Stricter
                          than `arg_params`: result types must be either
                          non-parameterised, in the loaded package, or in
                          the hardcoded stdlib table — otherwise the import
                          pass errors out.
      - `tensor_arg_contracts` : per-argument tensor-trait contract for
                          trait-bound tensor arguments (the trait lane). For
                          a trait-bound argument, each entry is a
                          `DictionaryAttr` with keys `role`, `fused`,
                          `access`, and `capabilities`. `access` and
                          `capabilities` are integer bitmasks of access
                          forms. A shape-function argument also carries a
                          `bound` key (`tensor`, `dense`, or `tile`); every
                          execute-lane argument omits this key. A non-trait
                          argument uses a `UnitAttr` placeholder instead.
                          The array is empty when the function has no
                          trait-bound arguments, so `ManagedTensorSlice`-
                          lane kernels are unaffected.
                          `distillTensorArgContract` distills this array.

    `arg_params` powers opaque-parameter inference in `MOToMOGGLowering`:
    given an `MO::OpaqueType` operand carrying `{S0 = 4, S1 = 8}` and a
    kernel argument typed `SIMDPair[S0 = P0, S1 = P1]`, the slot dict
    `{S0 -> ParamDeclRef<P0>, S1 -> ParamDeclRef<P1>}` lets the lowering
    thread `P0 = 4, P1 = 8` into the kernel call.

    `result_params` is consumed by lookups like "for the result type's
    `dtype` slot, which function parameter is bound there?" — needed for
    `shape` and `elementwise` only; left empty for `execute`, `update_view`,
    and primitives.

    `params` is kept because `FnTypeGeneratorType.getMetadata()` carries
    names and defaults but not declared types, so reconstructing
    `ParamDeclAttr`s from the signature alone is not possible.
  }];

  let parameters = (ins
    "::M::KGEN::LIT::FnTypeGeneratorType":$signature,
    "::mlir::ArrayAttr":$params,
    "::mlir::ArrayAttr":$arg_params,
    "::mlir::ArrayAttr":$result_params,
    DefaultValuedParameter<"::mlir::ArrayAttr",
                           "::mlir::ArrayAttr::get($_ctxt, {})">:
        $tensor_arg_contracts
  );

  let assemblyFormat = [{
    `<` $signature `,` $params `,` $arg_params `,` $result_params
        (`,` $tensor_arg_contracts^)?
    `>`
  }];
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
    Stores the op's location directly so the fusion tree carries full provenance
    info. The FusedLoc on the outer kernel is still maintained separately for
    compiler debugging purposes.
  }];

  let parameters = (ins "mlir::StringAttr":$opName,
                        OptionalParameter<"mlir::LocationAttr">:$loc,
                        OptionalParameter<"mlir::TypeAttr">:$signature);
  let assemblyFormat = "`<` $opName (`,` `loc` `=` $loc^)? (`,` `sig` `=` $signature^)? `>`";
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
    std::string toString(bool includeLineNumbers = false, bool quote = false,
                         bool stripDialectPrefixes = false) const;

    /// Return a multi-line, indented tree visualization of the fusion info.
    /// If \p highlightLeaves is non-empty, leaf nodes present in the set are
    /// annotated with "<-- error" in the output. Matching is by attribute
    /// identity, not by op name, so same-named ops are disambiguated.
    /// If \p indent is true, each line is prefixed with two spaces.
    std::string toPrettyTree(
        const llvm::DenseSet<mlir::Attribute> &highlightLeaves = {},
        bool indent = false) const;
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD
