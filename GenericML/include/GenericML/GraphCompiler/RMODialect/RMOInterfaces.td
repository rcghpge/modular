//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces in the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD

include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def RMO_TensorOpInterface : OpInterface<"RMOOp", [MO_TensorOpInterface, MO_DefaultParameterization]> {
  let description = [{
    Interface for easy pattern matching on all RMO Ops.

    Each RMO op needs to have a parent `mo.graph_op`. This is needed to
    maintain unique shape parameters.

    Each RMO op also has a method to resolve unknown parameters. This is done
    by desugaring `?` in the shape of results. We reuse
    MO_DefaultParameterization interface to do this.
  }];
  let cppNamespace = "::M::RMO";
  let verify = [{ return verifyRMOOp(cast<RMOOp>($_op)); }];
}

def RMO_MOAnalogue : OpInterface<"MOAnalogue"> {
  let description = [{
    An RMO operation which maps 1-1 in semantics to an existing MO operation.

    Such RMO operations should have the same semantics as the MO operation
    including the number and type of operands and results.

    The MO operation however will have stricter verifiers (since they must
    guaranteed to be statically compatible). Therefore, each analogue must find
    a way to reconcile the types given in the RMO graph to something that
    type-checks MO.

    To provide this, we declare an `inferMOTypes` method which given the
    current input types, figures out the types to both the MO operands and
    results.
  }];
  let cppNamespace = "::M::RMO";
  let methods = [
    InterfaceMethod<"Given an RMO operator with an analogous MO "
    "operation, return the input and output types of the analogous MO "
    "operation. These types should always pass the associated MO operation's "
    "verifier. Under the hood, the `rmoOperands` will be rebound to the given "
    "types before being fed into the analogous MO operator. Therefore, the "
    "types in `rmoOperands` and `moOperandTypes` should differ only on shape. "
    "If `rmoResults` is present, only shapes should differ between analogous"
    "elements to the returned `moResultTypes`."
    "Note, in the process of determining rebound MO types we may declare new "
    "shape parameters. Any new declarations belong in `moInputParamDecls` and "
    "`moOutputParamDecls` respectively, where `moInputParamDecls[i] and "
    "moOutputParamDecls[j] are the new parameters for the ith operand and jth "
    "result respectively. If a new parameter is used more than once, it "
    "should be declared in the earliest operand/result only."
    "This method must always populate `mo*` fields with values which can "
    "construct the analogous MO operation.",

      /*retTy=*/"void",
      /*methodName=*/"inferMOTypes",
      /*args=*/(ins "MO::GraphOp":$graphOp,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$moOperandTypes,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$moResultTypes,
                    "::llvm::SmallVectorImpl<SmallVector<::M::KGEN::ParamDeclAttr>>&":$moInputParamDecls,
                    "::llvm::SmallVectorImpl<SmallVector<::M::KGEN::ParamDeclAttr>>&":$moOutputParamDecls),
      /*methodBody=*/ [{
        typename ConcreteOp::Adaptor adaptor(
            $_op->getOperands(),
            $_op->getAttrDictionary(),
            $_op->getPropertiesStorage(),
            $_op->getRegions()
        );

        ConcreteOp::inferMOTypesImpl(
          $_op->getContext(),
          graphOp,
          adaptor,
          $_op->getResults(),
          moOperandTypes,
          moResultTypes,
          moInputParamDecls,
          moOutputParamDecls
        );
      }]
    >,

    // NOTE: This is used to guarantee all ops have this as an attribute.
    InterfaceMethod<"Get the list of output parameters declared by the RMO op.",
      /*retTy=*/"::llvm::ArrayRef<ParamDeclAttr>",
      /*methodName=*/"getMOAnalogueOutputParamDecls",
      /*args=*/(ins),
      [{ return $_op.getOutputParamDecls(); }]
    >,

    InterfaceMethod<"Get the name of the output parameters attribute.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getMOAnalogueOutputParamDeclsAttrName",
      /*args=*/(ins),
      [{ return $_op.getOutputParamDeclsAttrName(); }]
    >,
  ];
}

def RMO_TensorSameDTypeOperandsAndResults : OpInterface<"TensorSameDTypeOperandsAndResults"> {
  let description = [{
    Interface for RMO ops where all tensor types in the operands and results
    have the same dtype. Also checks at least one tensor type operand/result.
  }];
  let cppNamespace = "::M::RMO";
  let verify = [{ return verifySameDTypeOperandsAndResults(cast<RMOOp>($_op)); }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_TD
