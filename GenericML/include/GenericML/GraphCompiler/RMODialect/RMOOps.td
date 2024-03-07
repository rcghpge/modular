//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD

include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/RMODialect/RMODialect.td"
include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"

// Base class for ops in this dialect.
class RMO_Op<string mnemonic, list<Trait> traits = []>
  : Op<RMODialect, mnemonic, traits # [RMO_TensorOpInterface, MO_ParamDeclaration]> {

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

// Base class for ops that map directly to a MO tensor op
class RMO_MOAnaloguePureTensorOp<string mnemonic, list<Trait> traits = []>
  : RMO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>, Pure]> {
  let builders = [
    /// This builder is an implicit return type based on "inferMOTypes" from
    /// the RMO_MOAnalogue interface.
    OpBuilder<
      (ins "ValueRange":$inputValues, "MO::GraphOp":$graphOp),
      [{
        SmallVector<Type> _moOperandTypes, moResultTypes;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;

        inferMOTypesImpl(odsState.getContext(), graphOp, inputValues, std::nullopt, _moOperandTypes,
                                moResultTypes, moInputParamDecls, moOutputParamDecls);

        // Flatten all decls to be used by the main operation
        SmallVector<KGEN::ParamDeclAttr> moOutputParamDeclsFlattened;
        for (const auto& resultDecls: moOutputParamDecls) {
          moOutputParamDeclsFlattened.append(resultDecls);
        }

        // Call normal output type, input value builder
        build($_builder, $_state, mlir::TypeRange{moResultTypes},
                            inputValues);

        // Manually set outputParamDecls field
        $_state.addAttribute(getOutputParamDeclsAttrName($_state.name), $_builder.getAttr<ParamDeclArrayAttr>(moOutputParamDeclsFlattened));
      }]
    >,
  ];

  let extraClassDeclaration = [{

    /// Implementation of `inferMOTypes` using operand adaptor
    static void inferMOTypesImpl(
      MLIRContext* context,
      MO::GraphOp graphOp,
      Adaptor operands,
      std::optional<ValueRange> rmoResults,
      SmallVectorImpl<Type>& moOperandTypes,
      SmallVectorImpl<Type>& moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moOutputParamDecls
    );
  }];
}

//===----------------------------------------------------------------------===//
// RMO Asserting Rebinds
//===----------------------------------------------------------------------===//
def RMO_RebindTensorShapeOp : RMO_Op<"rebind_tensor_shape", [RMO_TensorSameDTypeOperandsAndResults]> {
  let summary = "Rebind the input tensor's shape with runtime check.";
  let description = [{
    Unlike `mo.rebind` this also has the semantics of doing a runtime check
    that the given shapes are compatible.

    Right now this is modeled as a side-effect of the operation due to
    limitations of modeling assertions in our stack. In the future this will
    change and this operation may return either an error or a tensor.
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$output);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "ShapeAttr":$newShape, CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls)>
  ];

  // If we can prove the rebind is rebinding to the same shape, become no-op
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// 1-1 MO Op Analogues
//===----------------------------------------------------------------------===//

class RMO_MOAnaloguePureElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits> {
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureBinaryElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y, MO_ParamDecls:$outputParamDecls);
}

def RMO_AddOp : RMO_MOAnaloguePureBinaryElementWiseOp<"add", [Commutative]> {
  let summary = "Binary addition";
  let description = [{
    Does a non-broadcasted elementwise addition.
  }];
}

def RMO_ShapeOfOp : RMO_MOAnaloguePureTensorOp<"shape_of"> {
  let summary = "Get shape of tensor";
  let description = [{
    Calculates the runtime shape of the given tensor as a tensor.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_IndicesTensor:$shape);
}

def RMO_BroadcastShapeOp : RMO_MOAnaloguePureTensorOp<"broadcast_shape"> {
  let summary = "Apply broadcasting rules";
  let description = [{
    Given two tensors representing shapes, calculate the result of broadcasting
    the shapes.
  }];
  let arguments = (ins MO_ShapeLikeIfRanked:$input_x,
                       MO_ShapeLikeIfRanked:$input_y,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLikeIfRanked:$shape);
}

def RMO_ReshapeOp : RMO_MOAnaloguePureTensorOp<"reshape"> {
  let summary = "Reshape a tensor";
  let description = [{
    Reshape the `input` tensor to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_BroadcastToOp : RMO_MOAnaloguePureTensorOp<"broadcast_to"> {
  let summary = "Broadcast a tensor";
  let description = [{
    Broadcast the given `input` to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_CastOp : RMO_MOAnaloguePureTensorOp<"cast"> {
  let summary = "Cast a tensor to another dtype";
  let description = [{
    Casts the given data, changing from one dtype to another.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

/// TODO: of showing this in it's own namespace.
/// This is an example experimental thing to prove this is easy to write
def RMO_BroadcastAddOp : RMO_Op<"broadcast.add", [Pure, RMO_TensorSameDTypeOperandsAndResults]> {
  let summary = "Add with implicit broadcast Operation";
  let description = [{
    Like `rmo.add` but supports implicit broadcasting!
  }];
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}


#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
