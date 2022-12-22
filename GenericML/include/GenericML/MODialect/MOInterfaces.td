//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOINTERFACES_TD
#define GENERICML_MODIALECT_MOINTERFACES_TD

include "mlir/IR/OpBase.td"

def MO_MatmulLike : OpInterface<"MatmulLike"> {
  let description = [{
    Interface for modeling operations that implement matmul-like behavior,
    including vanilla matmul and batchmatmul.

    Dynamically ranked/shaped inputs and results are permitted, but if the shape
    is known, the last 2 dimensions of each input are assumed to be the matrix
    dimension.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get the left input", "Value", "getInputA", (ins)>,
    InterfaceMethod<"Get the right input", "Value", "getInputB", (ins)>,
    InterfaceMethod<"Get the result", "Value", "getResult", (ins)>,
    InterfaceMethod<"Get the left input tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorA", (ins), [{}],
      [{ return TypedValue<TensorType>($_op.getInputA()); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorB", (ins), [{}],
      [{ return TypedValue<TensorType>($_op.getInputB()); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorResult", (ins), [{}],
      [{ return TypedValue<TensorType>($_op.getResult()); }]>,
    InterfaceMethod<"Get the shape of the left tensor",
      "ArrayRef<int64_t>", "getShapeA", (ins), [{}],
      [{ return $_op.getTensorA().getType().getShape(); }]>,
    InterfaceMethod<"Get the shape of the right tensor",
      "ArrayRef<int64_t>", "getShapeB", (ins), [{}],
      [{ return $_op.getTensorB().getType().getShape(); }]>,
    InterfaceMethod<"Get the shape of the result tensor",
      "ArrayRef<int64_t>", "getShapeResult", (ins), [{}],
      [{ return $_op.getTensorResult().getType().getShape(); }]>,
  ];

  let verify = [{ return impl::verifyMatmulLike($_op); }];
}


#endif // GENERICML_MODIALECT_MOINTERFACES_TD
