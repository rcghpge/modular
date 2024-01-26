//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/GraphBuilder/Module.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "MOF/GraphBuilder/Symbol.h"
#include "Support/AlignedAlloc.h"
#include "Support/Compiler/MLIRDenseAttr.h"

#include "mlir/IR/Builders.h"

using namespace llvm;

using namespace M::MOF::GraphBuilder;

static mlir::NamedAttribute denseResourceAttr(StringRef name,
                                              ArrayRef<char> data, mlir::Type t,
                                              bool owned) {
  auto type = dyn_cast<M::MO::TensorType>(t);

  // Choose a default blob alignment to match the attribute's element size.
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(type.getDtype().getWidthInBits(), CHAR_BIT));

  mlir::AsmResourceBlob blob;
  if (owned) {
    blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
        data, elementByteAlign,
        // Take ownership of `data`, which must have been allocated with
        // `alignedAlloc()`.
        [](void *data, size_t, size_t) { M::alignedFree(data); });
  } else {
    blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
        data, elementByteAlign);
  }

  auto resourceManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(type.getContext());

  auto shapedType =
      mlir::RankedTensorType::get(type.getIntDims(), type.getElementType());
  auto denseResource = mlir::DenseResourceElementsAttr::get(
      shapedType, resourceManager.insert(name, std::move(blob)));
  mlir::StringAttr nameAttr = mlir::StringAttr::get(type.getContext(), name);
  return mlir::NamedAttribute(nameAttr, denseResource);
}

Module::Module(Location loc) : op(ModuleOp::create(loc)), builder(op) {
  builder.setInsertionPointToStart(&op.getBodyRegion().front());
}

Module::Module(mlir::ModuleOp op) : op(op), builder(op) {
  builder.setInsertionPointToStart(&op.getBodyRegion().front());
}

Graph Module::newGraph(Loc loc, StringRef name, const Arity &inTypes,
                       const Arity &outTypes) {
  return Graph(builder.create<M::MO::GraphOp>(
      loc.getVal(), name, inTypes.getVal(), outTypes.getVal()));
}

mlir::Type Module::newDType(DType value) {
  auto mlirType = M::MO::toMLIRType(getContext(), M::DType(value));
  assert(!mlirType.isError());
  return *mlirType;
}

mlir::Type Module::newTensorType(mlir::Type elementType, bool ranked,
                                 ArrayRef<int64_t> dims) {
  if (ranked) {
    return MO::TensorType::get(dims, elementType);
  } else {
    return MO::TensorType::get(
        KGEN::UnknownAttr::get(MOSH::ShapeType::get(getContext())),
        elementType);
  }
}

mlir::Type Module::f32(ArrayRef<int64_t> shape) {
  return MO::TensorType::get(shape, Float32Type::get(op.getContext()));
}

mlir::Type Module::ui(uint64_t width, ArrayRef<int64_t> shape) {
  return MO::TensorType::get(
      shape, IntegerType::get(op->getContext(), width, IntegerType::Unsigned));
}

mlir::Type Module::si(uint64_t width, ArrayRef<int64_t> shape) {
  return MO::TensorType::get(
      shape, IntegerType::get(op->getContext(), width, IntegerType::Signed));
}

mlir::Type Module::bool_(ArrayRef<int64_t> shape) {
  // There's no constructor for (shape, dtype) yet, but
  // signless with width 1 is hard-coded to end up as a DType::kBool tensor
  // in getEquivalentDtype().
  return MO::TensorType::get(
      shape, IntegerType::get(op->getContext(), 1, IntegerType::Signless));
}

mlir::NamedAttribute Module::tensorAttr(StringRef name, ArrayRef<uint8_t> data,
                                        mlir::Type t, bool owned) {
  if (shouldUseOutOfLineAttrStorage(data.size()))
    return denseResourceAttr(
        name, ArrayRef<char>((const char *)data.data(), data.size()), t, owned);

  auto moTensorType = dyn_cast<MO::TensorType>(t);
  auto shapedType = RankedTensorType::get(moTensorType.getIntDims(),
                                          moTensorType.getElementType());

  auto attr = ArrayElementsAttr::get({data.data(), data.size()}, shapedType);

  if (owned) {
    // Free owned `data`, which was copied into the MLIRContext above.
    alignedFree(const_cast<uint8_t *>(data.data()));
  }

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return NamedAttribute(nameAttr, attr);
}

mlir::NamedAttribute Module::stringAttr(StringRef name, StringRef value) {
  StringAttr nameAttr = StringAttr::get(getContext(), name);
  StringAttr valueAttr = StringAttr::get(getContext(), value);
  return NamedAttribute(nameAttr, valueAttr);
}

mlir::MLIRContext *Module::getContext() { return op->getContext(); }
