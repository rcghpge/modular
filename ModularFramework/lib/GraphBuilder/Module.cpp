//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/GraphBuilder/Module.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "LLCL/Runtime/Runtime.h"
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

mlir::Operation *Module::newGraph(mlir::Location loc, StringRef name,
                                  const mlir::FunctionType signature) {
  return builder
      .create<M::MO::GraphOp>(loc, name, signature.getInputs(),
                              signature.getResults())
      .getOperation();
}

mlir::Type Module::newDType(DType value) {
  auto mlirType = M::MO::toMLIRType(getContext(), M::DType(value));
  assert(!mlirType.isError());
  return *mlirType;
}

mlir::Type Module::newTensorType(mlir::Type elementType, bool ranked,
                                 ArrayRef<TypedAttr> dims) {
  if (ranked) {
    return MO::TensorType::get(dims, elementType);
  } else {
    return MO::TensorType::get(
        KGEN::UnknownAttr::get(MOSH::ShapeType::get(getContext())),
        elementType);
  }
}

mlir::Type Module::newListType(mlir::Type elementType) {
  return MO::ListType::get(getContext(), elementType);
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
    // Free owned `data`, which was copied into the MLIRContext above. Note that
    // this data was allocated in mojo land, which uses the runtime allocator.
    // So, we need to use the runtime deallocator to free.
    auto rt = Runtime::getCurrentRuntimeOrNull();
    assert(rt && "runtime not present");
    rt->getAllocator()->deallocateBytes(const_cast<uint8_t *>(data.data()));
  }

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return NamedAttribute(nameAttr, attr);
}

mlir::MLIRContext *Module::getContext() { return op->getContext(); }
