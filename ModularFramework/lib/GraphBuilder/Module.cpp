//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/GraphBuilder/Module.h"

#include "AsyncRT/Runtime/Runtime.h"
#include "Support/Compiler/MLIRDenseAttr.h"

#include "mlir/IR/Builders.h"

using namespace llvm;
using namespace M;
using namespace M::MOF;
using namespace M::MOF::GraphBuilder;

static mlir::NamedAttribute denseResourceAttr(StringRef name,
                                              ArrayRef<char> data, mlir::Type t,
                                              bool owned,
                                              LLCL::Runtime &runtime) {
  auto type = dyn_cast<M::MO::TensorType>(t);

  // Choose a default blob alignment to match the attribute's element size.
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(type.getDtype().getWidthInBits(), CHAR_BIT));

  mlir::AsmResourceBlob blob;
  if (owned) {
    // Take ownership of `data`, which must have been allocated with
    // `Singleton CurrentRuntime` and deallocate it with the current
    // allocator.
    M::Allocator *currentAllocator = runtime.getAllocator();
    blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
        data, elementByteAlign, [currentAllocator](void *data, size_t, size_t) {
          currentAllocator->deallocateBytes(data);
        });
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

mlir::NamedAttribute GraphBuilder::tensorAttr(StringRef name,
                                              ArrayRef<uint8_t> data,
                                              mlir::Type t, bool owned,
                                              LLCL::Runtime &runtime) {
  if (shouldUseOutOfLineAttrStorage(data.size()))
    return denseResourceAttr(
        name, ArrayRef<char>((const char *)data.data(), data.size()), t, owned,
        runtime);

  auto moTensorType = dyn_cast<MO::TensorType>(t);
  auto shapedType = RankedTensorType::get(moTensorType.getIntDims(),
                                          moTensorType.getElementType());

  auto attr = ArrayElementsAttr::get({data.data(), data.size()}, shapedType);

  if (owned) {
    // Free owned `data`, which was copied into the MLIRContext above. Note that
    // this data was allocated in mojo land, which uses the global runtime
    // allocator. So, we need to use the same global runtime deallocator to
    // free.
    runtime.getAllocator()->deallocateBytes(const_cast<uint8_t *>(data.data()));
  }

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return NamedAttribute(nameAttr, attr);
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

mlir::MLIRContext *Module::getContext() { return op->getContext(); }
