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

static Attr denseResourceAttr(StringRef name, ArrayRef<char> data, Type t,
                              bool owned) {
  auto type = dyn_cast<M::MO::TensorType>(t.getVal());

  // Choose a default blob alignment to match the attribute's element size.
  size_t elementByteAlign =
      PowerOf2Ceil(divideCeil(type.getDtype().getWidthInBits(), CHAR_BIT));

  mlir::AsmResourceBlob blob;
  if (owned) {
    blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
        data, elementByteAlign,
        [](void *data, size_t, size_t) { M::alignedFree(data); });
  } else {
    blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
        data, elementByteAlign);
  }

  auto resourceManager =
      M::DenseResourceElementsHandle::getManagerInterface(type.getContext());

  auto shapedType =
      M::RankedTensorType::get(type.getIntDims(), type.getElementType());
  auto denseResource = M::DenseResourceElementsAttr::get(
      shapedType, resourceManager.insert(name, std::move(blob)));
  M::StringAttr nameAttr = M::StringAttr::get(type.getContext(), name);
  return Attr(M::NamedAttribute(nameAttr, denseResource));
}

Module::Module(Location loc) : op(ModuleOp::create(loc)), builder(op) {
  builder.setInsertionPointToStart(&op.getBodyRegion().front());
}

Graph Module::newGraph(Loc loc, StringRef name, const Arity &inTypes,
                       const Arity &outTypes) {
  return Graph(builder.create<M::MO::GraphOp>(
      loc.getVal(), name, inTypes.getVal(), outTypes.getVal()));
}

Type Module::newDType(DType value) {
  auto mlirType = M::MO::toMLIRType(getContext(), M::DType(value));
  assert(!mlirType.isError());
  return Type(*mlirType);
}

Type Module::newTensorType(Type elementType, bool ranked,
                           ArrayRef<int64_t> dims) {
  if (ranked) {
    return Type(M::MO::TensorType::get(dims, elementType.getVal()));
  } else {
    return Type(M::MO::TensorType::get(
        M::KGEN::UnknownAttr::get(M::MOSH::ShapeType::get(getContext())),
        elementType.getVal()));
  }
}

Attr Module::tensorAttr(StringRef name, ArrayRef<uint8_t> data, Type t,
                        bool owned) {
  if (shouldUseOutOfLineAttrStorage(data.size()))
    return denseResourceAttr(
        name, ArrayRef<char>((const char *)data.data(), data.size()), t, owned);

  auto moTensorType = dyn_cast<MO::TensorType>(t.getVal());
  auto shapedType = RankedTensorType::get(moTensorType.getIntDims(),
                                          moTensorType.getElementType());

  auto attr = ArrayElementsAttr::get({data.data(), data.size()}, shapedType);

  if (owned) {
    alignedFree(const_cast<uint8_t *>(data.data()));
  }

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return Attr(NamedAttribute(nameAttr, attr));
}

Attr Module::stringAttr(StringRef name, StringRef value) {
  return Attr(NamedAttribute(StringAttr::get(getContext(), name),
                             StringAttr::get(getContext(), value)));
}

mlir::MLIRContext *Module::getContext() { return op->getContext(); }

Loc Module::unknownLoc() {
  return Loc(mlir::UnknownLoc::get(op->getContext()));
}
