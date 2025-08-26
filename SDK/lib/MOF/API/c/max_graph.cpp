//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h"
#include "SDK/MOF/API/c/symbol_export.h"
#include "SDK/MOF/GraphBuilder/Module.h"
#include "Support/Buffer.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"

using namespace M;
using namespace M::MOF::GraphBuilder;
using namespace llvm;
using namespace mlir;

MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

static MlirNamedAttribute wrap(mlir::NamedAttribute attr) {
  return MlirNamedAttribute{
      wrap(attr.getName()),
      wrap(attr.getValue()),
  };
}

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensorBorrowed(llvm::StringRef name, void *data, MlirType type) {
  // Create an ND array from data, taking the size and shape from the tensor
  // type argument.
  auto moTensorType = cast<M::MO::TensorType>(unwrap(type));
  size_t size = moTensorType.getDtype().getSizeInBytes() *
                moTensorType.getNumElements().value();
  ArrayRef<uint8_t> bytesArray(llvm::bit_cast<uint8_t *>(data), size);

  auto shapedType = RankedTensorType::get(moTensorType.getIntDims(),
                                          moTensorType.getElementType());

  auto attr = ArrayElementsAttr::get({bytesArray.data(), bytesArray.size()},
                                     shapedType);

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return wrap(NamedAttribute(nameAttr, attr));
}

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensor(StringRef name,
                                                         void *data, MlirType t,
                                                         bool owned) {
  auto runtime = Runtime::getCurrentRuntimeOrNull();
  auto moType = cast<M::MO::TensorType>(unwrap(t));
  size_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();
  return wrap(tensorAttr(
      name, ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data), size),
      unwrap(t), owned, *runtime));
}

MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensorFromFile(StringRef name, StringRef fileName, MlirType t) {
  auto runtime = Runtime::getCurrentRuntimeOrNull();
  auto moType = cast<M::MO::TensorType>(unwrap(t));
  assert(moType.getNumElements().has_value() &&
         "tensor attrs should always be statically sized");
  int64_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();

  auto bufferOr = M::Buffer::getFile(std::filesystem::path{fileName.str()});
  if (bufferOr.isError())
    errs() << bufferOr.takeError() << "\n";
  assert(!bufferOr.isError());
  M::BufferRef buffer = bufferOr.takeValue();

  auto buffSize = static_cast<int64_t>(buffer->getBufferSize());
  if (buffSize != size)
    errs() << "Size mismatch, want " << size << " got " << buffSize << "\n";
  assert(buffSize == size);

  auto data = buffer->getBufferStart();
  return wrap(
      tensorAttr(name,
                 ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                                   static_cast<size_t>(size)),
                 unwrap(t), /*owned=*/false, *runtime));
}

MODULAR_API_EXPORT MlirLocation MAXG_opaqueFrameLocNew(
    MlirContext context, void *frame_ptr, char *(*to_str_fn)(void *),
    void (*incref_fn)(void *), void (*decref_fn)(void *)) {
  mlir::Location loc =
      RMO::PythonFrameAttr::getOpaque(unwrap(context), {.ptr = frame_ptr,
                                                        .to_str = to_str_fn,
                                                        .incref = incref_fn,
                                                        .decref = decref_fn});
  return wrap(loc);
}

MODULAR_API_EXPORT void *MAXG_opaqueFrameLocGetFrame(MlirLocation loc) {
  if (!mlir::isa<OpaqueLoc>(unwrap(loc)))
    return nullptr;

  void *attrPtr = mlir::OpaqueLoc::getUnderlyingLocationOrNull<
      RMO::PythonFrameAttr::ImplType *>(unwrap(loc));
  if (!attrPtr)
    return nullptr;

  auto attr = RMO::PythonFrameAttr::getFromOpaquePointer(attrPtr);
  return attr.frame();
}
