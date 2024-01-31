//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "MOF/API/c/symbol_export.h"
#include "Support/Buffer.h"
#include "Support/ML/TensorSpec.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/bit.h"

using namespace M;
using namespace M::MOF::GraphBuilder;
using namespace llvm;
using namespace mlir;

MlirNamedAttribute wrap(mlir::NamedAttribute attr) {
  return MlirNamedAttribute{
      wrap(attr.getName()),
      wrap(attr.getValue()),
  };
}

mlir::NamedAttribute unwrap(MlirNamedAttribute attr) {
  return {unwrap(attr.name), unwrap(attr.attribute)};
}

MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

// Note: Keep sections below in sync with max_graph.cpp.

//===----------------------------------------------------------------------===//
// Op factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirOperation MAXG_graphNew(MlirModule m, MlirLocation loc,
                                               StringRef name,
                                               MlirType signature) {
  auto module = Module(unwrap(m));
  return wrap(module.newGraph(unwrap(loc), name,
                              unwrap(signature).cast<FunctionType>()));
}

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensor(MlirModule m,
                                                         StringRef name,
                                                         void *data, MlirType t,
                                                         bool owned) {
  auto moType = cast<M::MO::TensorType>(unwrap(t));
  size_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();
  return wrap(Module(unwrap(m)).tensorAttr(
      name, ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data), size),
      unwrap(t), owned));
}

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensorFromFile(
    MlirModule m, StringRef name, StringRef fileName, MlirType t) {
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

  auto buff_size = static_cast<int64_t>(buffer->getBufferSize());
  if (buff_size != size)
    errs() << "Size mismatch, want " << size << " got " << buff_size << "\n";
  assert(buff_size == size);

  auto data = buffer->getBufferStart();
  return wrap(Module(unwrap(m)).tensorAttr(
      name,
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                        static_cast<size_t>(size)),
      unwrap(t), /*owned=*/false));
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirType MAXG_dTypeNew(MlirModule m, uint8_t value) {
  return wrap(Module(unwrap(m)).newDType(M::DType(value)));
}

MODULAR_API_EXPORT int64_t MAXG_dimTypeNewDynamic() {
  return ShapedType::kDynamic;
}

MODULAR_API_EXPORT MlirType MAXG_tensorTypeNew(MlirModule m, MlirType dtype,
                                               bool ranked, int64_t *dims,
                                               int32_t rank) {
  return wrap(Module(unwrap(m)).newTensorType(
      unwrap(dtype), ranked, llvm::ArrayRef<int64_t>(dims, rank)));
}

MODULAR_API_EXPORT uint8_t MAXG_tensorTypeGetDType(MlirValue v) {
  auto tensorType = unwrap(v).getType().cast<MO::TensorType>();
  return tensorType.getDtype().getValue();
}

MODULAR_API_EXPORT int64_t *MAXG_tensorTypeGetShape(MlirValue v,
                                                    int32_t *rank) {
  auto tensorType = unwrap(v).getType().cast<MO::TensorType>();
  SmallVector<int64_t> dims = tensorType.getIntDims();
  *rank = dims.size();
  int64_t *retval = new int64_t[*rank];
  for (size_t i = 0; i < dims.size(); i++)
    retval[i] = dims[i];
  return retval;
}

MODULAR_API_EXPORT bool MAXG_tensorTypeIsRanked(MlirValue v) {
  auto tensorType = unwrap(v).getType().cast<MO::TensorType>();
  return tensorType.getDtype().getValue();
}

MODULAR_API_EXPORT MlirType MAXG_listTypeNew(MlirModule m, MlirType eltype) {
  return wrap(Module(unwrap(m)).newListType(unwrap(eltype)));
}
