//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "KGEN/KGENDialect/KGENDType.h"
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
                                               bool ranked, MlirAttribute *dims,
                                               int32_t rank) {
  llvm::SmallVector<TypedAttr> dim_attrs;
  for (int i = 0; i < rank; ++i) {
    dim_attrs.push_back(unwrap(dims[i]).cast<TypedAttr>());
  }
  return wrap(Module(unwrap(m)).newTensorType(
      unwrap(dtype), ranked, llvm::ArrayRef<TypedAttr>(dim_attrs)));
}

MODULAR_API_EXPORT uint8_t MAXG_tensorTypeGetDType(MlirType t) {
  auto tensorType = unwrap(t).cast<MO::TensorType>();
  return tensorType.getDtype().getValue();
}

MODULAR_API_EXPORT bool MAXG_tensorTypeIsRanked(MlirType t) {
  return unwrap(t).cast<MO::TensorType>().hasRank();
}

MODULAR_API_EXPORT int64_t MAXG_tensorTypeGetRank(MlirType t) {
  return unwrap(t).cast<MO::TensorType>().getRank();
}

MODULAR_API_EXPORT MlirAttribute MAXG_tensorTypeShapeGetDim(MlirType t,
                                                            int64_t dim) {
  return wrap(unwrap(t).cast<MO::TensorType>().getDim(dim));
}

MODULAR_API_EXPORT MlirAttribute MAXG_dimNewDynamic(MlirContext context) {
  return wrap(KGEN::UnknownAttr::get(IndexType::get(unwrap(context))));
}

MODULAR_API_EXPORT MlirAttribute MAXG_dimNewStatic(MlirContext context,
                                                   int64_t dim) {
  return wrap(IntegerAttr::get(IndexType::get(unwrap(context)), dim));
}

MODULAR_API_EXPORT MlirAttribute MAXG_dimNewSymbolic(MlirContext context,
                                                     StringRef name) {
  return wrap(
      KGEN::ParamDeclRefAttr::get(name, IndexType::get(unwrap(context))));
}

MODULAR_API_EXPORT bool MAXG_dimIsDynamic(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  return attr.isa<KGEN::UnknownAttr>();
}

MODULAR_API_EXPORT bool MAXG_dimIsSymbolic(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  return attr.isa<KGEN::ParamDeclRefAttr>();
}

MODULAR_API_EXPORT bool MAXG_dimIsSymbolicExpression(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  return attr.isa<KGEN::ParamOperatorAttr>();
}

MODULAR_API_EXPORT bool MAXG_dimIsStatic(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  return attr.isa<IntegerAttr>();
}

MODULAR_API_EXPORT int64_t MAXG_dimStaticValue(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  if (!attr.isa<IntegerAttr>()) {
    return ShapedType::kDynamic;
  }
  return attr.cast<IntegerAttr>().getInt();
}

MODULAR_API_EXPORT MlirIdentifier MAXG_dimSymbolicName(MlirAttribute a) {
  auto attr = unwrap(a).cast<TypedAttr>();
  if (!attr.isa<KGEN::ParamDeclRefAttr>()) {
    return wrap(StringAttr::get(attr.getContext()));
  }
  return wrap(attr.cast<KGEN::ParamDeclRefAttr>().getName());
}

MODULAR_API_EXPORT MlirType MAXG_listTypeNew(MlirModule m, MlirType eltype) {
  return wrap(Module(unwrap(m)).newListType(unwrap(eltype)));
}

MODULAR_API_EXPORT MlirType MAXG_listTypeElementType(MlirType t) {
  return wrap(unwrap(t).cast<M::MO::ListType>().getElementType());
}

MODULAR_API_EXPORT bool MAXG_typeIsList(MlirType t) {
  return unwrap(t).isa<M::MO::ListType>();
}

MODULAR_API_EXPORT bool MAXG_typeIsTensor(MlirType t) {
  return unwrap(t).isa<M::MO::TensorType>();
}
