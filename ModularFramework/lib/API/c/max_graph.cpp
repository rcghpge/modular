//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "MOF/API/c/symbol_export.h"
#include "MOF/GraphBuilder/Module.h"
#include "Support/Buffer.h"
#include "Support/ML/TensorSpec.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"

using namespace M;
using namespace M::MOF::GraphBuilder;
using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

struct MAXG_Status {
  ErrorOrSuccess status;
};

MODULAR_API_EXPORT MAXG_Status *MAXG_newStatus() {
  auto *status = new MAXG_Status();
  status->status = M::success();
  return status;
}

MODULAR_API_EXPORT const char *MAXG_getError(const MAXG_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.getError();
}

MODULAR_API_EXPORT int MAXG_isError(const MAXG_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.isError();
}

MODULAR_API_EXPORT void MAXG_freeStatus(MAXG_Status *status) { delete status; }

static AsyncRT::Runtime &unwrap(AsyncRTRuntimeRef ref) {
  return *reinterpret_cast<AsyncRT::Runtime *>(ref.ptr);
}

static MlirNamedAttribute wrap(mlir::NamedAttribute attr) {
  return MlirNamedAttribute{
      wrap(attr.getName()),
      wrap(attr.getValue()),
  };
}

MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

// Note: Keep sections below in sync with max/graph/_c.mojo.

//===----------------------------------------------------------------------===//
// Op factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirOperation MAXG_graphNew(MlirModule m, MlirLocation loc,
                                               StringRef name,
                                               MlirType signature) {
  auto module = Module(unwrap(m));
  return wrap(module.newGraph(unwrap(loc), name,
                              cast<FunctionType>(unwrap(signature))));
}

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirNamedAttribute MAXG_weightsAttr(const char *filepath,
                                                       size_t offsetBytes,
                                                       MlirType type,
                                                       StringRef name,
                                                       MAXG_Status *status) {
  auto moType = cast<M::MO::TensorType>(unwrap(type));
  if (!moType.getNumElements().has_value()) {
    status->status = M::Error::getStaticString(
        "tensor attrs should always be statically sized");
    return {{nullptr}, {nullptr}};
  }
  int64_t sizeBytes =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();

  // mmap the provided filepath as read-only.
  auto bufferOr = M::Buffer::getFile(filepath, sizeBytes, offsetBytes);
  if (bufferOr.isError()) {
    status->status = bufferOr.takeError();
    return {{nullptr}, {nullptr}};
  }
  M::BufferRef bufferRef = bufferOr.takeValue();

  auto buff_size = static_cast<int64_t>(bufferRef->getBufferSize());
  if (buff_size != sizeBytes) {
    status->status =
        M::Error(Twine("Size mismatch, want ") + std::to_string(sizeBytes) +
                 " got " + std::to_string(buff_size) + "\n");
    return {{nullptr}, {nullptr}};
  }

  // Choose a default blob alignment to match the attribute's element size.
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(moType.getDtype().getWidthInBits(), CHAR_BIT));

  // Reuses the data within the BufferRef and extends the lifetime of it.
  mlir::AsmResourceBlob blob =
      mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
          arrayRefFromStringRef<char>(bufferRef->getBuffer()), elementByteAlign,
          [storage = bufferRef.copy()](void *, size_t, size_t) {
            // No-op that drops the `bufferRef` when the closure is
            // destroyed.
          });

  auto resourceManager = mlir::DenseResourceElementsHandle::getManagerInterface(
      moType.getContext());

  // Create the dense_resource backed by the mmap'ed file and with the passed
  // tensor shape.
  auto shapedType =
      mlir::RankedTensorType::get(moType.getIntDims(), moType.getElementType());
  auto denseResource = mlir::DenseResourceElementsAttr::get(
      shapedType, resourceManager.insert(name, std::move(blob)));
  mlir::StringAttr nameAttr = mlir::StringAttr::get(moType.getContext(), name);
  return wrap(mlir::NamedAttribute(nameAttr, denseResource));
}

MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensorBorrowed(llvm::StringRef name, void *data, MlirType type) {
  // Create an ND array from data, taking the size and shape from the tensor
  // type argument.
  auto moTensorType = cast<M::MO::TensorType>(unwrap(type));
  size_t size = moTensorType.getDtype().getSizeInBytes() *
                moTensorType.getNumElements().value();
  ArrayRef<uint8_t> bytes_array(llvm::bit_cast<uint8_t *>(data), size);

  auto shapedType = RankedTensorType::get(moTensorType.getIntDims(),
                                          moTensorType.getElementType());

  auto attr = ArrayElementsAttr::get({bytes_array.data(), bytes_array.size()},
                                     shapedType);

  StringAttr nameAttr = StringAttr::get(moTensorType.getContext(), name);
  return wrap(NamedAttribute(nameAttr, attr));
}

MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensor(StringRef name, void *data, MlirType t, bool owned,
                   AsyncRTRuntimeRef runtime) {
  auto moType = cast<M::MO::TensorType>(unwrap(t));
  size_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();
  return wrap(tensorAttr(
      name, ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data), size),
      unwrap(t), owned, unwrap(runtime)));
}

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensorFromFile(
    StringRef name, StringRef fileName, MlirType t, AsyncRTRuntimeRef runtime) {
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
  return wrap(
      tensorAttr(name,
                 ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                                   static_cast<size_t>(size)),
                 unwrap(t), /*owned=*/false, unwrap(runtime)));
}

MODULAR_API_EXPORT MlirAttribute MAXG_attrNewDimParamDecl(MlirContext ctx,
                                                          StringRef name) {
  return wrap(KGEN::ParamDeclAttr::get(name, IndexType::get(unwrap(ctx))));
}

MODULAR_API_EXPORT MlirAttribute MAXG_attrNewParamDeclArray(
    MlirContext ctx, MlirAttribute *attrs, int32_t size) {
  SmallVector<KGEN::ParamDeclAttr> param_attrs;
  for (int i = 0; i < size; ++i)
    param_attrs.push_back(cast<KGEN::ParamDeclAttr>(unwrap(attrs[i])));
  return wrap(KGEN::ParamDeclArrayAttr::get(unwrap(ctx), param_attrs));
}

MODULAR_API_EXPORT MlirAttribute MAXG_attrNewShape(MlirContext ctx,
                                                   MlirAttribute *dims,
                                                   int32_t rank) {
  SmallVector<TypedAttr> dim_attrs;
  for (int i = 0; i < rank; ++i)
    dim_attrs.push_back(cast<TypedAttr>(unwrap(dims[i])));

  auto shapeType = MOSH::ShapeType::get(unwrap(ctx));
  return wrap(MOSH::ShapeAttr::get(unwrap(ctx), dim_attrs, shapeType));
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirType MAXG_dTypeNewFromString(MlirContext ctx,
                                                    const char *value,
                                                    MAXG_Status *status) {
  auto typeOr = M::DType::getFromString(value);
  if (failed(typeOr)) {
    status->status = M::Error(Twine("invalid dtype argument: ") + value);
    return wrap(Type());
  }

  auto mlirTypeOr = MO::toMLIRType(unwrap(ctx), *typeOr);
  if (mlirTypeOr.isError()) {
    status->status = mlirTypeOr.takeError();
    return wrap(Type());
  }

  return wrap(*mlirTypeOr);
}

MODULAR_API_EXPORT MlirType MAXG_dTypeNew(MlirContext ctx, uint8_t value) {
  auto mlirType = M::MO::toMLIRType(unwrap(ctx), M::DType(value));
  assert(!mlirType.isError());
  return wrap(*mlirType);
}

MODULAR_API_EXPORT int64_t MAXG_dimTypeNewDynamic() {
  return ShapedType::kDynamic;
}

MODULAR_API_EXPORT MlirType MAXG_tensorTypeNew(MlirContext ctx, MlirType dtype,
                                               bool ranked, MlirAttribute *dims,
                                               int32_t rank) {
  SmallVector<TypedAttr> dim_attrs;
  for (int i = 0; i < rank; ++i)
    dim_attrs.push_back(cast<TypedAttr>(unwrap(dims[i])));

  if (ranked) {
    return wrap(MO::TensorType::get(dim_attrs, unwrap(dtype)));
  } else {
    return wrap(MO::TensorType::get(
        KGEN::UnknownAttr::get(MOSH::ShapeType::get(unwrap(ctx))),
        unwrap(dtype)));
  }
}

MODULAR_API_EXPORT char *MAXG_tensorTypeGetDTypeAsString(MlirType type) {
  auto tensorType = cast<MO::TensorType>(unwrap(type));
  std::string dtypeStr = tensorType.getDtype().getAsString();
  // Initialize to zeros so that result is null terminated.
  char *result = new char[dtypeStr.size() + 1]{};
  // Copy to a string owned by the caller.
  dtypeStr.copy(result, dtypeStr.size());
  return result;
}

MODULAR_API_EXPORT uint8_t MAXG_tensorTypeGetDType(MlirType t) {
  auto tensorType = cast<MO::TensorType>(unwrap(t));
  return tensorType.getDtype().getValue();
}

MODULAR_API_EXPORT bool MAXG_tensorTypeIsRanked(MlirType t) {
  return cast<MO::TensorType>(unwrap(t)).hasRank();
}

MODULAR_API_EXPORT int64_t MAXG_tensorTypeGetRank(MlirType t) {
  return cast<MO::TensorType>(unwrap(t)).getRank();
}

MODULAR_API_EXPORT MlirAttribute MAXG_tensorTypeShapeGetDim(MlirType t,
                                                            int64_t dim) {
  return wrap(cast<MO::TensorType>(unwrap(t)).getDim(dim));
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
  auto attr = cast<TypedAttr>(unwrap(a));
  return isa<KGEN::UnknownAttr>(attr);
}

MODULAR_API_EXPORT bool MAXG_dimIsSymbolic(MlirAttribute a) {
  auto attr = cast<TypedAttr>(unwrap(a));
  return isa<KGEN::ParamDeclRefAttr>(attr);
}

MODULAR_API_EXPORT bool MAXG_dimIsSymbolicExpression(MlirAttribute a) {
  auto attr = cast<TypedAttr>(unwrap(a));
  return isa<KGEN::ParamOperatorAttr>(attr);
}

MODULAR_API_EXPORT bool MAXG_dimIsStatic(MlirAttribute a) {
  auto attr = cast<TypedAttr>(unwrap(a));
  return isa<IntegerAttr>(attr);
}

MODULAR_API_EXPORT int64_t MAXG_dimStaticValue(MlirAttribute a) {
  auto attr = cast<TypedAttr>(unwrap(a));
  if (!isa<IntegerAttr>(attr)) {
    return ShapedType::kDynamic;
  }
  return cast<IntegerAttr>(attr).getInt();
}

MODULAR_API_EXPORT MlirIdentifier MAXG_dimSymbolicName(MlirAttribute a) {
  auto attr = cast<TypedAttr>(unwrap(a));
  if (!isa<KGEN::ParamDeclRefAttr>(attr)) {
    return wrap(StringAttr::get(attr.getContext()));
  }
  return wrap(cast<KGEN::ParamDeclRefAttr>(attr).getName());
}

MODULAR_API_EXPORT MlirType MAXG_listTypeNew(MlirContext ctx, MlirType eltype) {
  return wrap(MO::ListType::get(unwrap(ctx), unwrap(eltype)));
}

MODULAR_API_EXPORT MlirType MAXG_listTypeElementType(MlirType t) {
  return wrap(cast<M::MO::ListType>(unwrap(t)).getElementType());
}

MODULAR_API_EXPORT bool MAXG_typeIsList(MlirType t) {
  return isa<M::MO::ListType>(unwrap(t));
}

MODULAR_API_EXPORT bool MAXG_typeIsTensor(MlirType t) {
  return isa<M::MO::TensorType>(unwrap(t));
}

MODULAR_API_EXPORT bool MAXG_typeIsOpaque(MlirType t) {
  return isa<M::MO::OpaqueType>(unwrap(t));
}

MODULAR_API_EXPORT MlirType MAXG_opaqueTypeNew(MlirContext context,
                                               StringRef name) {
  StringAttr nameAttr = StringAttr::get(unwrap(context), name);
  return wrap(MO::OpaqueType::get(unwrap(context), nameAttr, std::nullopt));
}

MODULAR_API_EXPORT MlirStringRef MAXG_opaqueTypeName(MlirType t) {
  return wrap(cast<M::MO::OpaqueType>(unwrap(t)).getSymbol().strref());
}
