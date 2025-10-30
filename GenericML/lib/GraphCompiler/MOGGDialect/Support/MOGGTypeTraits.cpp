//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MO supported types and conversion to KGEN types
//
// All utilities around dtype conversion for KGEN and MO are here so that there
// is only one place to add a new dtype.
//
// If you want to add a new supported type, add it to `allDTypeTraits` below
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "llvm/ADT/APFloat.h"

using namespace M;
using namespace M::MOGG;

namespace {

llvm::SmallVector<mlir::TypedAttr>
convertFloatsToSIMD(DType moggType, KGEN::KGENDType kgenType,
                    mlir::MLIRContext *context, const void *data,
                    int64_t numElems, const llvm::fltSemantics &semantic) {
  auto simdType = KGEN::POP::SIMDType::get(context, 1, moggType);
  auto sizeInBytes = moggType.getSizeInBytes();

  llvm::SmallVector<mlir::TypedAttr> values;
  values.resize_for_overwrite(numElems);

  APInt loadedValue(sizeInBytes * CHAR_BIT, 0);

  for (auto i : llvm::seq(numElems)) {
    auto valuePtr = reinterpret_cast<const uint8_t *>(data) + sizeInBytes * i;
    llvm::LoadIntFromMemory(loadedValue, valuePtr, sizeInBytes);

    KGEN::POP::DTypeValue dtypeValue(APFloat(semantic, loadedValue), kgenType);
    values[i] = KGEN::POP::SIMDAttr::get(dtypeValue, simdType);
  }

  return values;
}

llvm::SmallVector<mlir::TypedAttr> convertIntsToSIMD(DType moggType,
                                                     KGEN::KGENDType kgenType,
                                                     mlir::MLIRContext *context,
                                                     const void *data,
                                                     int64_t numElems) {
  auto simdType = KGEN::POP::SIMDType::get(context, 1, moggType);
  auto sizeInBytes = moggType.getSizeInBytes();

  llvm::SmallVector<mlir::TypedAttr> values;
  values.resize_for_overwrite(numElems);

  APInt loadedValue(sizeInBytes * CHAR_BIT, 0);

  for (auto i : llvm::seq(numElems)) {
    auto valuePtr = reinterpret_cast<const uint8_t *>(data) + sizeInBytes * i;
    llvm::LoadIntFromMemory(loadedValue, valuePtr, sizeInBytes);

    KGEN::POP::DTypeValue dtypeVal(loadedValue, kgenType);
    values[i] = KGEN::POP::SIMDAttr::get(dtypeVal, simdType);
  }

  return values;
}

} // namespace

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, ...)                              \
  static MOGGTypeTraits SHORT_NAME() {                                         \
    return MOGGTypeTraits(#LONG_NAME, KGEN::KGENDType::SHORT_NAME,             \
                          DType::SHORT_NAME);                                  \
  }
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

static MOGGTypeTraits ui8() {
  return MOGGTypeTraits("uint8", KGEN::KGENDType::ui8, DType::ui8);
}

static MOGGTypeTraits ui16() {
  return MOGGTypeTraits("uint16", KGEN::KGENDType::ui16, DType::ui16);
}

static MOGGTypeTraits ui32() {
  return MOGGTypeTraits("uint32", KGEN::KGENDType::ui32, DType::ui32);
}

static MOGGTypeTraits ui64() {
  return MOGGTypeTraits("uint64", KGEN::KGENDType::ui64, DType::ui64);
}

static MOGGTypeTraits si8() {
  return MOGGTypeTraits("int8", KGEN::KGENDType::si8, DType::si8);
}

static MOGGTypeTraits si16() {
  return MOGGTypeTraits("int16", KGEN::KGENDType::si16, DType::si16);
}

static MOGGTypeTraits si32() {
  return MOGGTypeTraits("int32", KGEN::KGENDType::si32, DType::si32);
}

static MOGGTypeTraits si64() {
  return MOGGTypeTraits("int64", KGEN::KGENDType::si64, DType::si64);
}

static MOGGTypeTraits b() {
  return MOGGTypeTraits("bool", KGEN::KGENDType::kBool, DType::kBool);
}

llvm::SmallVector<mlir::TypedAttr>
MOGGTypeTraits::convertToSIMD(mlir::MLIRContext *context, const void *data,
                              int64_t numElems) const {
  if (auto fltSemantics = moggType.getFloatSemantics())
    return convertFloatsToSIMD(moggType, kgenType, context, data, numElems,
                               *fltSemantics);

  if (moggType.isInt() || moggType.isBool())
    return convertIntsToSIMD(moggType, kgenType, context, data, numElems);

  llvm_unreachable("unimplemented DType");
}

llvm::SmallVector<MOGGTypeTraits> M::MOGG::allTypeTraits() {
  return {
#define DECLARE_FLOAT(SHORT_NAME, ...) SHORT_NAME(),
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

      ui8(), ui16(), ui32(), ui64(), si8(), si16(), si32(), si64(), b()};
}

ErrorOr<MOGGTypeTraits> MOGGTypeTraits::lookupType(Type type) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  if (isa<MLIR_TYPE>(type))                                                    \
    return SHORT_NAME();
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  if (type.isUnsignedInteger(8))
    return ui8();
  if (type.isUnsignedInteger(16))
    return ui16();
  if (type.isUnsignedInteger(32))
    return ui32();
  if (type.isUnsignedInteger(64))
    return ui64();
  if (type.isSignedInteger(8))
    return si8();
  if (type.isSignedInteger(16))
    return si16();
  if (type.isSignedInteger(32))
    return si32();
  if (type.isSignedInteger(64))
    return si64();
  if (type.isInteger(1))
    return b();

  std::string typeStr;
  llvm::raw_string_ostream rso(typeStr);
  type.print(rso);
  return Error("No MOGGTypeTraits implementation for " + rso.str() +
               ". This is not a supported MOGG DType.");
}

ErrorOr<MOGGTypeTraits> MOGGTypeTraits::lookupDType(DType dtype) {
  switch (dtype.getValue()) {

#define DECLARE_FLOAT(SHORT_NAME, ...)                                         \
  case DType::SHORT_NAME:                                                      \
    return SHORT_NAME();
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  case DType::ui8:
    return ui8();
  case DType::ui16:
    return ui16();
  case DType::ui32:
    return ui32();
  case DType::ui64:
    return ui64();
  case DType::si8:
    return si8();
  case DType::si16:
    return si16();
  case DType::si32:
    return si32();
  case DType::si64:
    return si64();
  case DType::kBool:
    return b();
  default:
    return Error("No MOGGTypeTraits implementation for " + dtype.getAsString() +
                 ". This is not a supported MOGG DType.");
  }
}

bool MOGGTypeTraits::isSupportedMOGGType(Type type) {
  return !lookupType(type).isError();
}
