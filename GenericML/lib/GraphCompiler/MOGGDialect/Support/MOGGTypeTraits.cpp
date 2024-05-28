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
#include "Support/AssertStream.h"
#include "llvm/ADT/APFloat.h"

namespace M::MOGG {
namespace {

template <typename T>
llvm::SmallVector<mlir::TypedAttr>
convertFloatsToSIMD(DType moggType, KGEN::KGENDType kgenType,
                    mlir::MLIRContext *context, const void *data,
                    int64_t numElems, const llvm::fltSemantics &semantic) {
  constexpr int numBits = sizeof(T) * CHAR_BIT;
  static_assert(std::is_integral_v<T>, "Use uintXX_t types as storage.");
  static_assert(std::is_unsigned_v<T>, "Use uintXX_t types as storage.");
  static_assert(numBits <= 64, "Only up to 64 bit floats supported right now.");

  llvm::SmallVector<mlir::TypedAttr> values;
  const T *raw = reinterpret_cast<const T *>(data);
  for (int64_t i = 0; i < numElems; ++i) {
    uint64_t upcast = raw[i];
    values.push_back(M::KGEN::POP::SIMDAttr::get(
        {APFloat(semantic, APInt(numBits, upcast)), kgenType},
        KGEN::POP::SIMDType::get(context, 1, moggType)));
  }
  return values;
}

template <typename T>
llvm::SmallVector<mlir::TypedAttr>
convertIntsToSIMD(DType moggType, KGEN::KGENDType kgenType,
                  mlir::MLIRContext *context, const void *data,
                  int64_t numElems) {
  // Retrieving metadata
  size_t typeBits = sizeof(T) * CHAR_BIT;
  size_t isTypeSigned = std::is_signed_v<T>;

  // Creating simd attributes
  auto asT = reinterpret_cast<const T *>(data);
  llvm::SmallVector<mlir::TypedAttr> values;
  for (int64_t i = 0; i < numElems; ++i) {
    // Convert the type for the APInt constructor.
    int64_t upcast = asT[i];
    uint64_t bitcast = *reinterpret_cast<uint64_t *>(&upcast);
    APInt apInt(typeBits, bitcast, isTypeSigned);
    auto kgenDtype = KGEN::KGENDType::getInt(typeBits, isTypeSigned).value();
    KGEN::POP::DTypeValue dtypeVal(apInt, kgenDtype);
    auto simdType = KGEN::POP::SIMDType::get(context, 1, moggType);
    values.push_back(KGEN::POP::SIMDAttr::get(dtypeVal, simdType));
  }
  return values;
}

llvm::SmallVector<mlir::TypedAttr>
convertBoolsToSIMD(DType moggType, KGEN::KGENDType kgenType,
                   mlir::MLIRContext *context, const void *data,
                   int64_t numElems) {
  size_t typeBits = sizeof(bool) * CHAR_BIT;

  // Creating simd attributes
  auto asT = reinterpret_cast<const bool *>(data);
  llvm::SmallVector<mlir::TypedAttr> values;
  for (int64_t i = 0; i < numElems; ++i) {
    // Convert the type for the APInt constructor.
    int64_t upcast = asT[i];
    uint64_t bitcast = *reinterpret_cast<uint64_t *>(&upcast);
    APInt apInt(typeBits, bitcast, false);
    KGEN::POP::DTypeValue dtypeVal(apInt, KGEN::KGENDType::kBool);
    auto simdType = KGEN::POP::SIMDType::get(context, 1, moggType);
    values.push_back(KGEN::POP::SIMDAttr::get(dtypeVal, simdType));
  }
  return values;
}

MOGGTypeTraits bf16() {
  return MOGGTypeTraits("bfloat16", KGEN::KGENDType::bf16, DType::bf16);
}

MOGGTypeTraits f16() {
  return MOGGTypeTraits("float16", KGEN::KGENDType::f16, DType::f16);
}

MOGGTypeTraits f32() {
  return MOGGTypeTraits("float32", KGEN::KGENDType::f32, DType::f32);
}

MOGGTypeTraits f64() {
  return MOGGTypeTraits("float64", KGEN::KGENDType::f64, DType::f64);
}

MOGGTypeTraits ui8() {
  return MOGGTypeTraits("uint8", KGEN::KGENDType::ui8, DType::ui8);
}

MOGGTypeTraits ui16() {
  return MOGGTypeTraits("uint16", KGEN::KGENDType::ui16, DType::ui16);
}

MOGGTypeTraits ui32() {
  return MOGGTypeTraits("uint32", KGEN::KGENDType::ui32, DType::ui32);
}

MOGGTypeTraits ui64() {
  return MOGGTypeTraits("uint64", KGEN::KGENDType::ui64, DType::ui64);
}

MOGGTypeTraits si8() {
  return MOGGTypeTraits("int8", KGEN::KGENDType::si8, DType::si8);
}

MOGGTypeTraits si16() {
  return MOGGTypeTraits("int16", KGEN::KGENDType::si16, DType::si16);
}

MOGGTypeTraits si32() {
  return MOGGTypeTraits("int32", KGEN::KGENDType::si32, DType::si32);
}

MOGGTypeTraits si64() {
  return MOGGTypeTraits("int64", KGEN::KGENDType::si64, DType::si64);
}

MOGGTypeTraits b() {
  return MOGGTypeTraits("bool", KGEN::KGENDType::kBool, DType::kBool);
}

} // namespace

llvm::SmallVector<mlir::TypedAttr>
MOGGTypeTraits::convertToSIMD(mlir::MLIRContext *context, const void *data,
                              int64_t numElems) const {
  if (moggType == DType::f16)
    return convertFloatsToSIMD<uint16_t>(moggType, kgenType, context, data,
                                         numElems, APFloat::IEEEhalf());
  else if (moggType == DType::bf16)
    return convertFloatsToSIMD<uint16_t>(moggType, kgenType, context, data,
                                         numElems, APFloat::BFloat());
  else if (moggType == DType::f32)
    return convertFloatsToSIMD<uint32_t>(moggType, kgenType, context, data,
                                         numElems, APFloat::IEEEsingle());
  else if (moggType == DType::f64)
    return convertFloatsToSIMD<uint64_t>(moggType, kgenType, context, data,
                                         numElems, APFloat::IEEEdouble());
  else if (moggType == DType::ui8)
    return convertIntsToSIMD<uint8_t>(moggType, kgenType, context, data,
                                      numElems);
  else if (moggType == DType::ui16)
    return convertIntsToSIMD<uint16_t>(moggType, kgenType, context, data,
                                       numElems);
  else if (moggType == DType::ui32)
    return convertIntsToSIMD<uint32_t>(moggType, kgenType, context, data,
                                       numElems);
  else if (moggType == DType::ui64)
    return convertIntsToSIMD<uint64_t>(moggType, kgenType, context, data,
                                       numElems);
  else if (moggType == DType::si8)
    return convertIntsToSIMD<int8_t>(moggType, kgenType, context, data,
                                     numElems);
  else if (moggType == DType::si16)
    return convertIntsToSIMD<int16_t>(moggType, kgenType, context, data,
                                      numElems);
  else if (moggType == DType::si32)
    return convertIntsToSIMD<int32_t>(moggType, kgenType, context, data,
                                      numElems);
  else if (moggType == DType::si64)
    return convertIntsToSIMD<int64_t>(moggType, kgenType, context, data,
                                      numElems);
  else if (moggType == DType::kBool)
    return convertBoolsToSIMD(moggType, kgenType, context, data, numElems);
  GENERICML_ASSERT(false) << "Unimplemented DType " << moggType;
}

llvm::SmallVector<MOGGTypeTraits> allTypeTraits() {
  return {f16(),  bf16(), f32(),  f64(),  ui8(),  ui16(), ui32(),
          ui64(), si8(),  si16(), si32(), si64(), b()};
}

ErrorOr<MOGGTypeTraits> MOGGTypeTraits::lookupType(Type type) {
  if (type.isF16()) {
    return f16();
  } else if (type.isBF16()) {
    return bf16();
  } else if (type.isF32()) {
    return f32();
  } else if (type.isF64()) {
    return f64();
  } else if (type.isUnsignedInteger(8)) {
    return ui8();
  } else if (type.isUnsignedInteger(16)) {
    return ui16();
  } else if (type.isUnsignedInteger(32)) {
    return ui32();
  } else if (type.isUnsignedInteger(64)) {
    return ui64();
  } else if (type.isSignedInteger(8)) {
    return si8();
  } else if (type.isSignedInteger(16)) {
    return si16();
  } else if (type.isSignedInteger(32)) {
    return si32();
  } else if (type.isSignedInteger(64)) {
    return si64();
  } else if (type.isInteger(1)) {
    return b();
  } else {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    type.print(rso);
    return Error("No MOGGTypeTraits implementation for " + rso.str() +
                 ". This is not a supported MOGG DType.");
  }
}

ErrorOr<MOGGTypeTraits> MOGGTypeTraits::lookupDType(DType dtype) {
  if (dtype == DType::f16)
    return f16();
  else if (dtype == DType::bf16)
    return bf16();
  else if (dtype == DType::f32)
    return f32();
  else if (dtype == DType::f64)
    return f64();
  else if (dtype == DType::ui8)
    return ui8();
  else if (dtype == DType::ui16)
    return ui16();
  else if (dtype == DType::ui32)
    return ui32();
  else if (dtype == DType::ui64)
    return ui64();
  else if (dtype == DType::si8)
    return si8();
  else if (dtype == DType::si16)
    return si16();
  else if (dtype == DType::si32)
    return si32();
  else if (dtype == DType::si64)
    return si64();
  else if (dtype == DType::kBool)
    return b();
  else
    return Error("No MOGGTypeTraits implementation for " + dtype.getAsString() +
                 ". This is not a supported MOGG DType.");
}

bool MOGGTypeTraits::isSupportedMOGGType(Type type) {
  return !lookupType(type).isError();
}

ErrorOr<Type> MOGGTypeTraits::simdTypeForTensor(Type type) {
  auto asTensor = cast<MO::TensorType>(type);
  Type dtype = asTensor.getElementType();
  auto traitsOr = lookupType(dtype);
  if (traitsOr.isError())
    return traitsOr.takeError();
  return {MOGG::SimdType::get(type.getContext(), traitsOr->moggType)};
}

} // namespace M::MOGG
