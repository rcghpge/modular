//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "SDK/DriverAPI/python/DType.h"

using namespace M;

DType M::Driver::Python::convertIntToDType(int dtype) {
  switch (dtype) {
  case 0:
    return DType::kBool;
  case 4:
    return DType::si8;
  case 5:
    return DType::si16;
  case 6:
    return DType::si32;
  case 7:
    return DType::si64;
  case 15:
    return DType::ui8;
  case 16:
    return DType::ui16;
  case 17:
    return DType::ui32;
  case 18:
    return DType::ui64;
  case 25:
    return DType::f16;
  case 26:
    return DType::f32;
  case 27:
    return DType::f64;
  case 33:
    return DType::bf16;
  default:
    return DType::invalid;
  }
}

int M::Driver::Python::convertDTypeToInt(DType dtype) {
  switch (dtype.getValue()) {
  case DType::kBool:
    return 0;
  case DType::si8:
    return 4;
  case DType::si16:
    return 5;
  case DType::si32:
    return 6;
  case DType::si64:
    return 7;
  case DType::ui8:
    return 15;
  case DType::ui16:
    return 16;
  case DType::ui32:
    return 17;
  case DType::ui64:
    return 18;
  case DType::f16:
    return 25;
  case DType::f32:
    return 26;
  case DType::f64:
    return 27;
  case DType::bf16:
    return 33;
  default:
    return -1;
  }
}
