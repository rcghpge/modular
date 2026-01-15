//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file instantiates the different Metal C++ templates.
//
//===----------------------------------------------------------------------===//

#include "Support/PlatformUtils.h"

#if MODULAR_MACOS

// Generate the metal-cpp implementations here.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wundef"

#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#endif // MODULAR_MACOS
