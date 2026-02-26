//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file centralizes all metal-cpp includes with proper warning
// suppressions.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_METALCPP_H
#define TARGET_LIB_METAL_METALCPP_H

#include "Support/PlatformUtils.h"

#if MODULAR_MACOS

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wundef"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#endif // MODULAR_MACOS

#endif // TARGET_LIB_METAL_METALCPP_H
