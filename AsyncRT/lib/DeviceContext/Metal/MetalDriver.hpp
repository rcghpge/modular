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

#ifndef ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDRIVER_HPP
#define ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDRIVER_HPP

#include "Support/PlatformUtils.h"

#ifdef MODULAR_MACOS

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#endif // MODULAR_MACOS

#endif // ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDRIVER_HPP
