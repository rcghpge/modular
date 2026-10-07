//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Target/mcl/Metal/MetalTraits.h"

namespace M::KGEN {

const MetalTraits &MetalTraits::get() {
  static const MetalTraits instance;
  return instance;
}

static constexpr TargetTraits::AcceleratorArch kMetalAcceleratorArchs[] = {
    {"apple-m1", "Apple M1"},
    {"apple-m1-metal4", "Apple M1 (Metal 4.0 / macOS 26+)"},
    {"apple-m2", "Apple M2"},
    {"apple-m2-metal4", "Apple M2 (Metal 4.0 / macOS 26+)"},
    {"apple-m3", "Apple M3"},
    {"apple-m3-metal4", "Apple M3 (Metal 4.0 / macOS 26+)"},
    {"apple-m4", "Apple M4"},
    {"apple-m4-metal4", "Apple M4 (Metal 4.0 / macOS 26+)"},
    {"apple-m5", "Apple M5"},
    {"apple-m5-metal4", "Apple M5 (Metal 4.0 / macOS 26+)"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
MetalTraits::supportedAcceleratorArchs() const {
  return kMetalAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<MetalTraits> registerMetalTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
