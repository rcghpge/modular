//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_METAL_METALADDRESSSPACE_H
#define KGEN_COMPILER_TARGET_METAL_METALADDRESSSPACE_H

namespace M::KGEN {

/// Metal-specific LLVM address spaces.
enum MetalAddressSpace : unsigned {
  None = 0,
  Device = 1,
  Constant = 2,
  ThreadGroup = 3,
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_METAL_METALADDRESSSPACE_H
