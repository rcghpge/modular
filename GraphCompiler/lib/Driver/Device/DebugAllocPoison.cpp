//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DebugAllocPoison.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>

using namespace M;

std::optional<M::GraphCompiler::Driver::PoisonPattern>
M::GraphCompiler::Driver::getPoisonPattern(const TensorSpec &spec) {
  DType dtype = spec.getEltType();
  size_t totalBytes = spec.getSizeInBytes();

  if (const llvm::fltSemantics *sem = dtype.getFloatSemantics()) {
    if (llvm::APFloat::getSizeInBits(*sem) < 8) {
      llvm::errs() << "DEBUG_ALLOC_POISON: skipping " << dtype
                   << " allocation (" << totalBytes
                   << " bytes; sub-byte float type)\n";
      return std::nullopt;
    }
    if (llvm::is_contained(
            std::initializer_list<DType>{DType::f8e3m4, DType::f8e8m0fnu,
                                         DType::f8e4m3fn, DType::f8e4m3fnuz,
                                         DType::f8e5m2, DType::f8e5m2fnuz},
            dtype)) {
      llvm::errs() << "DEBUG_ALLOC_POISON: skipping " << dtype
                   << " allocation (" << totalBytes
                   << " bytes; not in Mojo poison-check set)\n";
      return std::nullopt;
    }
    llvm::APFloat largest = llvm::APFloat::getLargest(*sem);
    return PoisonPattern{largest.bitcastToAPInt().getZExtValue(),
                         llvm::APFloat::getSizeInBits(*sem) / 8,
                         "largest-finite"};
  }

  // Integer types. Every float dtype we support has LLVM APFloat semantics
  // (see `Support/include/Support/ML/FloatTypes.def`), so a future float type
  // without semantics is the only way to land here as a float — at which
  // point this code needs to grow a fill strategy for it rather than
  // silently fall through to 0xCD.
  assert(!dtype.isFloat() && "float dtype lacks LLVM APFloat semantics");
  return PoisonPattern{0xCD, 1, "byte-fill"};
}

void M::GraphCompiler::Driver::poisonUninitializedMemory(
    void *ptr, const TensorSpec &spec) {
  std::optional<PoisonPattern> pattern = getPoisonPattern(spec);
  if (!pattern)
    return;

  size_t totalBytes = spec.getSizeInBytes();
  llvm::errs() << "DEBUG_ALLOC_POISON: Filling host " << spec.getEltType()
               << " allocation (" << totalBytes << " bytes) with "
               << pattern->patternKind << " pattern ";
  (llvm::errs().changeColor(llvm::raw_ostream::RED)
   << "0x"
   << llvm::format_hex_no_prefix(pattern->bits, pattern->patternBytes * 2))
      .resetColor();
  llvm::errs() << "\n";

  uint8_t *dst = static_cast<uint8_t *>(ptr);
  if (pattern->patternBytes == 1) {
    std::memset(dst, static_cast<uint8_t>(pattern->bits), totalBytes);
  } else {
    // Replicate the patternBytes-byte little-endian pattern across the
    // buffer. We avoid `std::fill_n` through a `uint{16,32,64}_t *` because
    // the caller's allocation does not guarantee `alignof(T)` for T larger
    // than a byte, and a misaligned typed store is UB. The `bits` argument to
    // `memcpy` is taken by address but read as a contiguous byte sequence, so
    // no alignment requirement applies.
    for (size_t off = 0; off + pattern->patternBytes <= totalBytes;
         off += pattern->patternBytes) {
      std::memcpy(dst + off, &pattern->bits, pattern->patternBytes);
    }
  }
}
