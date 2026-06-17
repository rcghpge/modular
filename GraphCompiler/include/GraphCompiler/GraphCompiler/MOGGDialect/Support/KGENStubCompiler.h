//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"

#include "Support/Buffer.h"
#include "Support/MDialect/MAttrs.h"

#include <string>

namespace M::MOGG {

/// Create a DenseResourceElementsAttr from a buffer. The name is used as part
/// of the attribute name.
DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name);

/// Get the resolved target info: the current host by default, or the
/// `set_virtual_cpu_target` override (bare arch or descriptor) when set.
ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx);

/// A resolved (de-sugared, validated) virtual-CPU-target request, letting a
/// host-warmed cache target a different CPU or even architecture.
struct VirtualCpuTargetSpec {
  /// Empty => use the host triple.
  std::string triple;
  /// Resolved CPU model name (de-sugared from "generic").
  std::string cpu;
  /// Verbatim feature delta; only meaningful when `hasExplicitFeatures`.
  std::string features;
  /// True: use `features` verbatim. False: derive name-only from `cpu` (no host
  /// CPUID leak); this is the legacy bare-arch behavior.
  bool hasExplicitFeatures = false;
};

/// Parse a `set_virtual_cpu_target` knob into a resolved spec. Accepts either a
/// bare arch name (legacy: host triple kept, features name-only) or a
/// descriptor "triple=<t>;cpu=<c>;features=<f>" (any order; `cpu` required).
/// `hostTriple` de-sugars "generic" and defaults a missing descriptor triple.
/// Rejects "native". Precondition: `knob` is non-empty (empty = host).
ErrorOr<VirtualCpuTargetSpec> parseVirtualCpuTarget(StringRef knob,
                                                    StringRef hostTriple);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
